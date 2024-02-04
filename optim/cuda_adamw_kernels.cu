
#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/Exceptions.h>

#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

using torch::Tensor;

template <typename T>
__global__ void kernel_cuda_single_tensor(
        T* __restrict__ p,
        const T * __restrict__ g,
        T* __restrict__ exp_avg,
        T* __restrict__ exp_avg_sq,

        const float beta1,
        const float beta2,
        const float lr,
        const float weight_decay,
        const float eps,
        const float step,
        const size_t total_size)
{
        const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_id >= total_size) return;

        float curr_grad = g[global_id];

        //decoupled weight decay
        p[global_id] = p[global_id] * (1 - lr * weight_decay);


        exp_avg[global_id] = beta1 * exp_avg[global_id] + (1 - beta1) * curr_grad;
        exp_avg_sq[global_id] = beta2 * exp_avg_sq[global_id] + (1 - beta2) * (curr_grad * curr_grad);

        const float correction1 = 1.0f - powf(beta1, step);
        const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));
        float step_size = lr / correction1;
        /*

        step_size = lr / correction1
        denom = ((tl.sqrt(exp_avg_sq_val) / correction2_sqrt) + eps) # * correction1
        update = (exp_avg_val / denom)
        # weight update
        p_val = p_val - step_size * update

        */

        float denom = (sqrtf(exp_avg_sq[global_id]) / correction2_sqrt + eps); // * correction1;
        float update = (exp_avg[global_id]/denom); // + (weight_decay * p[global_id]);
        p[global_id] = p[global_id] - (step_size * update);
}

// interface and launcher for fused adamw cuda kernel
void cuda_fused_single_tensor(Tensor& p, Tensor& g, Tensor& exp_avg, Tensor& exp_avg_sq,
                      float beta1, float beta2, float lr, float weight_decay, float eps, float step) {
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");

    const int block_dim = 128;
    int grid_dim = ((total_size + block_dim - 1) / block_dim);
    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "cuda_fused_single_tensor", ([&] {
        kernel_cuda_single_tensor<scalar_t><<<blocks, block_dim>>>(
            p.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            exp_avg_sq.data_ptr<scalar_t>(),
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
            step,
            total_size
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}

// local vs global max
__device__ __forceinline__ float atomicPosMax(float * addr, float value) {

    return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
}

// binary search for quantization
__device__ __forceinline__ float q_mapping( const float* __restrict__ qmap,
                                            const float* __restrict__ qmidpt,
                                            float x)
{
    // 4 bit range
    int low = 0;
    int high = 15;

    if (x <= qmap[low]) return low;
    if (qmap[high] <=x) return high;

    while (low < high) {
        int mid = (low + high) >> 1;
        if (qmap[mid] <= x)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }

    return (qmidpt[low-1] < x) ? low : low-1;

}



template <typename T>
__global__ void cuda_fused_4bit_kernel(
    T* __restrict__ p,
    const T* __restrict__ g,
    int8_t* __restrict__ exp,
    int8_t* __restrict__ sq,
    T* __restrict__ exp_qscale,
    T* __restrict__ sq_qscale,
    const float* __restrict__ exp_qmap,
    const float* __restrict__ exp_qmidpt,
    const float* __restrict__ sq_qmap,
    const float* __restrict__ sq_qmidpt,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float step,
    const size_t total_size,
    const float correction1,
    const float correction2_sqrt,
    const float step_size


)
{
    const int thread_id = threadIdx.x;
    const int global_id = blockIdx.x * blockDim.x + thread_id;
    const int block_id = blockIdx.x;


    const int left_id = global_id << 1;
    const int right_id = (global_id << 1) + 1;

    __shared__ float absmax_exp;
    __shared__ float absmax_sq;

    if (thread_id == 0) {
        absmax_exp = 0;
        absmax_sq = 0;
    }
    __syncthreads();


    if (left_id >= total_size) return;

    // universal processing
    const int8_t bitmask = 15; //(1 << 4) -1;


    // left side processing
    const int8_t exp_left_index = (exp[global_id]) & bitmask;
    const int8_t sq_left_index = (sq[left_id]) & bitmask;


    //decoupled weight decay
    p[left_id] = p[left_id] * (1 - lr * weight_decay);

    // left exp and sq updates
    float curr_grad = g[left_id];
    float exp_avg_qscale = exp_qscale[block_id];

    T exp_left = (T)exp_qmap[exp_left_index] * exp_avg_qscale;
    exp_left = beta1 * exp_left + (1 - beta1) * curr_grad;

    T sq_left = (T)sq_qmap[sq_left_index] * sq_qscale[block_id];
    sq_left = beta2 * sq_left + (1 - beta2) * (curr_grad * curr_grad);

    float denom = (sqrtf(sq_left) / correction2_sqrt + eps);
    float update = (exp_left/denom);

    // param update
    p[left_id] = p[left_id] - (step_size * update);


    // right side processing
    T exp_right =0;
    T sq_right = 0;

    if (right_id < total_size) {
        const int8_t exp_right_index = (exp[global_id] >> 4) & bitmask;
        const int8_t sq_right_index = (sq[global_id]>>4) & bitmask;
        curr_grad = g[right_id];

        //decoupled weight decay, right side
        p[right_id] = p[right_id] * (1 - lr * weight_decay);

        exp_right = (T)exp_qmap[exp_right_index] * exp_avg_qscale;
        exp_right = beta1 * exp_right + (1-beta1) * curr_grad;

        sq_right = (T)sq_qmap[sq_right_index] * sq_qscale[block_id];
        sq_right = beta2 * sq_right + (1 - beta2) * (curr_grad * curr_grad);

        denom = (sqrtf(sq_right) / correction2_sqrt + eps);
        update = (exp_right/denom);

        // param update
        p[right_id] = p[right_id] - (step_size * update);

        }

    // prepare quantization info - update absmax scales
    float local_absmax_exp = fmax(fabsf((float)exp_left), fabsf((float)exp_right));
    float local_absmax_sq = fmaxf((float)sq_left, (float)sq_right);

    atomicPosMax(&absmax_exp, local_absmax_exp);
    atomicPosMax(&absmax_sq, local_absmax_sq);
    __syncthreads();

    int8_t local_packed_exp = 0;
    int8_t local_packed_sq = 0;

    // quantize and pack
    const int8_t q_exp_left = (int8_t)q_mapping(exp_qmap, exp_qmidpt, (float)exp_left / absmax_exp);
    const int8_t q_sq_left = (int8_t)q_mapping(sq_qmap, sq_qmidpt, (float)sq_left / absmax_sq);
    local_packed_exp |= (q_exp_left & bitmask);
    local_packed_sq |= (q_sq_left & bitmask);

    if (right_id < total_size) {
        const int8_t q_exp_right = (int8_t)q_mapping(exp_qmap, exp_qmidpt, (float)exp_right / absmax_exp);
        const int8_t q_sq_right = (int8_t)q_mapping(sq_qmap, sq_qmidpt, (float)sq_right / absmax_sq);
        local_packed_exp |= (q_exp_right & bitmask << 4);
        local_packed_sq |= (q_sq_right & bitmask << 4);

    }

    // store updated exp and sq
    exp[global_id] = local_packed_exp;
    sq[global_id] = local_packed_sq;

    if (thread_id == 0) {
        exp_qscale[block_id] = (T)absmax_exp;
        sq_qscale[block_id] = (T)absmax_sq;
    }
    __syncthreads();

}

// interface and launcher for 4bit quantized cuda kernel
void cuda_fused_4bit(Tensor& p, Tensor& g,
                        Tensor& exp, Tensor& sq,
                        Tensor& exp_scale, Tensor& sq_scale,
                        Tensor& exp_qmap, Tensor& exp_qmidpt,
                        Tensor& sq_qmap, Tensor& sq_qmidpt,
                        float beta1, float beta2,
                        float lr, float weight_decay,
                        float eps, float step
                        ){

    int total_size = p.numel();
    const int block_size = 128;
    int grid = ((total_size + block_size -1) / block_size);
    const dim3 blocks(grid);
    //universal computations
    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));
    const float step_size = lr / correction1;


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "cuda_fused_4bit", ([&] {
        cuda_fused_4bit_kernel<scalar_t><<<blocks, block_size/2>>>(
            p.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            exp.data_ptr<int8_t>(),
            sq.data_ptr<int8_t>(),
            exp_scale.data_ptr<scalar_t>(),
            sq_scale.data_ptr<scalar_t>(),
            exp_qmap.data_ptr<float>(),
            exp_qmidpt.data_ptr<float>(),
            sq_qmap.data_ptr<float>(),
            sq_qmidpt.data_ptr<float>(),
            beta1,
            beta2,
            lr,
            weight_decay,
            eps,
            step,
            total_size,
            correction1,
            correction2_sqrt,
            step_size

        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());
}
