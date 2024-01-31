
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


void cuda_fused_single_tensor(Tensor& p, Tensor& g, Tensor& exp_avg, Tensor& exp_avg_sq,
                      float beta1, float beta2, float lr, float weight_decay, float eps, float step) {
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");

    const int block_dim = 128;
    int grid_dim = ((total_size + block_dim - 1) / block_dim);
    const dim3 blocks(grid_dim);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(p.scalar_type(), "kernel_cuda_single_tensor", ([&] {
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


__device__ __forceinline__ float atomicMax(float * addr, float value) {

    return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
}
