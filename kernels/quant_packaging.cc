// Cuda functions for quantization

#include <torch/extension.h>
#include <torch/torch.h>

#include 'common.h'

using torch::Tensor;
using torch::autograd::Function;

// Pack
Tensor pack_nonlinear_cuda(Tensor data, Tensor qmap);

// Unpack
Tensor unpack_nonlinear_cuda(Tensor data, Tensor qmap, int64_t num_groups, int64_t group_size);


Tensor pack_nonlinear(Tensor data, Tensor qmap) {
    CHECK_CUDA_TENSOR_DIM_FLOAT(data,2)
    CHECK_CUDA_TENSOR_DIM_FLOAT(qmap,1)

    return pack_nonlinear_cuda(data, qmap)
}

Tensor unpack_nonlinear(Tensor data, Tensor qmap, int64_t num_groups,
                        int64_t group_size) {
                        CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);
                        CHECK_CUDA_TENSOR_DIM_FLOAT(qmap, 1);

                        return unpack_nonlinear_cuda(data, qmap, num_groups, group_size);
                        }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_nonlinear", &pack_nonlinear);
    m.def("unpack_nonlinear", &unpack_nonlinear);
}
