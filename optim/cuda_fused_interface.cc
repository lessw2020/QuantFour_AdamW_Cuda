// based on lpmm cuda impl
#include <torch/torch.h>
#include <torch/extension.h>

using torch::Tensor;

void cuda_fused_single_tensor(Tensor& p, Tensor& g,
                        Tensor& exp_avg,Tensor& exp_avg_sq,
                        float beta1, float beta2,
                        float lr, float weight_decay,
                        float eps, float step);



void fused_single_tensor(
    Tensor& p,
    Tensor& g,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    float step
) {

    // call the cuda kernel
    cuda_fused_single_tensor(p, g, exp_avg, exp_avg_sq,
                    beta1, beta2, lr, weight_decay, eps, step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_single_tensor", &fused_single_tensor);
}
