// based on lpmm cuda impl
#include <torch/torch.h>
#include <torch/extension.h>

using torch::Tensor;

// declarations
void cuda_fused_4bit(Tensor& p, Tensor& g,
                        Tensor& exp, Tensor& sq,
                        Tensor& exp_scale, Tensor& sq_scale,
                        Tensor& exp_qmap, Tensor& exp_qmidpt,
                        Tensor& sq_qmap, Tensor& sq_qmidpt,
                        float beta1, float beta2,
                        float lr, float weight_decay,
                        float eps, float step
                        );



// python interface for quantized 4bit AdamW
void fused_4bit(Tensor& p, Tensor& g,
                Tensor& exp, Tensor& sq,
                Tensor& exp_scale, Tensor& sq_scale,
                Tensor& exp_qmap, Tensor& exp_qmidpt,
                Tensor& sq_qmap, Tensor& sq_qmidpt,
                float beta1, float beta2,
                float lr, float weight_decay,
                float eps, float step
)
 {

    // call the cuda kernel
    printf("about to launch:");

    cuda_fused_4bit(p, g,
                    exp, sq,
                    exp_scale, sq_scale,
                    exp_qmap, exp_qmidpt,
                    sq_qmap, sq_qmidpt,
                    beta1, beta2,
                    lr, weight_decay,
                    eps, step
                    );
    printf("back from fused kernel!");

}

void cuda_fused_single_tensor(Tensor& p, Tensor& g,
                        Tensor& exp_avg,Tensor& exp_avg_sq,
                        float beta1, float beta2,
                        float lr, float weight_decay,
                        float eps, float step);


// python interface for single tensor AdamW
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
    m.def("fused_4bit", &fused_4bit);
    m.def("fused_single_tensor", &fused_single_tensor);
}
