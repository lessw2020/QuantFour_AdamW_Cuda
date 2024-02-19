import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
import sys

from quantfour_cuda import fused_4bit

__all__ = ["AdamW_Fused_QuantFour"]

def lprint(msg=""):
  print(f"Debug ++> {sys._getframe().f_back.f_lineno}: {msg}")


def enable_param_quantization(p, threshold) -> bool:
    """ only enable quantization if the parameter is large enough """
    if threshold and p.numel() <= threshold:
        return False
    return True


class AdamWFused_QuantFour(torch.optim.Optimizer):
    """Fused 4bit AdamW in CUDA """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        fused: Optional[bool] = False,
    ):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr=}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid eps value: {eps=}")
        if not 0.0 < betas[0] < 1.0:
            raise ValueError(f"Invalid Beta[0]: {betas[0]=}")
        if not 0.0 < betas[1] < 1.0:
            raise ValueError(f"Invalid Beta[1]: {betas[1]=}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay=}")

        self.param_quant_threshold = 2 # 128 # TODO - adjust this...forcing it to 2 for testing

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
        )
        super().__init__(params, defaults)


    def init_qstate(self, p, state_name):
        state = self.state[p]
        field = f"{state_name}_qstate"
        lprint(f"{field=}")
        state[field] = {
            "enable": True,
            "overhead": dict(),
            "qmap": None,
        }
        state[field]["enable"] = enable_param_quantization(p, self.param_quant_threshold)


    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(
            state_values[0]["step"]
        )

        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))


    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avgs_sqs,
        state_steps,
        momentum_meta,
        variance_meta,
        momentum_quant_enabled,

    ):

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("QuantFour AdamW does not support sparse gradients")
            grads.append(p.grad)
            params_with_grad.append(p)
            state = self.state[p]

            # lazy init state ------
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros((), dtype=torch.float, device=p.device)
                self.init_qstate(p, "momentum")

                state["exp_avg_sq"] = torch.zeros(
                    (), dtype=torch.float, device=p.device
                )
                self.init_qstate(p, "variance")
            # ------ end state init

            state_steps.append(state["step"])
            exp_avgs.append(state["exp_avg"])
            exp_avgs_sqs.append(state["exp_avg_sq"])

            momentum_meta.append(state["momentum_qstate"]["overhead"])
            variance_meta.append(state["variance_qstate"]["overhead"])
            momentum_quant_enabled.append(state["momentum_qstate"]["enable"])


    @torch.no_grad()
    def step(self, closure=None):
        """single optimization step"""

        loss = None
        if closure:
            with torch.enable_grad:
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            momentum_meta = []
            variance_meta = []
            momentum_quant_enabled = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                momentum_meta,
                variance_meta,
                momentum_quant_enabled
            )

            # settings

            lr=group["lr"]
            weight_decay=group["weight_decay"]
            eps=group["eps"]


            # step processing
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                q_exp_avg = exp_avgs[i]
                q_exp_avg_sq = exp_avg_sqs[i]
                t_step = state_steps[i]

                # update step
                t_step += 1

                if momentum_quant_enabled[i]:
                    p_num_elem = param.numel()

                    bytelength = p_num_elem # todo - undo this.... (p_num_elem + 1) // 2
                    blocks = (p_num_elem + 127) // 128
                    curr_dtype = torch.int8

                    if q_exp_avg.numel() <= 1:
                        # q_exp_avg.data = exp_avg = torch.zeros_like(
                        # param, memory_format=torch.preserve_format
                        # )
                        q_exp_avg.data = torch.zeros((bytelength,), dtype=curr_dtype, device=param.device)

                    if q_exp_avg_sq.numel() <= 1:
                        q_exp_avg_sq.data = torch.zeros((bytelength,), dtype=curr_dtype, device=param.device)
                        #q_exp_avg_sq.data = exp_avg_sq = torch.zeros_like(
                        #param, memory_format=torch.preserve_format
                        #)

                    exp_avg_scale = torch.zeros((blocks,), dtype=torch.float32, device=param.device)
                    momentum_meta[i]["max1"] = exp_avg_scale


                    exp_avg_sq_scale = torch.zeros((blocks,), dtype=torch.float32, device=param.device)
                    variance_meta[i]["max1"] = exp_avg_sq_scale

                    # ==== control math =============
                    """p2 = param.clone().detach()
                    p2.mul_(1 - lr * weight_decay)


                    exp_avg2 = q_exp_avg.clone().detach()
                    exp_avg2_full = q_exp_avg.clone().detach()

                    lprint(f"{exp_avg2.shape=}, {grad.shape=}")
                    exp_avg_sq2 = q_exp_avg_sq.clone().detach()
                    lprint(f"{grad.shape=}")
                    exp_avg2.lerp_(grad, 1 - beta1)
                    exp_avg2_full = beta1 * exp_avg2 + (1 - beta1) * grad
                    torch.allclose(exp_avg2, exp_avg2_full, atol=1e-04, rtol=1e-0)


                    #exp_avg_val = beta1 * exp_avg_val + (1 - beta1) * g_val


                    lprint(f"expv update: {exp_avg2=}")
                    #lprint(f"{step_t=}, check first: {exp_avg2=}")

                    exp_avg_sq2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                    step = t_step.item()
                    bias_corr1 = 1 - beta1**step
                    bias_corr2 = 1 - beta2**step
                    lprint(f"{step=}, {bias_corr1=}, {bias_corr2=}")
                    step_size = lr / bias_corr1

                    bias_corr2_sqrt = math.sqrt(bias_corr2)

                    denom = (exp_avg_sq2.sqrt() / bias_corr2_sqrt).add_(eps)
                    lprint(f"321: {denom=}")
                    # lprint(f"321: {denom=}")
                    # weight update
                    # lprint(f"323: {param=}")
                    p3 = p2.clone().detach()
                    update = exp_avg2 / denom
                    lprint(f"{update=}")

                    p2 = p2 - step_size * update
                    p3.addcdiv_(exp_avg2, denom, value=-step_size)
                    assert torch.allclose(p2, p3, atol=1e-04, rtol=1e-0)
                    lprint(f"{p2=}, {p3=}")
                    #assert False, 'next check'
                    """
                    # start fused kernel here....
                    assert param.is_cuda, f"param must be on cuda"
                    assert param.is_contiguous(), f"param must be contiguous"
                    p_num_elem = param.numel()
                    # verify params numel matches relevant partners numel
                    assert q_exp_avg.numel() == p_num_elem, f"exp_avg numel {q_exp_avg.numel()} != param numel {p_num_elem}"
                    assert q_exp_avg_sq.numel() == p_num_elem, f"exp_avg_sq numel {q_exp_avg_sq.numel()} != param numel {p_num_elem}"
                    assert grad.numel() == p_num_elem, f"grad numel {grad.numel()} != param numel {p_num_elem}"

                    #lprint(f"calling triton fused kernel {q_exp_avg=}, {q_exp_avg_sq=}, {grad=}, {param=}, {t_step=}, {beta1=},")

                    #fused_4bit_triton_wrapper_starter(param, p_num_elem, grad, q_exp_avg, q_exp_avg_sq,
                    #                beta1, beta2, lr, weight_decay, eps, t_step)

                    # fused_single_tensor(param, grad, q_exp_avg, q_exp_avg_sq,
                    #             beta1, beta2, lr, weight_decay, eps, t_step)

                # momentum_meta.append(state["momentum_qstate"]["overhead"])
                # variance_meta.append(state["variance_qstate"]["overhead"])
                    # if "max1" in exp_avgs_q_overhead[i]:
                    exp_avg_scale = variance_meta[i]["max1"]
                    exp_avg_sq_scale = momentum_meta[i]["max1"]
                    lprint(f"types beta1{type(beta1)}, lr {type(lr)}, eps {type(eps)}, step {type(t_step)}")
                    lprint(f"types exp_avg_scale {type(exp_avg_scale)}")
                    lprint(f"types step {type(t_step.item())}, step {type(t_step)}, tstep {type(t_step)}, {type(lr)}")
                    lprint(f"{q_exp_avg.dtype=}, {q_exp_avg_sq.dtype=}, {grad.dtype=}, {param.dtype=},")

                    lprint(f"Start of 4bit! {exp_avg_scale=}, {exp_avg_sq_scale=}")
                    lprint(f"{param[0]=}")
                    lprint(f"{q_exp_avg[0:10]=}, {q_exp_avg_sq[0:10]=}")

                    # with torch.cuda.device(param.device):
                    fused_4bit(param, grad, q_exp_avg, q_exp_avg_sq,
                    exp_avg_scale, exp_avg_sq_scale,
                            beta1, beta2,
                            lr, weight_decay,
                            eps, t_step)

                    torch.cuda.synchronize(param.device)
                    lprint(f"Back from 4bit! {t_step=}, {exp_avg_scale=}, {exp_avg_sq_scale=}")
                    lprint(f"{param[0]=}")
                    lprint(f"{q_exp_avg[0:5]=}, {q_exp_avg_sq[0:5]=}")
                    # assert False, 'end of step check'
                    '''
                    lprint(f"back from cuda {beta1=}")
                    print(f"back from cuda {_momentum_qmap[0]=}")
                    lprint(f"back from cuda {grad[0]=}")
                    assert False, 'end of step check'
                    # p, g, exp_avg, exp_avg_sq,
                    # beta1, beta2, lr, weight_decay, eps, step

                    assert torch.allclose(exp_avg2, q_exp_avg, atol=1e-04, rtol=1e-0)
                    print(f"success with exp_avg! ")
                    assert torch.allclose(exp_avg_sq2, q_exp_avg_sq, atol=1e-04, rtol=1e-0)
                    print(f"success with exp_avg_sq! ")
                    #lprint(f"{p2[0][0:5]=}, check main param: {param[0][0:5]=}")
                    assert torch.allclose(p2, param, atol=1e-04, rtol=1e-0)
                    print(f"success with param! ")
                    #assert False, 'end of step check'
                    '''
