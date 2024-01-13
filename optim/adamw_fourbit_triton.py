import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from .quant_opt_base import create_dynamic_map, create_pow_map, create_qmap

__all__ = ["AdamW_QuantFour"]


@dataclass
class QuantParams:
    bits: int
    group_size: int
    scale_type: str
    quant_type: str
    round_type: str
    signed: bool
    threshold: int
    enable: bool = True


class FirstMoment(QuantParams):
    bits = 4
    group_size = 128
    scale_type = "group"
    quant_type = "nonlinear"
    round_type = "real-nearest"
    signed = True
    threshold = 4096


class SecondMoment(QuantParams):
    bits = 4
    group_size = 128
    scale_type = "rank1"
    quant_type = "power-1"
    round_type = "real-nearest"
    signed = False
    threshold = 4096


def _get_qenable_fn(p, threshold) -> bool:
    if threshold and p.numel() <= threshold:
        return False
    return True


class AdamW_QuantFour(torch.optim.Optimizer):
    """4bit AdamW with Triton fusion
    based on lpmm 4bit Optimizers"""

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

        # q specific
        if dist.is_initialized():
            seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
            dist.broadcast(seed, src=0)

        self.config_momentum = FirstMoment
        self.config_variance = SecondMoment
        self.qmaps = {}
        self.momentum_qmap = torch.tensor(
            [
                -0.8875,
                -0.6625,
                -0.4375,
                -0.2125,
                -0.0775,
                -0.0325,
                -0.0055,
                0.0000,
                0.0055,
                0.0325,
                0.0775,
                0.2125,
                0.4375,
                0.6625,
                0.8875,
                1.0000,
            ]
        )

        self.variance_qmap = torch.tensor(
            [
                0.0625,
                0.1250,
                0.1875,
                0.2500,
                0.3125,
                0.3750,
                0.4375,
                0.5000,
                0.5625,
                0.6250,
                0.6875,
                0.7500,
                0.8125,
                0.8750,
                0.9375,
                1.0000,
            ]
        )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=fused,
        )
        super().__init__(params, defaults)

    def get_qmetadata_by_state_name(self, optimizer_state_name):
        subconfig = self.get_subqconfig(optimizer_state_name)
        md = dict(
            b=subconfig.bits,
            scale_type=subconfig.scale_type,
            quant_type=subconfig.quant_type,
            round_type=subconfig.round_type,
            gp_sz=subconfig.group_size,
            signed=subconfig.signed,
        )
        return md

    def init_qstate(self, p, state_name):
        state = self.state[p]
        # print(f"{state=}")
        field = f"{state_name}_qstate"
        state[field] = {
            "enable": True,
            "overhead": dict(),
            "qmap": None,
        }
        subconfig = self.get_subqconfig(state_name)
        state[field]["enable"] = _get_qenable_fn(p, subconfig.threshold)

        md = self.get_qmetadata_by_state_name(state_name)
        # print(f"{md=}")
        qmap_key = (md["quant_type"], md["b"], md["signed"])
        # print(f"{qmap_key=}")

        if qmap_key not in self.qmaps:
            self.qmaps[qmap_key] = create_qmap(*qmap_key)
            print(f"created qmap = {self.qmaps[qmap_key]=}")
        # self.qmaps[qmap_key] = self.qmaps[qmap_key].to(p.device)
        state[field]["qmap"] = self.qmaps[qmap_key]
        # print(f"completing state for {state_name=}, with {state=}")

    def create_qmap(quant_type, bit, signed):
        """create qmap for quantization"""
        if quant_type == "nonlinear":
            return create_dynamic_map(signed, bit - 1, bit if signed else bit - 1)
        elif quant_type == "power-1":
            return create_pow_map(bit, signed, 1)

        else:
            raise ValueError(f"Not support {quant_type} quant type.")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        # have to store step as tensor, not scalar
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(
            state_values[0]["step"]
        )

        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def get_subqconfig(self, optimizer_state_name):
        if optimizer_state_name == "momentum":
            return self.config_momentum
        elif optimizer_state_name == "variance":
            return self.config_variance
        else:
            raise ValueError(f" invalid state name {optimizer_state_name=}")

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avgs_sqs,
        state_steps,
        exp_avgs_qmap,
        exp_avgs_sqs_qmap,
    ):
        # print(f"{exp_avgs_qmap=}")
        # print(f"{exp_avgs_sqs_qmap=}")
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW_FourBit does not support sparse gradients")
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

            # exp_avgs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_qstate"]["enable"])
            # exp_avg_sqs_q_enabled.append(self.override_q_enable[id(p)] if id(p) in self.override_q_enable else state["exp_avg_sq_qstate"]["enable"])
            # exp_avgs_q_overhead.append(state["exp_avg_qstate"]["overhead"])
            # exp_avg_sqs_q_overhead.append(state["exp_avg_sq_qstate"]["overhead"])
            # exp_avgs_qmap.append(state["exp_avg_qstate"]["qmap"])
            # exp_avg_sqs_qmap.append(state["exp_avg_sq_qstate"]["qmap"])

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
            exp_avgs_qmap = []
            exp_avg_sqs_qmap = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                exp_avgs_qmap,
                exp_avg_sqs_qmap,
            )

            kwargs = dict(
                params_with_grad=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                state_steps=state_steps,
                exp_avgs_qmap=exp_avgs_qmap,
                exp_avg_sqs_qmap=exp_avg_sqs_qmap,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

            _single_tensor_step(**kwargs)


def _single_tensor_step(
    params_with_grad: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    exp_avgs_qmap: List,
    exp_avg_sqs_qmap: List,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    # print(f"273: In Step function")
    # print(f" len params with grad {len(params_with_grad)}, len grads {len(grads)}, len exp_avgs {len(exp_avgs)}, len exp_avg_sqs    {len(exp_avg_sqs)}, len state_steps {len(state_steps)}")
    for i, param in enumerate(params_with_grad):
        # print(f"step loop {i}, {param[0]=}")
        # print(f"++++++++++++++++++++++++")
        grad = grads[i]
        q_exp_avg = exp_avgs[i]
        q_exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t += 1

        # decoupled weight decay
        param.mul_(1 - lr * weight_decay)

        # dequant
        q_enabled = True  # todo
        sq_enabled = True

        # print(f"288: {step_t=}, and {q_exp_avg=}")
        if q_exp_avg.numel() < 2:
            q_exp_avg.data = exp_avg = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        else:
            exp_avg = avgs_dequant(q_exp_avg, shape=param.shape)

        if q_exp_avg_sq.numel() < 2:
            q_exp_avg_sq.data = exp_avg_sq = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        else:
            exp_avg_sq = sqs_dequant(
                q_exp_avg_sq,
                shape=param.shape,
            )

        # update avgs

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step = step_t.item()
        bias_corr1 = 1 - beta1**step
        bias_corr2 = 1 - beta2**step
        step_size = lr / bias_corr1
        # if isinstance(bias_corr2, torch.Tensor):
        #    bias_corr2_sqrt = bias_corr2.sqrt()
        # else:
        bias_corr2_sqrt = math.sqrt(bias_corr2)

        denom = (exp_avg_sq.sqrt() / bias_corr2_sqrt).add_(eps)
        # print(f"321: {denom=}")
        # weight update
        # print(f"323: {param=}")
        param.addcdiv_(exp_avg, denom, value=-step_size)
        # print(f"325: after add cdiv {param=}")

        # quantize
        qx, gen = avgs_quant(exp_avg, shape=param.shape)
        # todo - err re: not tensor but should be tensor list
        # q_exp_avg.data = qx

        qx, gen = sqs_quant(exp_avg_sq, shape=param.shape)


def avgs_dequant(qx, shape):
    """dequantize the exp_avg"""
    x = qx.detach()
    b, signed = 4, True
    if isinstance(kwargs["qmap"], torch.Tensor):
        qmap = kwargs["qmap"]
    else:
        qmap = kwargs["qmap"][(b, signed)][quant_type]
    # x = nonlinear_dequant(x, qmap, b, shape=kwargs['scaled_shape'], )
    num_groups = (shape.numel() + 2047) // 2048
    grouped_x = ext_quantization.unpack_nonlinear(qx, qmap, b, num_groups, 2048)
    x = recon_grouped_tensor(grouped_x, shape)

    x = x.mul(max1)
    shape = kwargs["shape"]

    # reconstruct grouped tensor
    numel = shape.numel()
    recon_flatten = grouped_tensor.flatten()[:numel]
    recon = recon_flatten.view(shape)
    if x.stride() != stride:
        recon_x = torch.empty_strided(
            x.shape, stride, dtype=dtype, layout=torch.strided, device=x.device
        )
        recon_x.copy_(x)
        del x
        return recon_x
    else:
        x = x.to(dtype=dtype)
        return x


def get_sqs_tensor_statistics(
    x,
):
    qx = x.abs()
    max_dims = []
    for i in range(x.dim()):
        new_max = max_reduce_except_dim(qx, i)
        max_dims.append(new_max)
    return max_dims


def compute_sqs_tensor_scale(max_dims):
    rank = len(max_dims)
    scale_tensor = max_dims[0].clone()
    for i in range(1, rank):
        # broadcasting
        scale_tensor = torch.min(scale_tensor, max_dims[i])
    return scale_tensor


'''def sqs_quant(x, shape):
    """quantize the exp_avg_sq"""
    group_size = 128

    qx = x.detach()

    meta = {}
    meta["dtype"] = x.dtype
    meta["stride"] = x.stride()

    # quant scaling for sqs
    max_dims = get_sqs_statistics(qx.abs())
    st = compute_sqs_scale_tensor(max_dims)
    meta["max_dims"] = max_dims
    qx = qx.div(st)

    qmap_variance = torch.tensor(
            [
                0.0625,
                0.1250,
                0.1875,
                0.2500,
                0.3125,
                0.3750,
                0.4375,
                0.5000,
                0.5625,
                0.6250,
                0.6875,
                0.7500,
                0.8125,
                0.8750,
                0.9375,
                1.0000,
            ]
        )

    qx = sqs_quant_kernel(qx,qmap_variance) #  qmap, b, round_type="real-nearest")

    return qx, generated_metadata
'''


def sqs_quant(x, shape):
    """quantize the exp_avg_sq with rank1"""
    # grouped_qx = group_tensor(qx, 2048)
    # todo next
    qx = x.detach()  # keep the reference of original tensor

    # save kwargs
    generated_metadata = {}
    generated_metadata["dtype"] = x.dtype
    generated_metadata["stride"] = x.stride()

    generated_metadata["dim"] = qx.dim()
    if qx.dim() == 1:  # group
        group_size = 128
        qx = group_tensor(qx, group_size)
        max1 = max_reduce_except_dim(qx.abs(), 0)
        qx = qx.div(max1)
        print(f"469: {qx=}")
        generated_metadata["max1"] = max1
    else:

        max_dims = get_sqs_tensor_statistics(qx.abs())
        print(f"{max_dims=}")
        st = compute_sqs_tensor_scale(max_dims)
        print(f"{st=}")
        generated_metadata["max_dims"] = max_dims
        generated_metadata["max1"] = None
        qx = qx.div(st)
        print(f"496: {qx=}")
    # generated_metadata.update(md)

    # qx = nonlinear_quant(qx, qmap, b, round_type=kwargs['round_type'])
    qmap_variance = torch.tensor(
        [
            0.0625,
            0.1250,
            0.1875,
            0.2500,
            0.3125,
            0.3750,
            0.4375,
            0.5000,
            0.5625,
            0.6250,
            0.6875,
            0.7500,
            0.8125,
            0.8750,
            0.9375,
            1.0000,
        ]
    )

    variance_midpoint_lut = torch.tensor(
        [
            0.09375,
            0.15625,
            0.21875,
            0.28125,
            0.34375,
            0.40625,
            0.46875,
            0.53125,
            0.59375,
            0.65625,
            0.71875,
            0.78125,
            0.84375,
            0.90625,
            0.96875,
        ]
    )

    print(f"*before* quant sqs {qx.shape=}, {qx=}")
    qx = kernel_quant_nonlinear(qx, qmap_variance, variance_midpoint_lut, debug=True)
    print(f"*after* quant sqs {qx.shape=}, {qx=}")
    return qx, generated_metadata


def nonlinear_quant(x, qmap, b, round_type="real-nearest"):
    """quantize the exp_avg_sq"""

    def real_nonlinear_quant(qx, qmap, b, stochastic):
        # kernel
        grouped_qx = group_tensor(qx, 2048)
        return ext_quantization.pack_nonlinear(grouped_qx, qmap, b, stochastic)

    idx = real_nonlinear_quant(qx, qmap, b, False)
    return idx


def nonlinear_de_quant(qx, qmap, b, shape, round_type="real-nearest"):
    num_groups = (shape.numel() + 2047) // 2048
    grouped_x = ext_quantization.unpack_nonlinear(qx, qmap, b, num_groups, 2048)
    x = recon_grouped_tensor(grouped_x, shape)

    return x


def avgs_quant(x, shape):
    """quantize the exp_avg"""
    group_size = 128

    qmap = torch.tensor(
        [
            -0.8875,
            -0.6625,
            -0.4375,
            -0.2125,
            -0.0775,
            -0.0325,
            -0.0055,
            0.0000,
            0.0055,
            0.0325,
            0.0775,
            0.2125,
            0.4375,
            0.6625,
            0.8875,
            1.0000,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    midpoint_lut = torch.tensor(
        [
            -0.775,
            -0.55,
            -0.325,
            -0.145,
            -0.055,
            -0.019,
            -0.00275,
            0.00275,
            0.019,
            0.055,
            0.145,
            0.325,
            0.55,
            0.775,
            0.94375,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    qx = x.detach()
    print(f"445: {qx=}")
    print(f"{qx.shape=}")

    meta = {}
    meta["dtype"] = x.dtype
    meta["stride"] = x.stride()

    # quant scaling for exp_avgs
    qx = group_tensor(x, group_size)
    print(f"453: qx after group {qx=}")
    print(f"{qx.shape=}\n+++++++++++++++++")

    max_per_row = max_reduce_except_dim(qx.abs(), 0)
    print(f"458: {max_per_row=}")
    qx = qx.div(max_per_row)
    print(f"460: {qx=}")
    scaled_shape = qx.shape
    print(f"462: {scaled_shape=}")

    # metadata = max_per_row, scaled_shape
    # quantize
    grouped_qx = group_tensor(qx, 2048)
    # qx = cuda_kernel_pack_nonlinear(grouped_qx)
    qx = kernel_quant_nonlinear(grouped_qx, qmap, midpoint_lut)
    # let's do this in place for now
    print(f"461: {grouped_qx=}")
    print(f"{grouped_qx.shape=}")

    return qx, meta


def kernel_quant_nonlinear(
    x,
    qmap,
    midpoint_lut,
    debug=False,
):
    """quantize the exp_avg"""

    if debug:
        print(f"670: quant func received {x.shape=}, \n {x=}")
    bits = 4
    # kernel
    num_groups = x.data.size(0)
    group_size = x.data.size(1)
    if debug:
        print(f"665: {num_groups=}, {group_size=}")

    # // Compute total bits
    work_per_int = 8 / bits
    workint_per_thread = 4
    work_per_thread = work_per_int * workint_per_thread
    assert 8 % bits == 0
    #assert group_size % work_per_thread == 0

    total_bits = bits * (num_groups * group_size)
    if debug:
        print(f"675: {total_bits=}")
    packed_size = int((total_bits + 8) / 8)
    if debug:
        print(f"677: {packed_size=}")
    packed = torch.zeros((packed_size,), dtype=torch.uint8, device=x.device)
    # Tensor packed = torch::empty({(total_bits + 8) / 8,}, options);
    # print(f"493: {packed.shape=}")

    """// Pack float16/32 data into int8 bit stream, for bits < 8 and 8 % bit == 0
template<typename scalar_t, bool STOCHASTIC>
__global__ void pack_nonlinear_4bit_kernel(int32_t bits,
                                          const scalar_t* __restrict__ data,
                                          const float* __restrict__ qmap,
                                          int8_t* __restrict__ packed,
                                          std::pair<uint64_t, uint64_t> seeds) {
  const int group_id = blockIdx.x;
  const int id_in_group = threadIdx.x;
  const int64_t global_id = group_id * blockDim.x + id_in_group;
  const int work_per_int = 8 / bits;
  const int workint_per_thread = 4;
  const int work_per_thread = work_per_int << 2;
  const int8_t mask = (1 << bits) - 1;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, global_id, seeds.second, &state);

  for (int i = 0; i < workint_per_thread; i++) {
    uint8_t local_packed = 0;
    int64_t packed_id = global_id * workint_per_thread + i;
    for (int j = 0; j < work_per_int; j++) {
      const int64_t data_id = global_id * work_per_thread + i * work_per_int + j;
      const float noise = curand_uniform(&state);
      const float x = data[data_id];
      const uint8_t qx = (uint8_t)quantize_bsearch<STOCHASTIC>(qmap, bits, x, noise);
      local_packed |= ((qx & mask) << (j * bits));
    }

    packed[packed_id] = local_packed;
  }
}
    """

    for index in range(len(x[0])):
        if debug:
            print(f"{index=}, {x[0][index]=}")
        val = x[0][index].item()
        if debug:
            print(f"{val=}")
        if val == 0:
            break
        qitem = bsearch(val, qmap, midpoint_lut)
        if debug:
            print(f"729: {qitem=}, {val=}")
        packed[index] = qitem
    if debug:
        print(f"\n")
        print(f"{x[0][0:10]=}\n{packed[0:10]=}\n ")
        print(f"========================")

    # todo mask = (1 << bits) - 1
    # print(f"552: {mask=}")
    return packed


def bsearch(x, qmap, midpoint_lut):
    """ """
    low = 0
    high = 16
    print(f"546: {qmap[0]=}, {qmap[15]=}, {x=}")
    if x <= qmap[0]:
        return low
    if x >= qmap[15]:
        return 15

    while low < high:
        mid = (low + high) // 2
        if x <= qmap[mid]:
            high = mid
        else:
            low = mid + 1

    rank = 0
    # mid_val = (qmap[low - 1] + qmap[low]) * 0.5
    mid_val = midpoint_lut[low - 1]
    # assert torch.allclose(mid_val, lut_val)# , f"{mid_val=}, {lut_val=}"

    if mid_val < x:
        rank = low
    else:
        rank = low - 1
    print(f"767: {x=}, {low=}, {mid_val=}, {qmap[rank]=},")
    return rank
    """
    int lo = 0;
    int hi = 1 << bits;

    if (x <= qmap[lo])
      return lo;
    if (qmap[hi - 1] <= x)
      return (hi - 1);

    while (lo < hi){
      int mi = (lo + hi) >> 1;
      if (qmap[mi] <= x) lo = mi + 1;
      else hi = mi;
    }
    // return lo - 1;

    int rank = 0;
    if (STOCHASTIC) {
      float proba = (x - qmap[lo - 1]) / (qmap[lo] - qmap[lo - 1]);
      int flag = __float2int_rn(proba + noise - 0.5f);
      rank = (flag) ? lo : lo - 1;
    } else {
      float mid_val = (qmap[lo - 1] + qmap[lo]) * 0.5f;
      rank = (mid_val < x) ? lo : lo - 1;
    }
    return rank;
}
    """


def group_tensor(x: Tensor, group_size: int):
    """break the tensor into rows of 'group size', padding if needed with zeros"""
    x_flat = x.flatten()
    num_flat = x_flat.shape[0]

    # reshape
    if num_flat % group_size != 0:
        # pad
        new_num_flat = (num_flat // group_size + 1) * group_size
        delta = new_num_flat - num_flat

        x_flat = torch.cat(
            [x_flat, torch.zeros([delta], dtype=x.dtype, device=x.device)], dim=0
        )
    x_groups = x_flat.view(-1, group_size)
    return x_groups


def max_reduce_except_dim(input, dim):
    """compute max along all dims except provided dim"""
    rank = input.dim()
    result = input
    if rank:
        assert dim < rank, f"reducing tensor with {rank} dimensions failed"
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result
