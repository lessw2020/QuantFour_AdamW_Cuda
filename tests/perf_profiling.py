
import torch
import torch.nn as nn
import torch.optim as torch_optim
import sys
sys.path.append("..")


from optimizer.fused_quantfour import AdamWFused_QuantFour


# set seed
torch.cuda.manual_seed(2020)

betas = (0.8, 0.88)
weight_decay = 0.03
lr = 0.005
eps = 1e-8

_size = 5

model = nn.Sequential(nn.Linear(_size, _size*2), nn.Linear(_size*2, _size))
model.cuda()


fourbit_adamw_opt = AdamWFused_QuantFour(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,)

# simple training loop
_total_iters = 10


for i in range(_total_iters):
    fourbit_adamw_opt.zero_grad(set_to_none=True)
    inp = torch.randn(_size, _size, device="cuda")
    model(inp).sum().backward()
    fourbit_adamw_opt.step()
