# 4Bit_AdamW_Cuda
Triton does not support thread indexing and so had to move to Cuda for parallelized binary search support with quantization. 
<br>Will HIP'ify for AMD support.

This is a productionized implementation of the paper:
"Memory Efficient Optimizers with 4-bit States"
Bingrui Li, Jianfei Chen, Jun Zhu
https://arxiv.org/abs/2309.01507

