# Meta copyright


import triton
import triton.language as tl

@triton.jit
def q_mapping_kernel(qmap, qmap_midpoints, x, output, block_size: tl.constexpr):
    pid = tl.program_id(0)
    # Guard to ensure we don't go out of bounds
    if pid >= block_size:
        return

    lo = 0
    hi = 16

    # Use Triton's memory load function
    qmap_lo = tl.load(qmap + lo)
    qmap_hi = tl.load(qmap + hi - 1)

    if x[pid] <= qmap_lo:
        output[pid] = lo
        return
    if qmap_hi <= x[pid]:
        output[pid] = 15  # 15 is the last index for 4 bit quantization
        return

    while lo < hi:
        mi = (lo + hi) >> 1
        qmap_mi = tl.load(qmap + mi)
        if qmap_mi <= x[pid]:
            lo = mi + 1
        else:
            hi = mi

    rank = 0
    mid_val = tl.load(qmap_midpoints[lo - 1])
    rank = tl.where(mid_val < x[pid], lo, lo - 1)
    output[pid] = rank
