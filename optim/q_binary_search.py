# Meta copyright


import triton
import triton.language as tl

@triton.jit
def q_mapping_kernel(qmap, qmap_midpoints, x, output, block_size: tl.constexpr):
    pid = tl.program_id(0)
    # Guard to ensure we don't go out of bounds
    if pid >= block_size:
        return

    x_offsets = tl.arange(0, block_size)
    x_vals = tl.load(x + x_offsets, mask = x_offsets < block_size)
    tl.device_print("x_vals ", x_vals)
    return
    """
    qmap_vals = tl.tensor(
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
            type=tl.float16,
        )

    lo = 0
    hi = 15

    #qmap_offsets = tl.arange(0, 16)
    #qmap_vals = tl.load(qmap + qmap_offsets) #, mask = )
    #qmap_midpoints = tl.load(qmap_midpoints + qmap_offsets-1)

    #tl.device_print("qmap ", qmap_vals)
    #tl.device_print("qmap_midpoints ", qmap_midpoints)
    # Use Triton's memory load function
    #qmap_lo = qmap_vals[lo]
    #qmap_hi = qmap_vals[hi]
    #tl.device_print("lo ", qmap_lo)
    #tl.device_print("high ", qmap_hi)
    return
    """
    """

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
    """
