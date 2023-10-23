import torch
import main
import triton
import triton.language as tl
import torchvision.transforms.functional as ft
import torch.utils.benchmark as benchmarkT


@triton.jit
def adjust_brightness_kernel(
        x_ptr,  # *Pointer* to first input vector.
        factor,  # Factor to adjust brightness by
        output_ptr,  # *Pointer* to output vector.
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = x * factor
    output = min(result, 1)
    tl.store(output_ptr + offsets, output, mask=mask)


def adjust_brightness_triton(x: torch.Tensor, factor: float, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = 1024
    if inplace:
        assert x.is_cuda
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        adjust_brightness_kernel[grid](x, factor, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
        return x

    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adjust_brightness_kernel[grid](x, factor, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def invert_kernel(
        x_ptr,  # *Pointer* to first input vector.
        output_ptr,  # *Pointer* to output vector.
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 1.0 - x
    tl.store(output_ptr + offsets, output, mask=mask)


def invert_triton(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = 1024
    if inplace:
        assert x.is_cuda
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        invert_kernel[grid](x, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
        return x

    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    invert_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def posterize_kernel(
        x_ptr,
        bits: int,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    bit_mask = -int(2 ** (8 - bits))
    x = x & bit_mask
    tl.store(y_ptr + offsets, x, mask=mask)


def posterize_triton(x: torch.Tensor, bits: int, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = 1024
    if inplace:
        assert x.is_cuda
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        posterize_kernel[grid](x, bits, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
        return x
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    posterize_kernel[grid](x, bits, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def rgb_to_grayscale_kernel(
        r_ptr,
        g_ptr,
        b_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    r = tl.load(r_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output = 0.2989 * r + 0.587 * g + 0.114 * b
    tl.store(y_ptr + offsets, output, mask=mask)


def rgb_to_grayscale_triton(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    output = torch.empty(1, x.size(dim=1), x.size(dim=2), device='cuda', dtype=torch.float32)
    assert x.is_cuda and output.is_cuda
    n_elements = int(output.numel())
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    r_ptr, g_ptr, b_ptr = x.unbind(dim=-3)
    rgb_to_grayscale_kernel[grid](r_ptr, g_ptr, b_ptr, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def autocontrast_kernel(
        x_ptr,
        y_ptr,
        n_elements,
        min,
        max,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    channel = tl.load(x_ptr + offsets, mask=mask)
    channel_scale = 1 / (max - min)
    output = (channel - min) * channel_scale
    tl.store(y_ptr + offsets, output, mask=mask)


def autocontrast_triton(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # BLOCK_SIZE = 1024
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = int(output.numel() / 3)  # Number of elements per channel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    min = x.amin(dim=(-2, -1), keepdim=True)
    max = x.amax(dim=(-2, -1), keepdim=True)
    # min = torch.tensor([[0],[0],[0]])
    # max = torch.tensor([[1], [1], [1]])
    autocontrast_kernel[grid](x[0], output[0], n_elements, min[0].item(), max[0].item(), num_warps=16,
                              BLOCK_SIZE=BLOCK_SIZE)
    autocontrast_kernel[grid](x[1], output[1], n_elements, min[1].item(), max[1].item(), num_warps=16,
                              BLOCK_SIZE=BLOCK_SIZE)
    autocontrast_kernel[grid](x[2], output[2], n_elements, min[2].item(), max[2].item(), num_warps=16,
                              BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def solarize_kernel(
        x_ptr,  # *Pointer* to first input vector.
        output_ptr,  # *Pointer* to output vector.
        threshold: float,
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x < threshold, x, 1.0 - x)
    tl.store(output_ptr + offsets, output, mask=mask)


def solarize_triton(x: torch.Tensor, threshold: float, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = 1024

    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    solarize_kernel[grid](x, output, threshold, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def adjust_gamma_kernel(
        x_ptr,  # *Pointer* to first input vector.
        output_ptr,  # *Pointer* to output vector.
        gamma: float,
        gain: float,
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
        # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = gain * tl.exp(tl.log(x) * gamma)

    tl.store(output_ptr + offsets, output, mask=mask)


def adjust_gamma_triton(x: torch.Tensor, gamma: float, gain: float = 1, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    # BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = 1024
    if inplace:
        assert x.is_cuda
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        posterize_kernel[grid](x, bits, x, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
        return x
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adjust_gamma_kernel[grid](x, output, gamma, gain, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def adjust_saturation_kernel(
        x_ptr,
        r_ptr,
        g_ptr,
        b_ptr,
        saturation,
        a,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # r = tl.load(r_ptr + offsets, mask=mask)
    # g = tl.load(g_ptr + offsets, mask=mask)
    # b = tl.load(b_ptr + offsets, mask=mask)
    a = tl.load(a + offsets, mask=mask)
    output = min((saturation * x) + ((1 - saturation) * a), 1)
    tl.store(y_ptr + offsets, output, mask=mask)


def adjust_saturation_triton(x: torch.Tensor, saturation: float, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    BLOCK_SIZE = 512
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = int(output.numel())
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    r_ptr, g_ptr, b_ptr = x.unbind(dim=-3)
    a = rgb_to_grayscale_triton(x).flatten().squeeze().repeat(3)
    adjust_saturation_kernel[grid](x, r_ptr, g_ptr, b_ptr, saturation, a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


@triton.jit
def adjust_contrast_kernel(
        r_ptr,
        g_ptr,
        b_ptr,
        contrast,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    r = tl.load(r_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    mean = (0.2989 * tl.sum(r, axis=0) + 0.587 * tl.sum(g, axis=0) + 0.114 * tl.sum(b, axis=0)) / 4
    output = min(contrast * r + (1 - contrast) * mean, 1)
    tl.store(output_ptr + offsets, output, mask=mask)

    offsets = BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output = min(contrast * g + (1 - contrast) * mean, 1)
    tl.store(output_ptr + offsets, output, mask=mask)

    offsets = 2*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output = min(contrast * b + (1 - contrast) * mean, 1)
    tl.store(output_ptr + offsets, output, mask=mask)

def adjust_contrast_triton(x: torch.Tensor, contrast: float, inplace: bool = False) -> torch.Tensor:
    n_channels, n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_rows * n_cols)
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = int(output.numel())
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    r_ptr, g_ptr, b_ptr = x.unbind(dim=-3)
    adjust_contrast_kernel[(1, )](r_ptr, g_ptr, b_ptr, contrast, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def pytorch1(in_tensor):
    return ft.adjust_contrast(in_tensor, 3.1)


def triton1(in_tensor):
    return adjust_contrast_triton(in_tensor, 3.1)


y = torch.rand(3, 2, 2, device='cuda')
# print(y)
output_torch = pytorch1(y)
output_triton = triton1(y)
# print("torch:")
print(output_torch)
print(torch.mean(ft.rgb_to_grayscale(y)))
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            224, 384, 512
        ],  # Different possible values for `x_name`.
        x_log=False,  # x-axis is not logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='MB/s',  # Label name for the y-axis.
        plot_name='autocontrast-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    tensor = torch.rand(3, size, size, device='cuda', dtype=torch.float32)
    # tensor_cpu = torch.rand(3, size, size, dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: pytorch1(tensor))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton1(tensor))
    # if provider == 'halide':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: halide(tensor_cpu))
    gbps = lambda ms: 12 * size / ms * 1e-3
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# benchmark.run(print_data=True, save_path="/mnt/Client/StrongUniversity/USYD-04/uni_lmur/")
# benchmark.run(print_data=True, save_path="/mnt/Client/StrongUniversity/USYD-04/uni_lmur/")
# benchmark.run(print_data=True, save_path="/mnt/Client/StrongUniversity/USYD-04/uni_lmur/")


results = []
# 224, 384, 512, 1024, 2048
sizes = [
    224, 384, 512
]
for size in sizes:
    tensor = torch.rand(3, size, size, device='cuda', dtype=torch.float32)
    label = "adjust_brightness"
    sub_label = "{}, {}".format(size, size)
    results.append(benchmarkT.Timer(
        stmt='triton1(x)',
        setup='from __main__ import triton1',
        globals={'x': tensor},
        label=label,
        sub_label=sub_label,
        description='Triton',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmarkT.Timer(
        stmt='pytorch1(x)',
        setup='from __main__ import pytorch1',
        globals={'x': tensor},
        label=label,
        sub_label=sub_label,
        description='PyTorch',
    ).blocked_autorange(min_run_time=1))

compare = benchmarkT.Compare(results)
compare.print()

# TODO: Row major ordering in memory - read by rows using block size?
