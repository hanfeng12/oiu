import torch
from torchvision import transforms

import main
import torchvision.transforms.functional
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
import math
from PIL import Image

# Halide functions from PyBind11's python module
def halide_adjust_brightness(tensor, tensor_out):
    main.adjust_brightness(tensor, 1.4, tensor_out)


def halide_rgb_to_grayscale(tensor, tensor_out):
    main.rgb_to_grayscale(tensor, tensor_out)


def halide_invert(tensor, tensor_out):
    main.invert(tensor, tensor_out)



def halide_auto_adjust_brightness(tensor, tensor_out):
    main.adjust_brightness_auto(tensor, 1.4, tensor_out)


def halide_auto_rgb_to_grayscale(tensor, tensor_out):
    main.rgb_to_grayscale_auto(tensor, tensor_out)


def halide_auto_invert(tensor, tensor_out):
    main.invert_auto(tensor, tensor_out)


def halide_solarize(tensor, tensor_out):
    main.solarize(tensor, 0.5, tensor_out)


def halide_auto_solarize(tensor, tensor_out):
    main.solarize_auto(tensor, 0.5, tensor_out)


def halide_adjust_saturation(tensor, tensor_out):
    main.adjust_saturation(tensor, 0.5, tensor_out)


def halide_auto_adjust_saturation(tensor, tensor_out):
    main.adjust_saturation_auto(tensor, 0.5, tensor_out)


def halide_adjust_gamma(tensor, tensor_out):
    main.adjust_gamma(tensor, 0.5, 1, tensor_out)


def halide_auto_adjust_gamma(tensor, tensor_out):
    main.adjust_gamma_auto(tensor, 0.5, 1, tensor_out)


def halide_adjust_contrast(tensor, tensor_out):
    main.adjust_contrast(tensor, 0.5, tensor_out)


def halide_auto_adjust_contrast(tensor, tensor_out):
    main.adjust_contrast_auto(tensor, 0.5, tensor_out)


def halide_normalize(tensor, tensor_out):
    main.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], tensor_out)

def halide_auto_normalize(tensor, tensor_out):
    main.normalize_auto(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], tensor_out)


def halide_posterize(tensor, tensor_out):
    main.posterize(tensor, 3, tensor_out)


def halide_auto_posterize(tensor, tensor_out):
    main.posterize_auto(tensor, 3, tensor_out)


def halide_adjust_hue(tensor, tensor_out):
    main.adjust_hue(tensor, 0.1, tensor_out)

def halide_auto_adjust_hue(tensor, tensor_out):
    main.adjust_hue_auto(tensor, 0.1, tensor_out)


def halide_autocontrast(tensor, tensor_out):
    main.autocontrast(tensor, tensor_out)


def halide_auto_autocontrast(tensor, tensor_out):
    main.autocontrast_auto(tensor, tensor_out)


def halide_adjust_sharpness(tensor, tensor_out):
    main.adjust_sharpness(tensor, 1.4, tensor_out)


def halide_auto_adjust_sharpness(tensor, tensor_out):
    main.adjust_sharpness_auto(tensor, 1.4, tensor_out)

def halide_elastic_transform(tensor, tensor_out):
    channels, height, width = tensor.shape
    displacement = torch.ones(1,height,width,2)
    main.elastic_transform(tensor, displacement, tensor_out,0)

# PyTorch functions
def pytorch_adjust_brightness(tensor):
    return torchvision.transforms.functional.adjust_brightness(tensor, 1.4)


def pytorch_rgb_to_grayscale(tensor):
    return torchvision.transforms.functional.rgb_to_grayscale(tensor)


def pytorch_invert(tensor):
    return torchvision.transforms.functional.invert(tensor)


def pytorch_hflip(tensor):
    return torchvision.transforms.functional.hflip(tensor)


def pytorch_vflip(tensor):
    return torchvision.transforms.functional.vflip(tensor)


def pytorch_erase(tensor):
    return torchvision.transforms.functional.erase(
        tensor, 50, 50, 100, 100, 0, True)


def pytorch_solarize(tensor):
    return torchvision.transforms.functional.solarize(tensor, 0.5)


def pytorch_adjust_saturation(tensor):
    return torchvision.transforms.functional.adjust_saturation(tensor, 0.5)


def pytorch_adjust_gamma(tensor):
    return torchvision.transforms.functional.adjust_gamma(tensor, 0.5)


def pytorch_adjust_contrast(tensor):
    return torchvision.transforms.functional.adjust_contrast(tensor, 0.5)


def pytorch_crop(tensor):
    return torchvision.transforms.functional.crop(tensor, 50, 50, 150, 150)


def pytorch_normalize(tensor):
    return torchvision.transforms.functional.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def pytorch_posterize(tensor):
    return torchvision.transforms.functional.posterize(tensor, 3)

def pytorch_elastic_transform(tensor):
    channels, height,width = tensor.shape
    displacement = torch.ones(1,height,width,2)
    return torchvision.transforms.functional.elastic_transform(tensor, displacement, interpolation=InterpolationMode.NEAREST, fill=[0])

def pytorch_adjust_hue(tensor):
    return torchvision.transforms.functional.adjust_hue(tensor, 0.1)


def pytorch_autocontrast(tensor):
    return torchvision.transforms.functional.autocontrast(tensor)


def pytorch_adjust_sharpness(tensor):
    return torchvision.transforms.functional.adjust_sharpness(tensor, 1.4)


def halide_matmul(tensor, tensor2, tensor_out):
    main.matmul(tensor, tensor2, tensor_out)

def halide_auto_matmul(tensor, tensor2, tensor_out):
    main.matmul_auto(tensor, tensor2, tensor_out)


def pytorch_matmul(tensor, tensor2):
    return torch.matmul(tensor, tensor2)



# Benchmarks a certain amount of times with a single type of tensor and 1 thread
def benchmark_basic(repeats, h, w, function):
    x = torch.ones(3, h, w)
    y = torch.empty_like(x)
    t0 = benchmark.Timer(
        stmt='pytorch_' + function + '(x)',
        setup='from __main__ import pytorch_' + function,
        globals={'x': x},
        num_threads=1)

    t1 = benchmark.Timer(
        stmt='halide_auto_' + function + '(x, y)',
        setup='from __main__ import halide_auto_' + function,
        globals={'x': x, 'y': y},
        num_threads=1)
    # print(t0.timeit(repeats))
    print(t1.timeit(repeats))


# Benchmark and compare using an array of tensor dimensions
def benchmark_multiple(sizes, functions):
    for f in functions:
        results = []
        print(f)
        for dim in sizes:
            # label and sub_label are the rows
            # description is the column
            label = f
            x = torch.rand(dim, dim)
            x1 = torch.rand(dim, dim)
            if type(dim) is tuple:  # Rectangle tensor
                sub_label = f'[{dim[1]}, {dim[0]}]'
                x = torch.ones(3, dim[1], dim[0])
                if f == "posterize":
                    x = torch.randint(0, 256, (3, dim[1], dim[0]), dtype=torch.uint8)
            else:
                sub_label = f'[{dim}, {dim}]'
                x = torch.rand(dim, dim)
                x1 = torch.rand(dim, dim)
                if f == "posterize":
                    x = torch.randint(0, 256, (3, dim, dim), dtype=torch.uint8)
            for num_threads in [4]:
                # if f == "crop":
                func = "y = torch.empty_like(x)"
                if f == "rgb_to_grayscale":
                    func = "y = torch.empty(x.size(dim=1), x.size(dim=2))"
                results.append(benchmark.Timer(
                    stmt='halide_auto_' + f + '(x, x1, y)',
                    setup='from __main__ import halide_auto_' + f + "\n" + func,
                    globals={'x': x,
                             'x1':x1},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description='Auto',
                ).blocked_autorange(min_run_time=1))
                results.append(benchmark.Timer(
                    stmt='halide_' + f + '(x, x1, y)',
                    setup='from __main__ import halide_' + f + "\n" + func,
                    globals={'x': x,
                             'x1':x1},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description='Halide',
                ).blocked_autorange(min_run_time=1))
                # results.append(benchmark.Timer(
                #     stmt='halide_auto_' + f + "_adam" + '(x, y)',
                #     setup='from __main__ import halide_auto_' + f + "_adam\n" + func,
                #     globals={'x': x1},
                #     num_threads=num_threads,
                #     label=label,
                #     sub_label=sub_label,
                #     description='AdamAuto',
                # ).blocked_autorange(min_run_time=1))
                # results.append(benchmark.Timer(
                #     stmt='halide_' + f + "_adam" + '(x, y)',
                #     setup='from __main__ import halide_' + f + "_adam\n" + func,
                #     globals={'x': x1},
                #     num_threads=num_threads,
                #     label=label,
                #     sub_label=sub_label,
                #     description='Adam',
                # ).blocked_autorange(min_run_time=1))
                results.append(benchmark.Timer(
                    stmt='pytorch_' + f + '(x, x1)',
                    setup='from __main__ import pytorch_' + f,
                    globals={'x': x,
                             'x1':x1},
                    num_threads=num_threads,
                    label=label,
                    sub_label=sub_label,
                    description='PyTorch',
                ).blocked_autorange(min_run_time=1))

        compare = benchmark.Compare(results)
        compare.print()


def visualize(img):
    #  TODO: Include posterize
    functions = ["adjust_brightness", "invert", "solarize", "rgb_to_grayscale",
                 "adjust_saturation", "adjust_gamma", "adjust_contrast", "normalize", "adjust_hue",
                 "autocontrast", "adjust_sharpness"]
    cols = 3  # original, torchvision, halide
    rows = len(functions)  # The +1 is for the black and white example

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(
        8 * cols, 8 * rows))  # replace with bigger numbers to display with more resolution

    convert = torchvision.transforms.ToTensor()
    tensor_image = convert(img)

    for i in range(rows):
        tensor_torch = tensor_image.clone()
        tensor_halide = tensor_image.clone()
        tensor_halide_out = torch.empty_like(tensor_halide)

        pytorch_func = globals()["pytorch_" + functions[i]]
        halide_func = globals()["halide_auto_" + functions[i]]
        axes[i, 0].set_ylabel(functions[i])
        axes[i, 0].imshow(tensor_image.permute(1, 2, 0))
        axes[i, 0].set_title("original")
        print(functions[i])
        if functions[i] == "rgb_to_grayscale":
            axes[i, 1].imshow(pytorch_func(tensor_torch).permute(1, 2, 0), cmap="gray")
            axes[i, 1].set_title("torchvision")
            tensor_halide_out = torch.empty(tensor_halide.size(dim=1), tensor_halide.size(dim=2))
            halide_func(tensor_halide, tensor_halide_out)
            axes[i, 2].imshow(tensor_halide_out, cmap="gray")
            axes[i, 2].set_title("halide")
            continue
        elif functions[i] == "crop":
            axes[i, 1].imshow(pytorch_func(tensor_torch).permute(1, 2, 0))
            axes[i, 1].set_title("torchvision")
            tensor_halide_out = torch.empty(3, 150, 150)
            halide_func(tensor_halide, tensor_halide_out)
            axes[i, 2].imshow(tensor_halide_out.permute(1, 2, 0))
            axes[i, 2].set_title("halide")
            continue
        axes[i, 1].imshow(pytorch_func(tensor_torch).permute(1, 2, 0))
        axes[i, 1].set_title("torchvision")
        halide_func(tensor_halide, tensor_halide_out)
        axes[i, 2].imshow(tensor_halide_out.permute(1, 2, 0))
        axes[i, 2].set_title("halide")

    # print("bw adjust_brightness")
    # bnw_image = torchvision.transforms.functional.rgb_to_grayscale(tensor_image)
    # tensor_torch = bnw_image.clone()
    # tensor_halide = bnw_image.clone()
    # pytorch_func = globals()["pytorch_adjust_brightness"]
    # halide_func = globals()["halide_adjust_brightness"]
    # axes[len(functions), 0].set_ylabel("adjust_brightness")
    # axes[len(functions), 0].imshow(bnw_image.permute(1, 2, 0), cmap="gray")
    # axes[len(functions), 0].set_title("original")
    # axes[len(functions), 1].imshow(pytorch_func(tensor_torch).permute(1, 2, 0), cmap="gray")
    # axes[len(functions), 1].set_title("torchvision")
    # tensor_halide_out = torch.empty_like(tensor_halide)
    # halide_func(tensor_halide, tensor_halide_out)
    # axes[len(functions), 2].imshow(tensor_halide_out.permute(1, 2, 0), cmap="gray")
    # axes[len(functions), 2].set_title("halide")

    plt.tight_layout()
    plt.savefig("Images/comparison.png")


# benchmark_basic(10, 1000, 1000, "adjust_brightness")

img = Image.open("Images/dog1.jpg")
visualize(img)

_all = ["adjust_brightness", "invert", "rgb_to_grayscale", "solarize", "hflip", "vflip", "erase",
        "adjust_saturation", "adjust_gamma", "crop", "adjust_contrast", "normalize", "posterize", "elastic_transform"]

#benchmark_multiple([224, 384, 512, (800, 600), 1024, 2048], _all)
#benchmark_basic(300, height, width, "invert")

