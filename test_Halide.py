import pytest
import torch
import main
import torchvision.io
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Feel free to change the images path if required
img1 = str(Path(__file__).parent / "." / "Images" / "dog1.jpg")
img2 = str(Path(__file__).parent / "." / "Images" / "dog2.jpg")

img_1 = Image.open(img1)
img_2 = Image.open(img2)
convert = transforms.ToTensor()
img1tensor = convert(img_1)
img2tensor = convert(img_2)

test_tensors = [img1tensor, img2tensor, torch.ones(3, 600, 800), torch.zeros(3, 600, 800), torch.rand(3, 600, 800)]

# test_tensors = []

tensor_sizes = [224, 384, 512, 1024, 2048]  # [224, 384, 512, 1024, 2048]

for size in tensor_sizes:
    test_tensors.append(torch.ones(3, size, size))
    test_tensors.append(torch.zeros(3, size, size))
    test_tensors.append(torch.rand(3, size, size))
    test_tensors.append(torch.rand(3, size, size))

posterize_test = []
for size in tensor_sizes:
    posterize_test.append(torch.zeros(3, size, size, dtype=torch.uint8))
    posterize_test.append(torch.ones(3, size, size, dtype=torch.uint8))
    posterize_test.append(torch.randint(0, 256, (3, size, size), dtype=torch.uint8))
    posterize_test.append(torch.randint(0, 256, (3, size, size), dtype=torch.uint8))


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("precision", [1.0, 1.2, 3.1415926])
def test_halide_brightness(tensor, precision):
    halide_output = torch.empty_like(tensor)
    main.adjust_brightness(tensor.clone(), precision, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_brightness(tensor.clone(), precision)
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
def test_invert(tensor):
    halide_output = torch.empty_like(tensor)
    main.invert(tensor.clone(), halide_output)
    torchvision_output = torchvision.transforms.functional.invert(tensor.clone())
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
def test_halide_grayscale(tensor):
    halide_output = torch.empty((tensor.size(dim=1), tensor.size(dim=2)), dtype=torch.float32)
    main.rgb_to_grayscale(tensor.clone(), halide_output)
    torchvision_output = torchvision.transforms.functional.rgb_to_grayscale(tensor)
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("threshold", [0.1, 0.2, 1])
def test_solarize(tensor, threshold):
    halide_output = torch.empty_like(tensor)
    main.solarize(tensor.clone(), threshold, halide_output)
    torchvision_output = torchvision.transforms.functional.solarize(tensor.clone(), threshold)
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("saturation", [0.1, 1, 2])
def test_adjust_saturation(tensor, saturation):
    halide_output = torch.empty_like(tensor)
    main.adjust_saturation(tensor.clone(), saturation, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_saturation(tensor.clone(), saturation)
    assert torch.allclose(torchvision_output, halide_output, atol=0.0000001)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("gamma", [0.1, 1, 10])
@pytest.mark.parametrize("gain", [0.3, 1])
def test_adjust_gamma(tensor, gamma, gain):
    halide_output = torch.empty_like(tensor)
    main.adjust_gamma(tensor.clone(), gamma, gain, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_gamma(tensor.clone(), gamma, gain)
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("contrast", [0.1, 1, 3])
def test_adjust_contrast(tensor, contrast):
    halide_output = torch.empty_like(tensor)
    main.adjust_contrast(tensor.clone(), contrast, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_contrast(tensor.clone(), contrast)
    assert torch.allclose(torchvision_output, halide_output, atol=0.001)


@pytest.mark.parametrize("tensor", posterize_test)
@pytest.mark.parametrize("bits", [1, 3, 5, 7])
def test_posterize(tensor, bits):
    halide_output = torch.empty_like(tensor)
    main.posterize(tensor.clone(), bits, halide_output)
    torchvision_output = torchvision.transforms.functional.posterize(tensor.clone(), bits)
    assert torch.allclose(torchvision_output, halide_output)


# @pytest.mark.parametrize("tensor", test_tensors)
# def test_autocontrast(tensor):
#     halide_output = torch.empty_like(tensor)
#     main.autocontrast(tensor.clone(), halide_output)
#     torchvision_output = torchvision.transforms.functional.autocontrast(tensor.clone())
#     assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("mean", [[0.1, 0.2, 0.3], [0.1, 0.5, 0.9], [0.1, 0.2, 0.5]])
@pytest.mark.parametrize("sd", [[0.3, 0.2, 0.1], [0.9, 0.5, 0.1], [0.5, 0.5, 0.5]])
def test_normalize(tensor, mean, sd):
    halide_output = torch.empty_like(tensor)
    main.normalize(tensor.clone(), mean, sd, halide_output)
    torchvision_output = torchvision.transforms.functional.normalize(tensor.clone(), mean, sd).clamp(0, 1)
    assert torch.allclose(torchvision_output, halide_output)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("hue", [-0.4, -0.1, 0.1, 0.4])
def test_adjust_hue(tensor, hue):
    halide_output = torch.empty_like(tensor)
    main.adjust_hue(tensor.clone(), hue, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_hue(tensor.clone(), hue)
    assert torch.allclose(torchvision_output, halide_output, atol=0.001)


@pytest.mark.parametrize("tensor", test_tensors)
@pytest.mark.parametrize("factor", [0.1, 0.4, 1.2, 3])
def test_adjust_sharpness(tensor, factor):
    halide_output = torch.empty_like(tensor)
    main.adjust_sharpness(tensor.clone(), factor, halide_output)
    torchvision_output = torchvision.transforms.functional.adjust_sharpness(tensor.clone(), factor)
    assert torch.allclose(torchvision_output, halide_output, atol=0.000001)

# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------------------#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("precision", [1.0, 1.2, 3.1415926])
# def test_halide_brightness(tensor, precision):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_brightness_auto(tensor.clone(), precision, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_brightness(tensor.clone(), precision)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# def test_invert(tensor):
#     halide_output = torch.empty_like(tensor)
#     main.invert_auto(tensor.clone(), halide_output)
#     torchvision_output = torchvision.transforms.functional.invert(tensor.clone())
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# def test_halide_grayscale(tensor):
#     halide_output = torch.empty((tensor.size(dim=1), tensor.size(dim=2)), dtype=torch.float32)
#     main.rgb_to_grayscale_auto(tensor.clone(), halide_output)
#     torchvision_output = torchvision.transforms.functional.rgb_to_grayscale(tensor)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("threshold", [0.1, 0.2, 1])
# def test_solarize(tensor, threshold):
#     halide_output = torch.empty_like(tensor)
#     main.solarize_auto(tensor.clone(), threshold, halide_output)
#     torchvision_output = torchvision.transforms.functional.solarize(tensor.clone(), threshold)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("saturation", [0.1, 1, 2])
# def test_adjust_saturation(tensor, saturation):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_saturation_auto(tensor.clone(), saturation, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_saturation(tensor.clone(), saturation)
#     assert torch.allclose(torchvision_output, halide_output, atol=0.0000001)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("gamma", [0.1, 1, 10])
# @pytest.mark.parametrize("gain", [0.3, 1])
# def test_adjust_gamma(tensor, gamma, gain):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_gamma_auto(tensor.clone(), gamma, gain, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_gamma(tensor.clone(), gamma, gain)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("contrast", [0.1, 1, 3])
# def test_adjust_contrast(tensor, contrast):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_contrast_auto(tensor.clone(), contrast, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_contrast(tensor.clone(), contrast)
#     assert torch.allclose(torchvision_output, halide_output, atol=0.001)
#
#
# @pytest.mark.parametrize("tensor", posterize_test)
# @pytest.mark.parametrize("bits", [1, 3, 5, 7])
# def test_posterize(tensor, bits):
#     halide_output = torch.empty_like(tensor)
#     main.posterize_auto(tensor.clone(), bits, halide_output)
#     torchvision_output = torchvision.transforms.functional.posterize(tensor.clone(), bits)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# def test_autocontrast(tensor):
#     halide_output = torch.empty_like(tensor)
#     main.autocontrast_auto(tensor.clone(), halide_output)
#     torchvision_output = torchvision.transforms.functional.autocontrast(tensor.clone())
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("mean", [[0.1, 0.2, 0.3], [0.1, 0.5, 0.9], [0.1, 0.2, 0.5]])
# @pytest.mark.parametrize("sd", [[0.3, 0.2, 0.1], [0.9, 0.5, 0.1], [0.5, 0.5, 0.5]])
# def test_normalize(tensor, mean, sd):
#     halide_output = torch.empty_like(tensor)
#     main.normalize_auto(tensor.clone(), mean, sd, halide_output)
#     torchvision_output = torchvision.transforms.functional.normalize(tensor.clone(), mean, sd).clamp(0, 1)
#     assert torch.allclose(torchvision_output, halide_output)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("hue", [-0.4, -0.1, 0.1, 0.4])
# def test_adjust_hue(tensor, hue):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_hue_auto(tensor.clone(), hue, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_hue(tensor.clone(), hue)
#     assert torch.allclose(torchvision_output, halide_output, atol=0.001)
#
#
# @pytest.mark.parametrize("tensor", test_tensors)
# @pytest.mark.parametrize("factor", [0.1, 0.4, 1.2, 3])
# def test_adjust_sharpness(tensor, factor):
#     halide_output = torch.empty_like(tensor)
#     main.adjust_sharpness_auto(tensor.clone(), factor, halide_output)
#     torchvision_output = torchvision.transforms.functional.adjust_sharpness(tensor.clone(), factor)
#     assert torch.allclose(torchvision_output, halide_output, atol=0.000001)
