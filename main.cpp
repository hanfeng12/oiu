#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "HalideBuffer.h"
#include "halide_adjust_brightness.h"
#include "halide_adjust_brightness_auto.h"
#include "halide_rgb_to_grayscale.h"
#include "halide_rgb_to_grayscale_auto.h"
#include "halide_invert.h"
#include "halide_invert_auto.h"
#include "halide_solarize.h"
#include "halide_solarize_auto.h"
#include "halide_autocontrast.h"
#include "halide_autocontrast_auto.h"
#include "halide_adjust_gamma.h"
#include "halide_adjust_gamma_auto.h"
#include "halide_adjust_saturation.h"
#include "halide_adjust_saturation_auto.h"
#include "halide_adjust_contrast.h"
#include "halide_adjust_contrast_auto.h"
#include "halide_normalize.h"
#include "halide_normalize_auto.h"
#include "halide_posterize.h"
#include "halide_posterize_auto.h"
#include "halide_adjust_hue.h"
#include "halide_adjust_hue_auto.h"
#include "halide_adjust_sharpness.h"
#include "halide_adjust_sharpness_auto.h"
#include "halide_adjust_brightness_bw.h"
#include "halide_adjust_brightness_bw_auto.h"
//#include "halide_elastic_transform.h"
//#include "halide_elastic_transform_auto.h"
#include "halide_matmul.h"
#include "halide_matmul_auto.h"



std::vector<int> get_dims(const torch::Tensor &tensor) { // Helper for wrap()
    int ndims = tensor.ndimension();
    std::vector<int> dims(ndims, 0);
    // PyTorch dim order is reverse of Halide
    for (int dim = 0; dim < ndims; ++dim) {
        dims[dim] = tensor.size(ndims - 1 - dim);
    }
    return dims;
}

// Converts Buffer to a 3D matrix (C*H*W)
torch::Tensor ConvertBufferToTensor(const Halide::Runtime::Buffer<float> &buffer) {
    torch::Tensor tensor = torch::from_blob(buffer.data(), {buffer.channels(), buffer.height(), buffer.width()}, torch::TensorOptions().dtype(torch::kFloat32));
    return tensor;
}

Halide::Runtime::Buffer<float> ConvertTensorToBuffer(const torch::Tensor &tensor) { // Function to wrap a tensor in a Halide Buffer
    std::vector<int> dims = get_dims(tensor);
    float* pData = tensor.data_ptr<float>();
    return Halide::Runtime::Buffer<float>(pData, dims);
}

Halide::Runtime::Buffer<float> CloneDims(const Halide::Runtime::Buffer<float> &input) {
    return Halide::Runtime::Buffer<float>(input.width(), input.height(), input.channels());
}

torch::Tensor ConvertBufferToTensor_int(const Halide::Runtime::Buffer<uint8_t> &buffer) {
    // Create a PyTorch tensor using the from_blob method
    torch::Tensor tensor = torch::from_blob(buffer.data(), {buffer.channels(), buffer.height(), buffer.width()},
                                            torch::TensorOptions().dtype(torch::kUInt8));
    return tensor;
}

Halide::Runtime::Buffer<uint8_t> ConvertTensorToBuffer_int(torch::Tensor &tensor) { // Function to wrap an ATen tensor in a Halide Buffer
    std::vector<int> dims = get_dims(tensor);
    uint8_t* pData = tensor.data_ptr<uint8_t>();
    return Halide::Runtime::Buffer<uint8_t>(pData, dims);
}


void adjust_brightness(torch::Tensor &tensor, float factor, torch::Tensor &tensor_out) {
    if(factor < 0) {
        throw std::invalid_argument("Brightness factor must be >=0\n");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    
    if(input.channels() < 3) {
        halide_adjust_brightness_bw(input, factor, output);
    } else {
        halide_adjust_brightness(input, factor, output);
    }
}

void adjust_brightness_auto(torch::Tensor &tensor, float factor, torch::Tensor &tensor_out) {
    if(factor < 0) {
        throw std::invalid_argument("Brightness factor must be >=0\n");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

//    halide_shutdown_thread_pool();
    
    if(input.channels() < 3) {
        halide_adjust_brightness_bw_auto(input, factor, output);
    } else {
        halide_adjust_brightness_auto(input, factor, output);
    }
}

void grayscale(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    

    halide_rgb_to_grayscale(input, output);
}

void grayscale_auto(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_rgb_to_grayscale_auto(input, output);
}

void invert(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_invert(input, output);
}

void invert_auto(torch::Tensor &tensor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_invert_auto(input, output);
}

void invert_batch(std::vector<torch::Tensor> in, std::vector<torch::Tensor> out) {
    if(in.size() != out.size()) {
        throw std::invalid_argument("size of input and output lists are not equal\n");
    }
    for (int i = 0; i < in.size(); i++) {
        Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(in[i]);
        Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(out[i]);
        halide_invert(input, output);
    }
}


void solarize(torch::Tensor &tensor, float threshold, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_solarize(input, threshold, output);

}

void solarize_auto(torch::Tensor &tensor, float threshold, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_solarize_auto(input, threshold, output);

}

void autocontrast(torch::Tensor &tensor, torch::Tensor &tensor_out){
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_autocontrast(input, output);
}

void autocontrast_auto(torch::Tensor &tensor, torch::Tensor &tensor_out){
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_autocontrast_auto(input, output);
}

void adjust_gamma(torch::Tensor &tensor, float gamma, float gain, torch::Tensor &tensor_out) {
    if(gamma < 0) {
        throw std::invalid_argument("Gamma should be a non-negative real number");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_gamma(input, gamma, gain, output);
}

void adjust_gamma_auto(torch::Tensor &tensor, float gamma, float gain, torch::Tensor &tensor_out) {

    if(gamma < 0) {
        throw std::invalid_argument("Gamma should be a non-negative real number");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_gamma_auto(input, gamma, gain, output);
}

void adjust_saturation(torch::Tensor &tensor, float saturation, torch::Tensor &tensor_out) {
    if(saturation < 0) {
        throw std::invalid_argument("saturation factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_saturation(input, saturation, output);
}

void adjust_saturation_auto(torch::Tensor &tensor, float saturation, torch::Tensor &tensor_out) {
    if(saturation < 0) {
        throw std::invalid_argument("saturation factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_saturation_auto(input, saturation, output);
}

void adjust_contrast(torch::Tensor &tensor, float contrast, torch::Tensor &tensor_out) {
    if(contrast < 0) {
        throw std::invalid_argument("contrast factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    Halide::Runtime::Buffer<float> input2 = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> gray(input.width(), input.height());
    
    halide_rgb_to_grayscale(input2, gray);
    float total = 0.0;
    for(int j = 0; j < input.height(); j++) {
        for(int k = 0; k < input.width(); k++) {
            total += gray(k, j);
        }
    }
    
    halide_adjust_contrast(input, contrast, total/(input.width() * input.height()), output);
}

void adjust_contrast_auto(torch::Tensor &tensor, float contrast, torch::Tensor &tensor_out) {
    if(contrast < 0) {
        throw std::invalid_argument("contrast factor must be non-negative");
    }
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    Halide::Runtime::Buffer<float> input2 = ConvertTensorToBuffer(tensor);
    Halide::Runtime::Buffer<float> gray(input.width(), input.height());
    
    halide_rgb_to_grayscale_auto(input2, gray);
    float total = 0.0;
    for(int j = 0; j < input.height(); j++) {
        for(int k = 0; k < input.width(); k++) {
            total += gray(k, j);
        }
    }
    halide_adjust_contrast_auto(input, contrast, total/(input.width() * input.height()), output);
}

void normalize(torch::Tensor &tensor, std::vector<float> mean, std::vector<float> sd, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    // This is supposed to add support for 2 channel images, does not work since halide_normalize will always expect
    // 2 more args than are provided if channels != 3
    if(input.channels() == 3) {
        
        halide_normalize(input, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2], output);
    } else {
        //halide_normalize(input, mean[0], mean[1], sd[0], sd[1]);
    }
}

void normalize_auto(torch::Tensor &tensor, std::vector<float> mean, std::vector<float> sd, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    // This is supposed to add support for 2 channel images, does not work since halide_normalize will always expect
    // 2 more args than are provided if channels != 3
    if(input.channels() == 3) {
        
        halide_normalize_auto(input, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2], output);
    } else {
        //halide_normalize(input, mean[0], mean[1], sd[0], sd[1]);
    }
}

void posterize(torch::Tensor &tensor, int bits, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<uint8_t> input = ConvertTensorToBuffer_int(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<uint8_t> output = ConvertTensorToBuffer_int(tensor_out);
    if(input.channels() != 3) {
        throw std::invalid_argument("Posterize required a tensor with 3 dimensions");
    }
    
    halide_posterize(input, bits, output);
}

void posterize_auto(torch::Tensor &tensor, int bits, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<uint8_t> input = ConvertTensorToBuffer_int(tensor);  // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<uint8_t> output = ConvertTensorToBuffer_int(tensor_out);
    if(input.channels() != 3) {
        throw std::invalid_argument("Posterize required a tensor with 3 dimensions");
    }
    
    halide_posterize_auto(input, bits, output);
}

void adjust_hue(torch::Tensor &tensor, float hue_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_hue(input, hue_factor, output);
}

void adjust_hue_auto(torch::Tensor &tensor, float hue_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_hue_auto(input, hue_factor, output);
}

void adjust_sharpness(torch::Tensor &tensor, float sharpness_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_sharpness(input, sharpness_factor, output);
}

void adjust_sharpness_auto(torch::Tensor &tensor, float sharpness_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    
    halide_adjust_sharpness_auto(input, sharpness_factor, output);
}

//void elastic_transform(torch::Tensor &tensor, torch::Tensor &dis, torch::Tensor &tensor_out, float fill=0){
//    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
//    Halide::Runtime::Buffer<float> displacement = ConvertTensorToBuffer(dis);  // Wrap the tensor in a Halide buffer
//    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
//
//    if (displacement.width()!=2 || displacement.height()!=input.width() || displacement.channels()!=input.height() || displacement.dim(3).extent()!=1){
//        throw std::invalid_argument("The displacement expected shape is [1,H,W,2], the displacement passed has a invalid shape.");
//    }
//
//    halide_elastic_transform(input, displacement, fill, output);
//
//}
//
//void elastic_transform_auto(torch::Tensor &tensor, torch::Tensor &dis, torch::Tensor &tensor_out, float fill=0){
//    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor);  // Wrap the tensor in a Halide buffer
//    Halide::Runtime::Buffer<float> displacement = ConvertTensorToBuffer(dis);  // Wrap the tensor in a Halide buffer
//    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
//
//    if (displacement.width()!=2 || displacement.height()!=input.width() || displacement.channels()!=input.height() || displacement.dim(3).extent()!=1){
//        throw std::invalid_argument("The displacement expected shape is [1,H,W,2], the displacement passed has a invalid shape.");
//    }
//
//    halide_elastic_transform_auto(input, displacement, fill, output);
//
//}

void matmul(torch::Tensor &tensor1, torch::Tensor &tensor2, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input1 = ConvertTensorToBuffer(tensor1);
    Halide::Runtime::Buffer<float> input2 = ConvertTensorToBuffer(tensor2);
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    halide_set_num_threads(4);
    halide_matmul(input1, input2, output);
}

void matmul_auto(torch::Tensor &tensor1, torch::Tensor &tensor2, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input1 = ConvertTensorToBuffer(tensor1);
    Halide::Runtime::Buffer<float> input2 = ConvertTensorToBuffer(tensor2);
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);
    halide_set_num_threads(4);
    halide_matmul_auto(input1, input2, output);
}

void adjust_sharpness(torch::Tensor &tensor, float sharpness_factor, torch::Tensor &tensor_out) {
    Halide::Runtime::Buffer<float> input = ConvertTensorToBuffer(tensor); // Wrap the tensor in a Halide buffer
    Halide::Runtime::Buffer<float> output = ConvertTensorToBuffer(tensor_out);

    halide_adjust_sharpness(input, sharpness_factor, output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adjust_brightness", &adjust_brightness);
    m.def("rgb_to_grayscale", &grayscale);
    m.def("invert", &invert);
    m.def("invert_batch", &invert_batch);
    m.def("solarize", &solarize);
    m.def("autocontrast", &autocontrast);
    m.def("adjust_gamma", &adjust_gamma);
    m.def("adjust_saturation", &adjust_saturation);
    m.def("adjust_contrast", &adjust_contrast);
    m.def("normalize", &normalize);
    m.def("posterize", &posterize);
    m.def("adjust_hue", &adjust_hue);
    m.def("adjust_sharpness", &adjust_sharpness);
//    m.def("elastic_transform", &elastic_transform);
    m.def("matmul", &matmul);

    m.def("adjust_brightness_auto", &adjust_brightness_auto);
    m.def("rgb_to_grayscale_auto", &grayscale_auto);
    m.def("invert_auto", &invert_auto);
    m.def("solarize_auto", &solarize_auto);
    m.def("autocontrast_auto", &autocontrast_auto);
    m.def("adjust_gamma_auto", &adjust_gamma_auto);
    m.def("adjust_saturation_auto", &adjust_saturation_auto);
    m.def("adjust_contrast_auto", &adjust_contrast_auto);
    m.def("normalize_auto", &normalize_auto);
    m.def("posterize_auto", &posterize_auto);
    m.def("adjust_hue_auto", &adjust_hue_auto);
    m.def("adjust_sharpness_auto", &adjust_sharpness_auto);
//    m.def("elastic_transform_auto", &elastic_transform_auto);
    m.def("matmul_auto", &matmul_auto);
}
