#include "Halide.h"
#include <cmath>
#include <stdio.h>

using namespace Halide;

class AdjustBrightness : public Halide::Generator<AdjustBrightness> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> factor{"factor"};
    Output<Buffer<float, 3>> brighter{"adjust_brightness"};
    Var x, y, c;
    void generate() {
        brighter(x, y, c) = min(input(x, y, c) * factor, 1);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            factor.set_estimate(1.4f);
            brighter.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});

        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustBrightnessBW : public Halide::Generator<AdjustBrightnessBW> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> factor{"factor"};
    Output<Buffer<float, 3>> brighter{"adjust_brightness_bw"};
    Var x, y, c;
    void generate() {
        brighter(x, y, c) = clamp(input(x, y, c) * factor, 0, 1);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {1, 1}});
            factor.set_estimate(1.4f);
            brighter.set_estimates({{2048, 2048}, {2048, 2048}, {1, 1}});

        } else {
            ; // Manual scheduling
        }
    }
};


class Invert : public Halide::Generator<Invert> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<void, 3>> inverted{"invert"};

    Var x, y, c;

    void generate() {
        inverted(x, y, c) = 1.0f - input(x, y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            inverted.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
        } else {
            ; // Manual scheduling
        }
    }
};

class Grayscale : public Halide::Generator<Grayscale> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<void, 2>> grayscale{"grayscale"};

    Var x, y;

    void generate() {
        grayscale(x, y) = (0.2989f * input(x, y, 0)) + (0.587f * input(x, y, 1)) + (0.114f * input(x, y, 2));
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            grayscale.set_estimates({{2048, 2048}, {2048, 2048}});
        } else {
            ; // Manual scheduling
        }
    }
};

class Hflip : public Halide::Generator<Hflip> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<void, 3>> hflipped{"hflip"};

    Var x, y, c;

    void generate() {
        hflipped(x, y, c) = input(input.dim(0).max() - x, y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            hflipped.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
        } else {
            ; // Manual scheduling
        }
    }
};

class Vflip : public Halide::Generator<Vflip> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<void, 3>> vflipped{"vflip"};

    Var x, y, c;

    void generate() {
        vflipped(x, y, c) = input(x, input.dim(1).max() - y, c);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            vflipped.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
        } else {
            ; // Manual scheduling
        }
    }
};

class Erase : public Halide::Generator<Erase> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<int> i{"i"};
    Input<int> j{"j"};
    Input<int> h{"h"};
    Input<int> w{"w"};
    Input<float> v{"v"};
    Output<Buffer<float, 3>> erase{"erase"};
    Var x, y, c;
    Func replace;
    void generate() {
        replace(x, y, c) = v;
        erase(x, y, c) = BoundaryConditions::constant_exterior(replace, input(x, y, c), i, w, j, h)(x, y, c);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            erase.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            i.set_estimate(1.0);
            j.set_estimate(1.0);
            h.set_estimate(1.0);
            w.set_estimate(1.0);
            v.set_estimate(0.0f);
        } else {
            ; // Manual scheduling
        }
    }
};

class Solarize : public Halide::Generator<Solarize> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> threshold{"threshold"};
    Output<Buffer<float, 3>> solarize{"solarize"};
    Func invert;
    Var x, y, c;
    void generate() {
        invert(x, y, c) = 1 - input(x, y, c);
        solarize(x, y, c) = select(input(x, y, c) >= threshold, invert(x, y, c), input(x, y, c));
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            solarize.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            threshold.set_estimate(0.5f);
        } else {
            ; // Manual scheduling
        }
    }
};

class Crop : public Halide::Generator<Crop> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<int> i{"i"};
    Input<int> j{"j"};
    Input<int> h{"h"};
    Input<int> w{"w"};
    Output<Buffer<float, 3>> crop{"crop"};
    Var x, y, c;
    void generate() {
        crop(x, y, c) = input(x + i, y + j, c);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            crop.set_estimates({{50, 2048}, {50, 2048}, {3, 3}});
            i.set_estimate(0);
            j.set_estimate(0);
            h.set_estimate(100);
            w.set_estimate(100);
        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustGamma : public Halide::Generator<AdjustGamma> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> gamma{"gamma"};
    Input<float> gain{"gain"};
    Output<Buffer<float, 3>> adjust_gamma{"adjust_gamma"};
    Var x, y, c;
    void generate() {
        adjust_gamma(x, y, c) = clamp(gain * (Halide::fast_pow(input(x, y, c), gamma)), 0, 1);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            adjust_gamma.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            gamma.set_estimate(0.5f);
            gain.set_estimate(1.0f);
        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustSaturation : public Halide::Generator<AdjustSaturation> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> saturation{"saturation"};
    Output<Buffer<float, 3>> adjust_saturation{"adjust_saturation"};
    Var x, y, c;
    Func rgb_to_gray;
    void generate() {
        rgb_to_gray(x, y) = (0.2989f * input(x, y, 0)) + (0.587f * input(x, y, 1)) + (0.114f * input(x, y, 2));
        adjust_saturation(x, y, c) = clamp((saturation * input(x, y, c) + (1 - saturation) * rgb_to_gray(x, y)),
                                           0, 1);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            adjust_saturation.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            saturation.set_estimate(1.5f);
        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustContrast : public Halide::Generator<AdjustContrast> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> contrast{"contrast"};
    Input<float> mean{"mean"};
    Output<Buffer<float, 3>> adjust_contrast{"adjust_contrast"};
    Var x, y, c;
    void generate() {
        adjust_contrast(x, y, c) = clamp((contrast * input(x, y, c) + (1 - contrast) * mean), 0, 1);
    }

    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            adjust_contrast.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            contrast.set_estimate(1.0f);
            mean.set_estimate(1.0f);
        } else {
            ; // Manual scheduling
        }
    }
};

class Normalize : public Halide::Generator<Normalize> {
public:
    Input<Buffer<float, 3>> input{"input"};

    Input<float> r_mean{"r_mean"};
    Input<float> g_mean{"g_mean"};
    Input<float> b_mean{"b_mean"};
    Input<float> r_sd{"r_sd"};
    Input<float> g_sd{"g_sd"};
    Input<float> b_sd{"b_sd"};

    Output<Buffer<float, 3>> normalize{"normalize"};
    Var x, y, c;
    Expr value;
    Func r;
    Func g;
    Func b;
    void generate() {
        r(x, y, c) = (input(x, y, 0) - r_mean) / r_sd, 0, 1;
        g(x, y, c) = (input(x, y, 1) - g_mean) / g_sd, 0, 1;
        b(x, y, c) = (input(x, y, 2) - b_mean) / b_sd, 0, 1;
        value = select(c == 0, r(x, y, c),
                       c == 1, g(x, y, c),
                       b(x, y, c));
        normalize(x, y, c) = clamp(value, 0.0f, 1.0f);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            normalize.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            r_mean.set_estimate(0.7f);
            g_mean.set_estimate(0.7f);
            b_mean.set_estimate(0.7f);
            r_sd.set_estimate(0.6f);
            g_sd.set_estimate(0.6f);
            b_sd.set_estimate(0.6f);
        } else {
            ; // Manual scheduling
        }
    }
};

class Posterize : public Halide::Generator<Posterize> {
public:
    Input<Buffer<uint8_t, 3>> input{"input"};
    Input<uint8_t> bits{"bits"};
    Output<Buffer<uint8_t, 3>> posterize{"posterize"};
    Var x, y, c;
    Expr mask;
    void generate() {
        mask = -cast<uint8_t>(Halide::pow(2, 8 - bits));
        posterize(x, y, c) = input(x, y, c) & mask;
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            posterize.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            bits.set_estimate(4);
        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustHue : public Halide::Generator<AdjustHue> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> hue_factor{"hue_factor"};
    Output<Buffer<float, 3>> adjust_hue{"adjust_hue"};
    Var x, y, c;

    void generate() {
        Expr r = input(x, y, 0);
        Expr g = input(x, y, 1);
        Expr b = input(x, y, 2);

        Expr c_max = max(r, g, b);
        Expr c_min = min(r, g, b);
        Expr delta = c_max - c_min;

        // calculate hue
        Expr hue = select(
                delta == 0, 0,
                c_max == r, ((g - b) / delta) % 6,
                c_max == g, (b - r) / delta + 2,
                (r - g) / delta + 4
        ) * 60;

        // calculate saturation
        Expr saturation = select(
                c_max == 0, 0,
                delta / c_max
        );

        // calculate value
        Expr value = c_max;

        // rgb to hsv func
        Func hsv;
        hsv(x, y, c) = select(
                c == 0, hue,
                c == 1, saturation,
                value
        );

        // adjust hue values
        Func hsv_adjust_hue;
        hsv_adjust_hue(x, y, c) = select(
                c == 0, hsv(x, y, 0) + hue_factor * 360,
                hsv(x, y ,c)
        );

        Func clamped;
        Halide::Expr clamped_value = hsv_adjust_hue(x, y, c);
        clamped_value = select(
                clamped_value > 360, clamped_value - 360,
                clamped_value < 0, clamped_value + 360,
                clamped_value
        );

        clamped(x, y, c) = clamped_value;

        Expr H = clamped(x, y, 0);
        Expr S = clamped(x, y, 1);
        Expr V = clamped(x, y, 2);

        Expr C = S * V;
        Expr X = C * (1 - abs(operator%(H / 60, 2) - 1));
        Expr m = V - C;

        Expr R1, G1, B1;

        R1 = select(
                H < 60 || H >= 300, C,
                H < 120 || H >= 240, X,
                0
        );

        G1 = select(
                H >= 60 && H < 180, C,
                H >= 240, 0,
                X
        );

        B1 = select(
                H < 120, 0,
                H >= 180 && H < 300, C,
                X
        );

        Func rbg_to_hsv;
        rbg_to_hsv(x, y, c) = select(
                c == 0, R1 + m,
                c == 1, G1 + m,
                B1 + m
        );

        adjust_hue(x, y, c) = rbg_to_hsv(x, y, c);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            adjust_hue.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            hue_factor.set_estimate(1.0f);
        } else {
            ; // Manual scheduling
        }
    }
};

class AdjustSharpness : public Halide::Generator<AdjustSharpness> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Input<float> sharpness_factor{"sharpness_factor"};
    Output<Buffer<float, 3>> adjust_sharpness{"adjust_sharpness"};
    Var x, y, c;
    Func edge_repeated;
    Func filter_applied;
    Expr sum = 0.0f;
    Func blended;
    float kernel[3][3] = {
            {1.0f/13.0f, 1.0f/13.0f, 1.0f/13.0f},
            {1.0f/13.0f, 5.0f/13.0f, 1.0f/13.0f},
            {1.0f/13.0f, 1.0f/13.0f, 1.0f/13.0f}
    };
    void generate() {
        edge_repeated = BoundaryConditions::repeat_edge(input);
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum = operator+(sum, (edge_repeated(x + i, y + j, c) * kernel[i + 1][j + 1]));
            }
        }

        filter_applied(x, y, c) = sum;
        blended(x, y, c) = clamp(sharpness_factor * edge_repeated(x, y, c) + (1.0f - sharpness_factor) * filter_applied(x, y, c), 0.0f, 1.0f);

        adjust_sharpness(x, y, c) = select(
                x == 0 || x == (input.width() - 1), input(x, y, c),
                y == 0 || y == (input.height() - 1), input(x, y, c),
                blended(x, y, c)
        );
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            adjust_sharpness.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            sharpness_factor.set_estimate(1.4f);
        } else {
            ; // Manual scheduling
        }
    }
};

//class ElasticTransform : public Halide::Generator<ElasticTransform> {
//public:
//    Input<Buffer<float, 3>> input{"input"};
//    Input<Buffer<float, 4>> displacement{"displacement"};
//    Input<float> fill{"fill"};
//    Output<Buffer<float, 3>> elastic_transform{"elastic_transform"};
//
//    Var x, y, c, d;
//
//    void generate() {
//        Expr near_x = cast<int>(round( displacement(0,x,y,0) * input.width() /2));
//        Expr near_y = cast<int>(round( displacement(1,x,y,0) * input.height() /2));
//        Func extended_image = BoundaryConditions::constant_exterior(input,fill);
//        elastic_transform(x,y,c) = extended_image(x + near_x, y + near_y, c);
//    }
//    void schedule() {
//        if (using_autoscheduler()) {
//            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
//            displacement.set_estimates({{1, 1}, {2048, 2048}, {2048, 2048}, {3, 3}});
//            fill.set_estimate(0.0);
//            elastic_transform.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
//        }
//    }
//};

class AutoContrast : public Halide::Generator<AutoContrast> {
public:
    Input<Buffer<float, 3>> input{"input"};
    Output<Buffer<float, 3>> autocontrast{"elastic_transform"};

    Var x, y, c;
    Func min_val, max_val, scale;
    void generate() {
        RDom r(0, input.width(), 0, input.height());
        min_val(c) = minimum(input(r.x, r.y, c));
        max_val(c) = maximum(input(r.x, r.y, c));
        scale(c) = 1 / (max_val(c) - min_val(c));
        autocontrast(x, y, c) = clamp((input(x, y, c) - min_val(c)) * scale(c), 0, 1);
    }
    void schedule() {
        if (using_autoscheduler()) {
            input.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
            autocontrast.set_estimates({{2048, 2048}, {2048, 2048}, {3, 3}});
        } else {
            scale.compute_root();
            min_val.compute_root();
            max_val.compute_root();
        }
    }
};

class MatMul : public Halide::Generator<MatMul> {
public:
    Input<Buffer<float, 2>> A{"A"};
    Input<Buffer<float, 2>> B{"B"};

    Output<Buffer<float, 2>> matmul{"matmul"};
    Var x, y, p;
    void generate() {
        Var x("x"), y("y");

        Func prod("prod");
        RDom r(0, 2048);

        prod(x, y) = 0.0f;
        prod(x, y) += A(x, r.x) * B(r.x, y);

        Func out;
        out(x, y) = prod(x, y);
        matmul(x, y) = prod(x, y);
    }

    void schedule() {
        if (using_autoscheduler()) {
            A.set_estimates({{2048, 2048}, {2048, 2048}});
            B.set_estimates({{2048, 2048}, {2048, 2048}});
            matmul.set_estimates({{2048, 2048}, {2048, 2048}});
        } else { ;

        }
    }
};


HALIDE_REGISTER_GENERATOR(AdjustBrightness, adjust_brightness)
HALIDE_REGISTER_GENERATOR(AdjustBrightnessBW, adjust_brightness_bw)
HALIDE_REGISTER_GENERATOR(Invert, invert)
HALIDE_REGISTER_GENERATOR(Grayscale, rgb_to_grayscale)
HALIDE_REGISTER_GENERATOR(Hflip, hflip)
HALIDE_REGISTER_GENERATOR(Vflip, vflip)
HALIDE_REGISTER_GENERATOR(Erase, erase)
HALIDE_REGISTER_GENERATOR(Solarize, solarize)
HALIDE_REGISTER_GENERATOR(Crop, crop)
HALIDE_REGISTER_GENERATOR(AdjustGamma, adjust_gamma)
HALIDE_REGISTER_GENERATOR(AdjustSaturation, adjust_saturation)
HALIDE_REGISTER_GENERATOR(AdjustContrast, adjust_contrast)
HALIDE_REGISTER_GENERATOR(AutoContrast, autocontrast)
HALIDE_REGISTER_GENERATOR(Normalize, normalize)
HALIDE_REGISTER_GENERATOR(Posterize, posterize)
HALIDE_REGISTER_GENERATOR(AdjustHue, adjust_hue)
HALIDE_REGISTER_GENERATOR(AdjustSharpness, adjust_sharpness)
//HALIDE_REGISTER_GENERATOR(ElasticTransform, elastic_transform)
HALIDE_REGISTER_GENERATOR(MatMul, matmul)