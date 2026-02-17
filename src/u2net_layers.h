#pragma once
#include "ggml.h"
#include "u2net.h"
#include <cmath>

inline struct ggml_tensor * get_tensor_checked(const u2net_model & model, const std::string & name) {
    return u2net_get_tensor(model, name);
}


static struct ggml_tensor * u2net_rebnconv(
    struct ggml_context * ctx,
    const u2net_model & model,
    struct ggml_tensor * input,
    const std::string & prefix,
    int dilation = 1
) {
    struct ggml_tensor * w = get_tensor_checked(model, prefix + ".conv_s1.weight");
    struct ggml_tensor * b = get_tensor_checked(model, prefix + ".conv_s1.bias");
    
    struct ggml_tensor * cur = ggml_conv_2d(ctx, w, input, 1, 1, dilation, dilation, dilation, dilation);

    int64_t nc = cur->ne[2];
    struct ggml_tensor * b_v = ggml_reshape_4d(ctx, b, 1, 1, nc, 1);
    cur = ggml_add(ctx, cur, ggml_repeat(ctx, b_v, cur));

    return ggml_relu(ctx, cur);
}

static struct ggml_tensor * u2net_maxpool(struct ggml_context * ctx, struct ggml_tensor * input) {
    return ggml_pool_2d(ctx, input, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
}

static struct ggml_tensor * u2net_upsample(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * target) {
    int scale_w = target->ne[0] / input->ne[0];
    int scale_h = target->ne[1] / input->ne[1];

    if (scale_w != scale_h) {
        fprintf(stderr, "Error: Anisotropic scaling not supported.\n");
        return NULL;
    }
    
    if (scale_w == 1) return input;

    return ggml_upscale(ctx, input, scale_w, GGML_SCALE_MODE_NEAREST);
}


static struct ggml_tensor * u2net_concat(
    struct ggml_context * ctx, 
    struct ggml_tensor * a, 
    struct ggml_tensor * b, 
    const char* label = "untagged"
) {
    bool mismatch = false;
    for (int i = 0; i < 4; i++) {
        if (i == 2) continue;
        if (a->ne[i] != b->ne[i]) mismatch = true;
    }

    if (mismatch) {
        fprintf(stderr, "\n[CONCAT FATAL] Mismatch at %s\n", label);
        fprintf(stderr, "  Tensor A: [%ld, %ld, %ld, %ld]\n", a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
        fprintf(stderr, "  Tensor B: [%ld, %ld, %ld, %ld]\n", b->ne[0], b->ne[1], b->ne[2], b->ne[3]);
        fflush(stderr);
    }
    return ggml_concat(ctx, a, b, 2);
}

static struct ggml_tensor * u2net_rsu7(
    struct ggml_context * ctx,
    const u2net_model & model,
    struct ggml_tensor * input, 
    const std::string & prefix
) {
    auto print_dim = [](const char* name, struct ggml_tensor* t) {
        printf("  [DIM] %-15s: %ld x %ld x %ld x %ld\n", name, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    };

    struct ggml_tensor * hxin = u2net_rebnconv(ctx, model, input, prefix + ".rebnconvin");

    struct ggml_tensor * hx1 = u2net_rebnconv(ctx, model, hxin, prefix + ".rebnconv1");
    struct ggml_tensor * hx2 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx1), prefix + ".rebnconv2");
    struct ggml_tensor * hx3 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx2), prefix + ".rebnconv3");
    struct ggml_tensor * hx4 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx3), prefix + ".rebnconv4");
    struct ggml_tensor * hx5 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx4), prefix + ".rebnconv5");
    struct ggml_tensor * hx6 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx5), prefix + ".rebnconv6");
    
    struct ggml_tensor * hx7 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx6), prefix + ".rebnconv7", 2);

    printf("--- RSU7 Decoder Trace (%s) ---\n", prefix.c_str());

    auto up_and_concat = [&](struct ggml_tensor* low, struct ggml_tensor* high, const std::string& p) {
        struct ggml_tensor* up = u2net_upsample(ctx, low, high);
        
        if (up->ne[0] != high->ne[0] || up->ne[1] != high->ne[1]) {
            printf("[CRITICAL] Shape mismatch before concat at %s\n", p.c_str());
            print_dim("Upsampled", up);
            print_dim("Skip Conn", high);
        }
        
        return u2net_rebnconv(ctx, model, u2net_concat(ctx, up, high), p);
    };

    struct ggml_tensor * hx6d = up_and_concat(hx7, hx6, prefix + ".rebnconv6d");
    struct ggml_tensor * hx5d = up_and_concat(hx6d, hx5, prefix + ".rebnconv5d");
    struct ggml_tensor * hx4d = up_and_concat(hx5d, hx4, prefix + ".rebnconv4d");
    struct ggml_tensor * hx3d = up_and_concat(hx4d, hx3, prefix + ".rebnconv3d");
    struct ggml_tensor * hx2d = up_and_concat(hx3d, hx2, prefix + ".rebnconv2d");
    struct ggml_tensor * hx1d = up_and_concat(hx2d, hx1, prefix + ".rebnconv1d");

    if (hx1d->ne[0] != hxin->ne[0] || hx1d->ne[1] != hxin->ne[1] || hx1d->ne[2] != hxin->ne[2]) {
        printf("[CRITICAL] Final residual mismatch!\n");
        print_dim("hx1d", hx1d);
        print_dim("hxin", hxin);
    }
    printf("--- RSU7 Final Residual Add ---\n");
    printf("  hx1d: [%ld, %ld, %ld, %ld]\n", hx1d->ne[0], hx1d->ne[1], hx1d->ne[2], hx1d->ne[3]);
    printf("  hxin: [%ld, %ld, %ld, %ld]\n", hxin->ne[0], hxin->ne[1], hxin->ne[2], hxin->ne[3]);
    fflush(stdout);

    return ggml_add(ctx, hx1d, hxin);
}

static struct ggml_tensor * u2net_rsu6(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input, const std::string & prefix) {
    struct ggml_tensor * hxin = u2net_rebnconv(ctx, model, input, prefix + ".rebnconvin");
    struct ggml_tensor * hx1 = u2net_rebnconv(ctx, model, hxin, prefix + ".rebnconv1");
    struct ggml_tensor * hx2 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx1), prefix + ".rebnconv2");
    struct ggml_tensor * hx3 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx2), prefix + ".rebnconv3");
    struct ggml_tensor * hx4 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx3), prefix + ".rebnconv4");
    struct ggml_tensor * hx5 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx4), prefix + ".rebnconv5");

    struct ggml_tensor * hx6 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx5), prefix + ".rebnconv6", 2);

    struct ggml_tensor * hx5d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx6, hx5), hx5), prefix + ".rebnconv5d");
    struct ggml_tensor * hx4d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx5d, hx4), hx4), prefix + ".rebnconv4d");
    struct ggml_tensor * hx3d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx4d, hx3), hx3), prefix + ".rebnconv3d");
    struct ggml_tensor * hx2d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx3d, hx2), hx2), prefix + ".rebnconv2d");
    struct ggml_tensor * hx1d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx2d, hx1), hx1), prefix + ".rebnconv1d");

    return ggml_add(ctx, hx1d, hxin);
}

static struct ggml_tensor * u2net_rsu5(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input, const std::string & prefix) {
    struct ggml_tensor * hxin = u2net_rebnconv(ctx, model, input, prefix + ".rebnconvin");
    struct ggml_tensor * hx1 = u2net_rebnconv(ctx, model, hxin, prefix + ".rebnconv1");
    struct ggml_tensor * hx2 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx1), prefix + ".rebnconv2");
    struct ggml_tensor * hx3 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx2), prefix + ".rebnconv3");
    struct ggml_tensor * hx4 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx3), prefix + ".rebnconv4");

    struct ggml_tensor * hx5 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx4), prefix + ".rebnconv5", 2);

    struct ggml_tensor * hx4d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx5, hx4), hx4), prefix + ".rebnconv4d");
    struct ggml_tensor * hx3d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx4d, hx3), hx3), prefix + ".rebnconv3d");
    struct ggml_tensor * hx2d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx3d, hx2), hx2), prefix + ".rebnconv2d");
    struct ggml_tensor * hx1d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx2d, hx1), hx1), prefix + ".rebnconv1d");

    return ggml_add(ctx, hx1d, hxin);
}

static struct ggml_tensor * u2net_rsu4(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input, const std::string & prefix) {
    struct ggml_tensor * hxin = u2net_rebnconv(ctx, model, input, prefix + ".rebnconvin");
    struct ggml_tensor * hx1 = u2net_rebnconv(ctx, model, hxin, prefix + ".rebnconv1");
    struct ggml_tensor * hx2 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx1), prefix + ".rebnconv2");
    struct ggml_tensor * hx3 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx2), prefix + ".rebnconv3");

    struct ggml_tensor * hx4 = u2net_rebnconv(ctx, model, u2net_maxpool(ctx, hx3), prefix + ".rebnconv4", 2);

    struct ggml_tensor * hx3d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx4, hx3), hx3), prefix + ".rebnconv3d");
    struct ggml_tensor * hx2d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx3d, hx2), hx2), prefix + ".rebnconv2d");
    struct ggml_tensor * hx1d = u2net_rebnconv(ctx, model, u2net_concat(ctx, u2net_upsample(ctx, hx2d, hx1), hx1), prefix + ".rebnconv1d");

    return ggml_add(ctx, hx1d, hxin);
}

static struct ggml_tensor * u2net_rsu4f(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input, const std::string & prefix) {
    struct ggml_tensor * hxin = u2net_rebnconv(ctx, model, input, prefix + ".rebnconvin");

    struct ggml_tensor * hx1 = u2net_rebnconv(ctx, model, hxin, prefix + ".rebnconv1");
    struct ggml_tensor * hx2 = u2net_rebnconv(ctx, model, hx1, prefix + ".rebnconv2", 2);
    struct ggml_tensor * hx3 = u2net_rebnconv(ctx, model, hx2, prefix + ".rebnconv3", 4);

    struct ggml_tensor * hx4 = u2net_rebnconv(ctx, model, hx3, prefix + ".rebnconv4", 8);

    struct ggml_tensor * hx3d = u2net_rebnconv(ctx, model, u2net_concat(ctx, hx4, hx3), prefix + ".rebnconv3d", 4);
    struct ggml_tensor * hx2d = u2net_rebnconv(ctx, model, u2net_concat(ctx, hx3d, hx2), prefix + ".rebnconv2d", 2);
    struct ggml_tensor * hx1d = u2net_rebnconv(ctx, model, u2net_concat(ctx, hx2d, hx1), prefix + ".rebnconv1d");

    return ggml_add(ctx, hx1d, hxin);
}


struct u2net_out {
    struct ggml_tensor * d0;
    struct ggml_tensor * d1, * d2, * d3, * d4, * d5, * d6;
};

static struct ggml_tensor * side_out(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input, const std::string & name) {
    struct ggml_tensor * w = get_tensor_checked(model, name + ".weight");
    struct ggml_tensor * b = get_tensor_checked(model, name + ".bias");
    
    int64_t k = w->ne[0];
    int p = (k == 3) ? 1 : 0; 

    struct ggml_tensor * cur = ggml_conv_2d(ctx, w, input, 1, 1, p, p, 1, 1);
    
    int64_t nc = cur->ne[2];
    struct ggml_tensor * b_v = ggml_reshape_4d(ctx, b, 1, 1, nc, 1);
    return ggml_add(ctx, cur, ggml_repeat(ctx, b_v, cur));
}

u2net_out u2net_build_graph(struct ggml_context * ctx, const u2net_model & model, struct ggml_tensor * input) {
    struct ggml_tensor * s1 = u2net_rsu7(ctx, model, input, "stage1");
    struct ggml_tensor * s2 = u2net_rsu6(ctx, model, u2net_maxpool(ctx, s1), "stage2");
    struct ggml_tensor * s3 = u2net_rsu5(ctx, model, u2net_maxpool(ctx, s2), "stage3");
    struct ggml_tensor * s4 = u2net_rsu4(ctx, model, u2net_maxpool(ctx, s3), "stage4");
    struct ggml_tensor * s5 = u2net_rsu4f(ctx, model, u2net_maxpool(ctx, s4), "stage5");
    struct ggml_tensor * s6 = u2net_rsu4f(ctx, model, u2net_maxpool(ctx, s5), "stage6");

    struct ggml_tensor * s6up = u2net_upsample(ctx, s6, s5);
    struct ggml_tensor * s5d = u2net_rsu4f(ctx, model, u2net_concat(ctx, s6up, s5), "stage5d");

    struct ggml_tensor * s5dup = u2net_upsample(ctx, s5d, s4);
    struct ggml_tensor * s4d = u2net_rsu4(ctx, model, u2net_concat(ctx, s5dup, s4), "stage4d");

    struct ggml_tensor * s4dup = u2net_upsample(ctx, s4d, s3);
    struct ggml_tensor * s3d = u2net_rsu5(ctx, model, u2net_concat(ctx, s4dup, s3), "stage3d");

    struct ggml_tensor * s3dup = u2net_upsample(ctx, s3d, s2);
    struct ggml_tensor * s2d = u2net_rsu6(ctx, model, u2net_concat(ctx, s3dup, s2), "stage2d");

    struct ggml_tensor * s2dup = u2net_upsample(ctx, s2d, s1);
    struct ggml_tensor * s1d = u2net_rsu7(ctx, model, u2net_concat(ctx, s2dup, s1), "stage1d");

    u2net_out out;
    out.d1 = side_out(ctx, model, s1d, "side1");
    
    out.d2 = u2net_upsample(ctx, side_out(ctx, model, s2d, "side2"), out.d1);
    out.d3 = u2net_upsample(ctx, side_out(ctx, model, s3d, "side3"), out.d1);
    out.d4 = u2net_upsample(ctx, side_out(ctx, model, s4d, "side4"), out.d1);
    out.d5 = u2net_upsample(ctx, side_out(ctx, model, s5d, "side5"), out.d1);
    out.d6 = u2net_upsample(ctx, side_out(ctx, model, s6,  "side6"), out.d1);

    printf("Starting side output fusion...\n"); fflush(stdout);
    
    struct ggml_tensor * combined = u2net_concat(ctx, out.d1, out.d2, "Fusion d1+d2");
    combined = u2net_concat(ctx, combined, out.d3, "Fusion +d3");
    combined = u2net_concat(ctx, combined, out.d4, "Fusion +d4");
    combined = u2net_concat(ctx, combined, out.d5, "Fusion +d5");
    combined = u2net_concat(ctx, combined, out.d6, "Fusion +d6");

    out.d0 = side_out(ctx, model, combined, "outconv");
    
    out.d0 = ggml_sigmoid(ctx, out.d0);

    return out;
}

