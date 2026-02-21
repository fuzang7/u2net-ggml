#include "u2net.h"
#include <cstdio>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>

void u2net_fuse_bn(u2net_model & model) {
    printf("Starting BN fusion...\n");
    int fused_count = 0;
    
    for (auto const& [name, t_conv_w] : model.tensors) {
        size_t pos = name.find(".conv_s1.weight");
        if (pos == std::string::npos) continue;

        std::string prefix = name.substr(0, pos);
        
        struct ggml_tensor * conv_b = model.tensors[prefix + ".conv_s1.bias"];
        struct ggml_tensor * bn_w   = model.tensors[prefix + ".bn_s1.weight"];
        struct ggml_tensor * bn_b   = model.tensors[prefix + ".bn_s1.bias"];
        struct ggml_tensor * bn_m   = model.tensors[prefix + ".bn_s1.running_mean"];
        struct ggml_tensor * bn_v   = model.tensors[prefix + ".bn_s1.running_var"];

        if (!conv_b || !bn_w || !bn_b || !bn_m || !bn_v) continue;
        if (!t_conv_w->data || !conv_b->data || !bn_w->data || !bn_b->data || !bn_m->data || !bn_v->data) continue;

        const float eps = 1e-5f;
        int64_t oc = t_conv_w->ne[3];
        int64_t ic = t_conv_w->ne[2];
        int64_t kh = t_conv_w->ne[1];
        int64_t kw = t_conv_w->ne[0];

        float* w_data = (float*)t_conv_w->data;
        float* b_data = (float*)conv_b->data;
        float* gamma  = (float*)bn_w->data;
        float* beta   = (float*)bn_b->data;
        float* mean   = (float*)bn_m->data;
        float* var    = (float*)bn_v->data;

        for (int i = 0; i < oc; ++i) {
            float scale = gamma[i] / std::sqrt(var[i] + eps);
            
            int64_t filter_size = ic * kh * kw;
            for (int j = 0; j < filter_size; ++j) {
                w_data[i * filter_size + j] *= scale;
            }

            b_data[i] = (b_data[i] - mean[i]) * scale + beta[i];
        }
        
        fused_count++;
    }
    printf("Successfully fused %d BN layers into Conv layers.\n", fused_count);
}

bool u2net_model_load(const char * fname, u2net_model & model) {
    model.ctx_data = NULL;
    model.ctx_gguf = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &model.ctx_data,
    };

    model.ctx_gguf = gguf_init_from_file(fname, params);
    if (!model.ctx_gguf) {
        fprintf(stderr, "Error: failed to load '%s'\n", fname);
        return false;
    }

    struct ggml_tensor * t = ggml_get_first_tensor(model.ctx_data);
    while (t != NULL) {
        model.tensors[t->name] = t;
        t = ggml_get_next_tensor(model.ctx_data, t);
    }
    
    printf("u2net_load: loaded %zu tensors from '%s'\n", model.tensors.size(), fname);
    
    u2net_fuse_bn(model);
    return true;
}

struct ggml_tensor * u2net_get_tensor(const u2net_model & model, const std::string & name) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        fprintf(stderr, "Warning: tensor '%s' not found.\n", name.c_str());
        return nullptr;
    }
    return it->second;
}
