#pragma once
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include <map>
#include <string>
#include <vector>

struct u2net_model {
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;
    
    std::map<std::string, struct ggml_tensor *> tensors;

    int hparams_c = 3;
    int hparams_h = 320;
    int hparams_w = 320;
};

bool u2net_model_load(const char * fname, u2net_model & model);
struct ggml_tensor * u2net_get_tensor(const u2net_model & model, const std::string & name);