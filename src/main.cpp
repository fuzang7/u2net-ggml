#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "u2net.h"
#include "u2net_layers.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <thread>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdio>

#if 0 // #ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

static void print_usage(const char* prog) {
    printf("Usage: %s [options] <model.gguf> <input.jpg> [output.png]\n", prog);
    printf("Options:\n");
    printf("  -t N    Number of CPU threads (default: %d)\n", std::thread::hardware_concurrency());
    printf("  -d D    Device: cpu, cuda, auto (default: auto)\n");
    printf("  -W N    Input width, must be multiple of 32 (default: 320)\n");
    printf("  -H N    Input height, must be multiple of 32 (default: 320)\n");
    printf("  -h      Show this help\n");
}

static struct letterbox_info g_letterbox_info;

void letterbox_preprocess(const char* filename, float* tensor_data, int target_w, int target_h, struct letterbox_info* info) {
    int orig_w, orig_h, c;
    unsigned char* data = stbi_load(filename, &orig_w, &orig_h, &c, 3);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        exit(1);
    }

    info->orig_w = orig_w;
    info->orig_h = orig_h;
    info->padded_w = target_w;
    info->padded_h = target_h;

    float scale_w = (float)target_w / (float)orig_w;
    float scale_h = (float)target_h / (float)orig_h;
    float img_scale = (scale_w < scale_h) ? scale_w : scale_h;

    info->scale = img_scale;
    int scaled_w = (int)(orig_w * img_scale);
    int scaled_h = (int)(orig_h * img_scale);
    info->pad_left = (target_w - scaled_w) / 2;
    info->pad_top = (target_h - scaled_h) / 2;

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    float norm_scale[3];
    float norm_offset[3];
    for (int i = 0; i < 3; ++i) {
        norm_scale[i]  = 1.0f / (255.0f * std[i]);
        norm_offset[i] = mean[i] / std[i];
    }

    for (int ch = 0; ch < 3; ++ch) {
        float fill_val = mean[ch] * norm_scale[ch] - norm_offset[ch];
        float* channel_ptr = tensor_data + ch * (target_w * target_h);
        
        for (int y = 0; y < target_h; ++y) {
            float* row_ptr = channel_ptr + y * target_w;
            for (int x = 0; x < target_w; ++x) {
                row_ptr[x] = fill_val;
            }
        }
    }

    int src_y_start = info->pad_top;
    int src_x_start = info->pad_left;
    int src_y_end = src_y_start + scaled_h;
    int src_x_end = src_x_start + scaled_w;

    for (int ch = 0; ch < 3; ++ch) {
        float* channel_ptr = tensor_data + ch * (target_w * target_h);
        
        for (int y = src_y_start; y < src_y_end; ++y) {
            int src_y = (int)((y - src_y_start) / img_scale);
            if (src_y >= orig_h) src_y = orig_h - 1;
            float* row_ptr = channel_ptr + y * target_w;

            for (int x = src_x_start; x < src_x_end; ++x) {
                int src_x = (int)((x - src_x_start) / img_scale);
                if (src_x >= orig_w) src_x = orig_w - 1;
                int src_idx = (src_y * orig_w + src_x) * 3;

                float val = (float)data[src_idx + ch];
                row_ptr[x] = val * norm_scale[ch] - norm_offset[ch];
            }
        }
    }

    stbi_image_free(data);
    printf("Letterbox preprocess: %dx%d -> %dx%d (scale=%.3f, pad=%d,%d)\n",
           orig_w, orig_h, target_w, target_h, img_scale, info->pad_left, info->pad_top);
}

void letterbox_postprocess(const char* out_name, float* data, const struct letterbox_info* info) {
    int padded_w = info->padded_w;
    int padded_h = info->padded_h;
    int orig_w = info->orig_w;
    int orig_h = info->orig_h;
    float scale = info->scale;
    int pad_left = info->pad_left;
    int pad_top = info->pad_top;

    int scaled_w = (int)(orig_w * scale);
    int scaled_h = (int)(orig_h * scale);

    if (scaled_w > padded_w - pad_left) scaled_w = padded_w - pad_left;
    if (scaled_h > padded_h - pad_top) scaled_h = padded_h - pad_top;

    // Crop the padded area to get the scaled image
    std::vector<float> scaled_float(scaled_w * scaled_h);
    for (int y = 0; y < scaled_h; ++y) {
        for (int x = 0; x < scaled_w; ++x) {
            int src_x = pad_left + x;
            int src_y = pad_top + y;
            scaled_float[y * scaled_w + x] = data[src_y * padded_w + src_x];
        }
    }

    // Scale back to original resolution using stb_image_resize2
    std::vector<float> out_float(orig_w * orig_h);
    stbir_resize_float_linear(scaled_float.data(), scaled_w, scaled_h, 0,
                              out_float.data(), orig_w, orig_h, 0,
                              STBIR_1CHANNEL);

    // Convert to uint8
    std::vector<unsigned char> out_data(orig_w * orig_h);
    for (int i = 0; i < orig_w * orig_h; ++i) {
        float val = fminf(fmaxf(out_float[i], 0.0f), 1.0f);
        out_data[i] = (unsigned char)(val * 255.0f);
    }

    stbi_write_png(out_name, orig_w, orig_h, 1, out_data.data(), orig_w);
}

int main(int argc, char** argv) {
    int n_threads = 1;  // default to single thread
    const char* device = "auto";
    int input_w = 320;
    int input_h = 320;
    
    // Parse arguments - support both old format (no -t) and new format
    int arg_idx = 1;
    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "-t") == 0 && arg_idx + 1 < argc) {
            n_threads = atoi(argv[++arg_idx]);
            if (n_threads < 1) n_threads = 1;
        } else if (strcmp(argv[arg_idx], "-d") == 0 && arg_idx + 1 < argc) {
            device = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "-W") == 0 && arg_idx + 1 < argc) {
            input_w = atoi(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-H") == 0 && arg_idx + 1 < argc) {
            input_h = atoi(argv[++arg_idx]);
        } else if (strcmp(argv[arg_idx], "-h") == 0) {
            printf("Usage: %s [options] <model.gguf> <input.jpg> [output.png]\n", argv[0]);
            printf("Options:\n");
            printf("  -t N    Number of CPU threads (default: 1)\n");
            printf("  -d D    Device: cpu, cuda, auto (default: auto)\n");
            printf("  -W N    Input width, must be multiple of 32 (default: 320)\n");
            printf("  -H N    Input height, must be multiple of 32 (default: 320)\n");
            printf("  -h      Show this help\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[arg_idx]);
            printf("Usage: %s [options] <model.gguf> <input.jpg> [output.png]\n", argv[0]);
            return 1;
        }
        arg_idx++;
    }

    if (input_w % 32 != 0 || input_h % 32 != 0) {
        fprintf(stderr, "Error: input width and height must be multiples of 32\n");
        return 1;
    }
    if (argc - arg_idx < 2) {
        printf("Usage: %s [options] <model.gguf> <input.jpg> [output.png]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[arg_idx];
    const char* img_path = argv[arg_idx + 1];
    const char* out_path = (argc - arg_idx > 2) ? argv[arg_idx + 2] : "output.png";

    // Load model
    u2net_model model;
    if (!u2net_model_load(model_path, model)) return 1;
    model.input_w = input_w;
    model.input_h = input_h;

    // Create context with no_alloc=false (allocate immediately)
    size_t mem_size = 5120ULL * 1024 * 1024;
    struct ggml_init_params params = {
        mem_size, NULL, false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create input tensor
    struct ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input_w, input_h, 3, 1);
    
    // Prepare input data
    size_t input_size = ggml_nbytes(input);
    letterbox_preprocess(img_path, (float*)input->data, input_w, input_h, &g_letterbox_info);

    printf("Building U-2-Net full graph...\n");
    u2net_out results = u2net_build_graph(ctx, model, input);

    // Build the compute graph
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, results.d0);

    printf("Graph built, n_nodes = %d\n", ggml_graph_n_nodes(gf));
    printf("Performing inference with %d threads...\n", n_threads);
    fflush(stdout);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    
    printf("Inference time: %.3f seconds\n", elapsed);

    // Get output data
    std::vector<float> output_data(ggml_nelements(results.d0));
    memcpy(output_data.data(), results.d0->data, ggml_nbytes(results.d0));

    letterbox_postprocess(out_path, output_data.data(), &g_letterbox_info);
    printf("Done! Mask saved to %s\n", out_path);

    ggml_free(ctx);
    return 0;
}
