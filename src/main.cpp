#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "u2net.h"
#include "u2net_layers.h"
#include "ggml-cpu.h"
#include <thread>
#include <cstring>

static void print_usage(const char* prog) {
    printf("Usage: %s [options] <model.gguf> <input.jpg> [output.png]\n", prog);
    printf("Options:\n");
    printf("  -t N    Number of threads (default: %d)\n", std::thread::hardware_concurrency());
    printf("  -h      Show this help\n");
}

void preprocess(const char* filename, struct ggml_tensor* input) {
    int w, h, c;
    unsigned char* data = stbi_load(filename, &w, &h, &c, 3);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        exit(1);
    }

    float* tensor_data = (float*)input->data;
    const int target_w = 320;
    const int target_h = 320;

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3]  = {0.229f, 0.224f, 0.225f};

    float scale[3];
    float offset[3];
    for (int i = 0; i < 3; ++i) {
        scale[i]  = 1.0f / (255.0f * std[i]);
        offset[i] = mean[i] / std[i];
    }

    for (int c = 0; c < 3; ++c) {
        float* channel_ptr = tensor_data + c * (target_w * target_h);
        
        for (int y = 0; y < target_h; ++y) {
            int src_y = y * h / target_h;
            float* row_ptr = channel_ptr + y * target_w;

            for (int x = 0; x < target_w; ++x) {
                int src_x = x * w / target_w;
                int src_idx = (src_y * w + src_x) * 3;

                float val = (float)data[src_idx + c];
                row_ptr[x] = val * scale[c] - offset[c];
            }
        }
    }

    stbi_image_free(data);
    printf("Preprocess completed: Image normalized and injected into WHCN tensor.\n");
}

void postprocess(const char* out_name, struct ggml_tensor* output) {
    std::vector<unsigned char> out_data(320 * 320);
    float* data = (float*)output->data;

    for (int i = 0; i < 320 * 320; ++i) {
        out_data[i] = (unsigned char)(data[i] * 255.0f);
    }
    stbi_write_png(out_name, 320, 320, 1, out_data.data(), 320);
}

int main(int argc, char** argv) {
    int n_threads = std::thread::hardware_concurrency();
    
    int arg_idx = 1;
    while (arg_idx < argc && argv[arg_idx][0] == '-') {
        if (strcmp(argv[arg_idx], "-t") == 0 && arg_idx + 1 < argc) {
            n_threads = atoi(argv[++arg_idx]);
            if (n_threads < 1) n_threads = 1;
        } else if (strcmp(argv[arg_idx], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[arg_idx]);
            print_usage(argv[0]);
            return 1;
        }
        arg_idx++;
    }

    if (argc - arg_idx < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[arg_idx];
    const char* img_path = argv[arg_idx + 1];
    const char* out_path = (argc - arg_idx > 2) ? argv[arg_idx + 2] : "output.png";

    printf("Threads: %d\n", n_threads);

    u2net_model model;
    if (!u2net_model_load(model_path, model)) return 1;

    size_t mem_size = 5120ULL * 1024 * 1024;
    struct ggml_init_params params = {
        mem_size, NULL, false,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 320, 320, 3, 1);
    preprocess(img_path, input);

    printf("Building U-2-Net full graph...\n");
    u2net_out results = u2net_build_graph(ctx, model, input);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, results.d0);
    
    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, nullptr);
    if (cplan.work_size > 0) {
        cplan.work_data = (uint8_t*)malloc(cplan.work_size);
        printf("Work buffer: %.2f MB\n", cplan.work_size / 1024.0 / 1024.0);
    }
    
    printf("Performing inference with %d thread(s)...\n", n_threads);
    ggml_graph_compute(gf, &cplan);
    
    if (cplan.work_data) {
        free(cplan.work_data);
    }

    postprocess(out_path, results.d0);
    printf("Done! Mask saved to %s\n", out_path);

    ggml_free(ctx);
    return 0;
}