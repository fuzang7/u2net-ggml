#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "u2net.h"
#include "u2net_layers.h"

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
    if (argc < 3) {
        printf("Usage: %s <model.gguf> <input.jpg> [output.png]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* img_path = argv[2];
    const char* out_path = (argc > 3) ? argv[3] : "output.png";

    u2net_model model;
    if (!u2net_model_load(model_path, model)) return 1;

    struct ggml_init_params params = {
        /*.mem_size   = */ 5120ULL * 1024 * 1024,
        /*.mem_buffer = */ NULL,
        /*.no_alloc   = */ false,
    };
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 320, 320, 3, 1);
    preprocess(img_path, input);

    printf("Building U-2-Net full graph...\n");
    u2net_out results = u2net_build_graph(ctx, model, input);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, results.d0);
    
    printf("Performing inference...\n");
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    postprocess(out_path, results.d0);
    printf("Done! Mask saved to %s\n", out_path);

    ggml_free(ctx);
    return 0;
}