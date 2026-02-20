# U-2-Net GGML

U-2-Net inference implementation using GGML.

| Input Image | Output Image |
| :---: | :---: |
| <img src="./test_images/human.jpg" width="300"> | <img src="./test_images/output.png" width="300"> |

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./u2net-cli [options] <model.gguf> <input.jpg> [output.png]
```

**Arguments:**
- `model.gguf`: U-2-Net model in GGUF format
- `input.jpg`: Input image
- `output.png`: Output mask (default: output.png)

**Options:**
- `-t N`: Number of threads (default: CPU cores)
- `-h`: Show help

## Model Conversion

Use `convert.py` to convert PyTorch model to GGUF format:

```bash
python convert.py
```

## Requirements

- CMake 3.14+
- C++17 compiler
- GGML (included as submodule)
