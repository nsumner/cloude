# cloude

A CLI tool that runs Claude Code against a local llama.cpp model.


## Installation

### Prerequisites

- Python 3.14+
- A local LLM model in GGUF format (e.g., from HuggingFace)
- llama-server from [llama.cpp](https://github.com/ggml-org/llama.cpp)
- The `claude` CLI tool installed and available in your PATH

### Pre-Setup

#### **Installing [llama.cpp](https://github.com/ggml-org/llama.cpp)**

If you don't have [llama.cpp](https://github.com/ggml-org/llama.cpp) installed,
that comes first. Per the [unsloth](https://unsloth.ai/docs/models/qwen3.5) guides:

1. ```bash
   apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev nvidia-cuda-toolkit python3-huggingface-hub
   ```
2. ```bash
   git clone https://github.com/ggml-org/llama.cpp
   ```
3. ```bash
   cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
   ```
4. ```bash
   cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
   ```
5. ```bash
   cp llama.cpp/build/bin/llama-* ~/.local/bin/
   ```

This assumes that `~/.local/bin` is already on your path. If not, make it and add it to your path.

#### **Downloading models**

Install the `hf` package if necessary:

```bash
python3 -m venv ~/.venv/hf
source ~/.venv/hf/bin/activate
pip install huggingface_hub hf_transfer
```

Download appropriate models from [huggingface](https://huggingface.co/).
The `--local-dir` argument determines where the model will be saved.

1. Qwen 27B 2-bit
   ```bash
   hf download unsloth/Qwen3.5-27B-GGUF --local-dir ~/models/unsloth/Qwen3.5-27B-GGUF --include "*Q2_K_M*"
   ```
2. Omnicoder 3-bit
   ```bash
   hf download Tesslate/OmniCoder-2-9B-GGUF    --local-dir Tesslate/OmniCoder-2-9B-GGUF/    --include "*q3_k_l*"
   ```
3. Omnicoder 5-bit
   ```bash
   hf download Tesslate/OmniCoder-2-9B-GGUF    --local-dir Tesslate/OmniCoder-2-9B-GGUF/    --include "*q5_k_m*"
   ```


### Setup

1. **Install as a local tool**:
   ```bash
   uv tool install --editable ./
   ```

2. **Download a GGUF model** and place it in your models directory.

3. **Configure models**: The default configuration includes pre-configured models;
  add your own by editing the `CONFIGS` dictionary in `main.py`. You can also
  configure their modes.

## Usage

### List Available Configurations

```bash
cloude --list
```

### Run Claude with the Default Local Model

```bash
cloude
```

### Run Claude with a Specific Local Model

```bash
cloude --config <model-name>
```

### Passthrough Options

```bash
cloude -p "Say hello"
```

### Server-Only Mode

Start the local LLM server without running Claude:

```bash
cloude --config <model-name> --server
```

### Override Port

```bash
cloude --config <model-name> --port <PORT>
```

## Configuring New Models

The default models are defined in `main.py`. To add custom models, edit the `CONFIGS` dictionary:

```python
CONFIGS: Dict[str, LlamaConfig] = {
    "my-model": LlamaConfig(
        name="My Model",
        gguf_path=Path("/path/to/model.gguf"),
        n_ctx=32768,
    ),
}
```

## Current Models

| Config | Model | Context Size |
|--------|-------|-------------|
| `omni-small` | OmniCoder-2-9B-Q3KL | 65,536 |
| `omni-medium` | OmniCoder-2-9B-Q5KM | 65,536 |
| `qwen-27-desk` | Qwen3.5-27B-UD-Q2 | 175,000 |


## License

MIT
