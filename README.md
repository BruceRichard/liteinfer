<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# LiteInfer

一个**轻量级但高性能**的大语言模型推理框架。

---

LiteInfer 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，旨在揭开现代 LLM 服务系统的复杂性。代码库仅约 **~5,000 行 Python**，既是一个功能完备的推理引擎，也是研究人员和开发者可读性极高的参考实现。

## ✨ 核心特性

- **高性能**：通过先进优化技术，实现业界领先的吞吐量和延迟。
- **轻量级 & 高可读性**：代码整洁、模块化、完整类型标注，易于理解和二次开发。
- **高级优化**：
  - **Radix Cache**：复用请求间共享前缀的 KV 缓存。
  - **Chunked Prefill（分块预填充）**：降低长上下文推理的峰值显存占用。
  - **Overlap Scheduling（重叠调度）**：将 CPU 调度开销隐藏在 GPU 计算中。
  - **Tensor Parallelism（张量并行）**：将推理扩展到多 GPU。
  - **优化 Kernel**：集成 **FlashAttention** 和 **FlashInfer**，最大化推理效率。
  - ...

## 🚀 快速开始

> **⚠️ 平台支持**：LiteInfer 目前仅支持 **Linux**（x86_64 和 aarch64）。由于依赖 Linux 专属的 CUDA kernel（`sgl-kernel`、`flashinfer`），Windows 和 macOS 不受支持。建议 Windows 用户使用 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install)，或使用 Docker 进行跨平台兼容。

### 1. 环境配置

推荐使用 `uv` 进行快速可靠的安装（`uv` 与 `conda` 不冲突）。

```bash
# 创建虚拟环境（推荐 Python 3.10+）
uv venv --python=3.12
source .venv/bin/activate
```

**前置条件**：LiteInfer 依赖 JIT 编译的 CUDA kernel。请确保已安装 **NVIDIA CUDA Toolkit**，且版本与驱动匹配。可通过 `nvidia-smi` 查看驱动的 CUDA 兼容版本。

### 2. 安装

从源码直接安装 LiteInfer：

```bash
git clone https://github.com/sgl-project/liteinfer.git
cd liteinfer && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 在 Windows 上安装（WSL2）</b></summary>

由于 LiteInfer 依赖 Linux 专属组件，Windows 用户应使用 WSL2：

1. **安装 WSL2**（如尚未安装）：
   ```powershell
   # 在 PowerShell 中（以管理员身份运行）
   wsl --install
   ```

2. **在 WSL2 中安装 CUDA**：
   - 参考 [NVIDIA 的 WSL2 CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保 Windows GPU 驱动支持 WSL2

3. **在 WSL2 中安装 LiteInfer**：
   ```bash
   # 在 WSL2 终端中
   git clone https://github.com/sgl-project/liteinfer.git
   cd liteinfer && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从 Windows 访问**：服务器启动后，可通过 `http://localhost:8000` 从 Windows 浏览器和应用中访问。

</details>

<details>
<summary><b>🐳 使用 Docker 运行</b></summary>

**前置条件**：
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **构建 Docker 镜像**：
   ```bash
   docker build -t liteinfer .
   ```

2. **启动服务**：
   ```bash
   docker run --gpus all -p 1919:1919 \
       liteinfer --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **交互式 Shell 模式**：
   ```bash
   docker run -it --gpus all \
       liteinfer --model Qwen/Qwen3-0.6B --shell
   ```

4. **使用 Docker 卷持久化缓存**（推荐，可加速后续启动）：
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       liteinfer --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. 在线推理

一行命令即可启动 OpenAI 兼容的 API 服务。

```bash
# 在单 GPU 上部署 Qwen/Qwen3-0.6B
python -m liteinfer --model "Qwen/Qwen3-0.6B"

# 在 4 GPU 上使用张量并行部署 meta-llama/Llama-3.1-70B-Instruct，端口 30000
python -m liteinfer --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

服务启动后，可以使用标准工具如 `curl` 或任意 OpenAI 兼容客户端发送请求。

> **💡 使用 ModelScope 下载模型**：如果从 HuggingFace 下载模型遇到网络问题，可以通过 `--model-source modelscope` 从 ModelScope 下载：
> ```bash
> # 使用 ModelScope 下载并部署
> python -m liteinfer --model "Qwen/Qwen3-0.6B" --model-source modelscope
> ```
>
> 也可以先用 `modelscope` CLI 预先下载模型，再指定本地路径使用：
> ```bash
> pip install modelscope
> modelscope download Qwen/Qwen3-0.6B --local_dir ./models/Qwen3-0.6B
> python -m liteinfer --model "./models/Qwen3-0.6B"
> ```

### 4. 交互式 Shell

添加 `--shell` 参数即可在终端中直接与模型对话。

```bash
python -m liteinfer --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

使用 `/reset` 命令可以清空对话历史。

## Benchmark

### 离线推理

详见 [bench.py](./benchmark/offline/bench.py)。设置 `LITEINFER_DISABLE_OVERLAP_SCHEDULING=1` 可进行重叠调度的消融实验。

测试配置：

- 硬件：1×H200 GPU。
- 模型：Qwen3-0.6B、Qwen3-14B
- 请求总数：256 条
- 输入长度：100-1024 tokens 随机采样
- 输出长度：100-1024 tokens 随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理

详见 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置：

- 硬件：4×H200 GPU，NVLink 互联。
- 模型：Qwen3-32B
- 数据集：[Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，回放前 1000 条请求。

启动命令：

```bash
# LiteInfer
python -m liteinfer --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **提示**：如果从 HuggingFace 下载模型遇到网络问题，可以尝试使用 `--model-source modelscope` 从 ModelScope 下载：
> ```bash
> python -m liteinfer --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 了解更多

- **[功能详解](./docs/features.md)**：探索所有可用功能和命令行参数。
- **[系统架构](./docs/structures.md)**：深入了解 LiteInfer 的设计和数据流。
