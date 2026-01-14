
# Voice ASS — 快速开始

本文件列出硬件与软件要求、模型下载、构建步骤以及常用测试命令。按照顺序执行即可完成本项目的准备与验证。

## 硬件要求

- CPU: 支持 AVX2（推荐）
- 内存: 最少 8GB（运行大模型建议 16GB+）
- 存储: ≥10GB 可用空间用于模型文件
- 音频: 支持麦克风输入

## 软件依赖

确保已安装：

- `CMake` 3.10+
- `GCC` 8+ 或 `Clang` 10+
- `SQLite3` 3.25+
- `PortAudio`（用于音频输入）
- `whisper.cpp`（ASR）
- `llama.cpp`（LLM）

## 模型文件（示例下载命令）

将模型放到对应目录下：

ASR 模型（选其一）:

```bash
wget -P whisper.cpp/models/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
wget -P whisper.cpp/models/ https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

LLM 模型（选其一）:

```bash
wget -P llama.cpp/models/ https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q8_0.gguf
wget -P llama.cpp/models/ https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/qwen2.5-vl-7b-instruct-q8_0.gguf
```

## 构建与编译

1. 创建构建目录并进入：

```bash
mkdir -p build
cd build
```

2. 配置 CMake：

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DWITH_LLM_SUPPORT=ON \
  -DWITH_ASR_SUPPORT=ON
```

3. 编译（使用所有 CPU 内核）：

```bash
make -j$(nproc)
# 或指定核心： make -j4
```

4. 验证生成文件：

```bash
ls -la build/test/test_*
```

## 常用测试命令（快速索引）

- 音频输入：列出设备

```bash
./test/test_audio_input --list
```

- 音频输入：录音 5 秒并保存为 `test.wav`

```bash
./test/test_audio_input --record 5
```

- VAD（语音活动检测）：运行全部测试

```bash
./test/test_vad --all
```

- ASR 引擎：使用指定模型运行所有测试

```bash
ASR_MODEL="whisper.cpp/models/ggml-large-v3-turbo.bin"
./build/test/test_asr_engine --model $ASR_MODEL --all
```

- 实时音频到 ASR 流水线（示例）

```bash
./build/test/test_audio_asr_pipeline --model $ASR_MODEL --vad-threshold 0.5
```

- SQLite 数据库测试（切换到 voice_ass 项目目录）

```bash
cd /home/ye/project/vass/voice_ass
./build/test/test_sqlite_db --all
```

- LLM 客户端测试（示例）

```bash
LLM_MODEL="llama.cpp/models/Qwen2.5-7B-Instruct-Q8_0.gguf"
./build/test/test_llm_client --model $LLM_MODEL --init
```

## 完整流水线示例命令

- LOCAL 模式（仅本地知识库）

```bash
./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --mode local \
  --language zh \
  --log-level info \
  --duration 180
```

- LLM 模式（使用 LLM 回答）

```bash
./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --llm-model llama.cpp/models/Qwen2.5-7B-Instruct-Q8_0.gguf \
  --mode llm \
  --language auto \
  --log-level debug \
  --device 0
```

- HYBRID 模式（优先本地知识库）

```bash
./build/test/test_full_pipeline \
  --asr-model whisper.cpp/models/ggml-large-v3-turbo.bin \
  --llm-model llama.cpp/models/Qwen3-8B-Q8_0.gguf \
  --mode hybrid \
  --confidence-threshold 0.7 \
  --max-local-results 3
```

更多测试选项请查看 `test/` 目录下各可执行文件的 `--help` 输出。

---

如果需要，我可以：

- 把其中的示例命令改为针对你的机器的绝对路径。
- 添加 Windows 或 macOS 的额外说明。
