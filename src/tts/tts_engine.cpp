#include "tts_engine.h"
#include "text_frontend.h"
#include "../utils/logger.h"
#include <stdexcept>
#include <algorithm>

namespace voice_assistant {
namespace tts {

TTSEngine::TTSEngine(const Config& config)
    : config_(config)
    , initialized_(false)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

TTSEngine::~TTSEngine() = default;

bool TTSEngine::initialize() {
    try {
        LOG_INFO("Initializing TTS engine...");

        // 创建 ONNX Runtime 环境
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TTSEngine");

        // 创建 session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(config_.num_threads);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 配置 CUDA 执行提供者
        if (config_.use_gpu) {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.gpu_device_id;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.arena_extend_strategy = 1;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
            LOG_INFO("CUDA execution provider enabled (device ", config_.gpu_device_id, ")");
#else
            LOG_WARN("CUDA support not compiled, using CPU");
#endif
        }

        // 加载三个模型
        std::string llm_path = config_.model_dir + "/cosyvoice2_llm.onnx";
        std::string flow_path = config_.model_dir + "/cosyvoice2_flow.onnx";
        std::string vocoder_path = config_.model_dir + "/cosyvoice2_vocoder.onnx";

        LOG_INFO("Loading LLM model: ", llm_path);
        llm_session_ = std::make_unique<Ort::Session>(
            *env_, llm_path.c_str(), *session_options_);

        LOG_INFO("Loading Flow model: ", flow_path);
        flow_session_ = std::make_unique<Ort::Session>(
            *env_, flow_path.c_str(), *session_options_);

        LOG_INFO("Loading Vocoder model: ", vocoder_path);
        vocoder_session_ = std::make_unique<Ort::Session>(
            *env_, vocoder_path.c_str(), *session_options_);

        // 初始化文本前端
        text_frontend_ = std::make_unique<TextFrontend>();

        // 初始化音频变换（用于 iSTFT）
        // HiFT 使用 n_fft=16, hop_length=4
        audio_transform_ = std::make_unique<utils::AudioTransform>(16, 4, "hann");
        LOG_INFO("AudioTransform initialized (n_fft=16, hop_length=4)");

        initialized_ = true;
        LOG_INFO("TTS engine initialized successfully");
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime error: ", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error initializing TTS engine: ", e.what());
        return false;
    }
}

std::vector<float> TTSEngine::synthesize(const std::string& text) {
    if (!initialized_) {
        LOG_ERROR("TTS engine not initialized");
        return std::vector<float>();
    }

    try {
        LOG_INFO("Synthesizing: ", text);

        // 1. 文本预处理
        std::string processed_text = preprocessText(text);

        // 2. 文本转 token
        std::vector<int64_t> tokens = textToTokens(processed_text);
        if (tokens.empty()) {
            LOG_ERROR("Text tokenization failed");
            return std::vector<float>();
        }

        // 3. LLM 推理
        std::vector<float> llm_output = runLLM(tokens);
        if (llm_output.empty()) {
            LOG_ERROR("LLM inference failed");
            return std::vector<float>();
        }

        // 4. Flow 推理
        std::vector<float> mel_spec = runFlow(llm_output);
        if (mel_spec.empty()) {
            LOG_ERROR("Flow inference failed");
            return std::vector<float>();
        }

        // 5. Vocoder 推理
        std::vector<float> waveform = runVocoder(mel_spec);
        if (waveform.empty()) {
            LOG_ERROR("Vocoder inference failed");
            return std::vector<float>();
        }

        LOG_INFO("Synthesis complete, audio length: ", waveform.size(), " samples");
        return waveform;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("ONNX Runtime error: ", e.what());
        return std::vector<float>();
    } catch (const std::exception& e) {
        LOG_ERROR("Error during synthesis: ", e.what());
        return std::vector<float>();
    }
}

void TTSEngine::synthesizeStream(const std::string& text, StreamCallback callback) {
    if (!initialized_) {
        LOG_ERROR("TTS engine not initialized");
        return;
    }

    if (!config_.enable_streaming) {
        // 非流式模式：一次性合成
        auto audio = synthesize(text);
        if (!audio.empty()) {
            callback(audio.data(), audio.size());
        }
        return;
    }

    // 流式合成：分块处理
    // TODO: 实现真正的流式合成
    // 这里简化实现：先完整合成再分块返回

    auto audio = synthesize(text);
    if (audio.empty()) {
        return;
    }

    // 分块回调
    size_t pos = 0;
    while (pos < audio.size()) {
        size_t chunk_size = std::min(
            static_cast<size_t>(config_.streaming_chunk_size),
            audio.size() - pos
        );

        callback(&audio[pos], chunk_size);
        pos += chunk_size;
    }
}

std::string TTSEngine::preprocessText(const std::string& text) {
    if (!text_frontend_) {
        return text;
    }
    return text_frontend_->normalize(text);
}

std::vector<int64_t> TTSEngine::textToTokens(const std::string& text) {
    if (!text_frontend_) {
        // 简单的字符级分词
        std::vector<int64_t> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int64_t>(c));
        }
        return tokens;
    }

    return text_frontend_->textToTokens(text);
}

std::vector<float> TTSEngine::runLLM(const std::vector<int64_t>& tokens) {
    // 准备输入张量
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(tokens.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        const_cast<int64_t*>(tokens.data()), tokens.size(),
        input_shape.data(), input_shape.size());

    std::vector<int64_t> length_shape = {1};
    std::vector<int64_t> length_data = {static_cast<int64_t>(tokens.size())};

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info_, length_data.data(), length_data.size(),
        length_shape.data(), length_shape.size());

    // 运行推理
    const char* input_names[] = {"text_tokens", "text_lengths"};
    const char* output_names[] = {"llm_output"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(length_tensor));

    auto output_tensors = llm_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 2,
        output_names, 1);

    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> TTSEngine::runFlow(const std::vector<float>& llm_output) {
    // 简化实现：假设 flow 模型输入就是 LLM 输出
    // 实际可能需要更复杂的处理

    // 假设 llm_output 形状为 [1, seq_len, hidden_dim]
    std::vector<int64_t> input_shape = {
        1,
        static_cast<int64_t>(llm_output.size() / 512),  // 假设 hidden_dim = 512
        512
    };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(llm_output.data()), llm_output.size(),
        input_shape.data(), input_shape.size());

    // 运行推理
    const char* input_names[] = {"llm_output"};
    const char* output_names[] = {"mel_spectrogram"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    auto output_tensors = flow_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 1,
        output_names, 1);

    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    return std::vector<float>(output_data, output_data + output_size);
}

std::vector<float> TTSEngine::runVocoder(const std::vector<float>& mel_spec) {
    // HiFT vocoder 现在输出幅度和相位，而不是直接输出音频
    // mel_spec 形状为 [1, n_mels, time_steps]
    std::vector<int64_t> input_shape = {
        1,
        80,  // n_mels
        static_cast<int64_t>(mel_spec.size() / 80)
    };

    int64_t num_mel_frames = input_shape[2];

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(mel_spec.data()), mel_spec.size(),
        input_shape.data(), input_shape.size());

    // 运行推理 - HiFT 输出 magnitude 和 phase
    const char* input_names[] = {"mel"};
    const char* output_names[] = {"magnitude", "phase"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    auto output_tensors = vocoder_session_->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 1,
        output_names, 2);  // 两个输出

    // 获取幅度和相位
    float* magnitude_data = output_tensors[0].GetTensorMutableData<float>();
    float* phase_data = output_tensors[1].GetTensorMutableData<float>();

    auto magnitude_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto phase_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    // 验证输出形状: [batch, n_fft/2+1, num_frames]
    // HiFT 使用 n_fft=16, 所以 n_fft/2+1 = 9
    if (magnitude_shape.size() != 3 || magnitude_shape[1] != 9) {
        LOG_ERROR("Unexpected magnitude shape from vocoder");
        return std::vector<float>();
    }

    int64_t batch_size = magnitude_shape[0];
    int64_t num_bins = magnitude_shape[1];  // Should be 9 (n_fft/2+1)
    int64_t num_spec_frames = magnitude_shape[2];

    LOG_DEBUG("Vocoder output: magnitude shape [", batch_size, ", ", num_bins, ", ", num_spec_frames, "]");

    // 计算幅度和相位数据大小
    size_t spec_size = num_bins * num_spec_frames;

    // 转换为 vector（假设 batch_size=1）
    std::vector<float> magnitude(magnitude_data, magnitude_data + spec_size);
    std::vector<float> phase(phase_data, phase_data + spec_size);

    // 使用 iSTFT 重建音频
    LOG_DEBUG("Performing iSTFT reconstruction...");
    std::vector<float> waveform = audio_transform_->istft(magnitude, phase, num_spec_frames);

    LOG_DEBUG("iSTFT complete, waveform length: ", waveform.size(), " samples");

    return waveform;
}

} // namespace tts
} // namespace voice_assistant
