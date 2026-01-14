#include "asr/asr_engine.h"
#include "utils/logger.h"
#include <algorithm>
#include <stdexcept>

namespace voice_assistant {
namespace asr {

ASREngine::ASREngine(const Config& config)
    : config_(config),
      initialized_(false),
      ctx_(nullptr),
      state_(nullptr) {
}

ASREngine::~ASREngine() {
    if (state_) {
        whisper_free_state(state_);
        state_ = nullptr;
    }
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

bool ASREngine::initialize() {
    try {
        LOG_INFO("Initializing Whisper ASR engine...");

        // 设置上下文参数
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = config_.use_gpu;
        cparams.gpu_device = config_.gpu_device_id;
        cparams.flash_attn = true;  // 启用 flash attention 优化

        // 加载模型
        LOG_INFO("Loading Whisper model: ", config_.model_path);
        ctx_ = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
        if (!ctx_) {
            LOG_ERROR("Failed to load Whisper model: ", config_.model_path);
            return false;
        }

        // 创建推理状态（线程安全）
        state_ = whisper_init_state(ctx_);
        if (!state_) {
            LOG_ERROR("Failed to initialize Whisper state");
            whisper_free(ctx_);
            ctx_ = nullptr;
            return false;
        }

        initialized_ = true;
        LOG_INFO("Whisper ASR engine initialized successfully");
        LOG_INFO("Model: ", config_.model_path);
        LOG_INFO("Language: ", config_.language);
        LOG_INFO("Threads: ", config_.num_threads);
        LOG_INFO("GPU: ", config_.use_gpu ? "enabled" : "disabled");
        if (config_.use_gpu) {
            LOG_INFO("GPU device: ", config_.gpu_device_id);
        }

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("ASR initialization failed: ", e.what());
        return false;
    }
}

std::string ASREngine::recognize(const float* audio, size_t num_samples) {
    if (!initialized_) {
        LOG_ERROR("ASR engine not initialized");
        return "";
    }

    if (!audio || num_samples == 0) {
        LOG_WARN("Empty audio input");
        return "";
    }

    try {
        // 设置推理参数
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = config_.language.c_str();
        wparams.n_threads = config_.num_threads;
        wparams.translate = config_.translate;
        wparams.no_timestamps = config_.no_timestamps;
        wparams.temperature = config_.temperature;
        wparams.single_segment = false;
        wparams.print_progress = false;
        wparams.print_realtime = false;
        wparams.print_special = false;
        wparams.token_timestamps = false;

        // 执行推理
        LOG_DEBUG("Running Whisper inference on ", num_samples, " samples");
        if (whisper_full_with_state(ctx_, state_, wparams, audio, num_samples) != 0) {
            LOG_ERROR("Whisper inference failed");
            return "";
        }

        // 提取识别结果
        std::string result;
        const int n_segments = whisper_full_n_segments_from_state(state_);
        LOG_DEBUG("Got ", n_segments, " segments");

        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text_from_state(state_, i);
            if (text) {
                result += text;
            }
        }

        // 移除前后空格
        if (!result.empty()) {
            size_t start = result.find_first_not_of(" \t\n\r");
            size_t end = result.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                result = result.substr(start, end - start + 1);
            }
        }

        LOG_DEBUG("Recognized text: \"", result, "\"");
        return result;

    } catch (const std::exception& e) {
        LOG_ERROR("Recognition failed: ", e.what());
        return "";
    }
}

void ASREngine::startStream() {
    stream_buffer_.clear();
    partial_result_.clear();
    LOG_DEBUG("Started streaming ASR");
}

void ASREngine::feedAudio(const float* audio, size_t num_samples) {
    if (!audio || num_samples == 0) {
        return;
    }
    stream_buffer_.insert(stream_buffer_.end(), audio, audio + num_samples);
    LOG_TRACE("Fed ", num_samples, " samples, buffer size: ", stream_buffer_.size());
}

std::string ASREngine::getPartialResult() {
    // 至少需要 1 秒音频才能识别
    const size_t min_samples = 16000;  // 16kHz × 1s
    if (stream_buffer_.size() < min_samples) {
        LOG_TRACE("Not enough audio for partial result (", stream_buffer_.size(), " < ", min_samples, ")");
        return partial_result_;
    }

    // 使用滑动窗口：保留最近 10 秒音频作为上下文
    const size_t max_context_samples = 10 * 16000;  // 16kHz × 10s
    size_t context_samples = std::min(stream_buffer_.size(), max_context_samples);
    size_t offset = stream_buffer_.size() - context_samples;

    LOG_DEBUG("Processing partial result with ", context_samples, " samples");

    // 识别上下文窗口
    partial_result_ = recognize(&stream_buffer_[offset], context_samples);

    return partial_result_;
}

std::string ASREngine::endStream() {
    std::string final_result;

    // 识别所有缓冲音频
    if (!stream_buffer_.empty()) {
        LOG_DEBUG("Processing final stream result with ", stream_buffer_.size(), " samples");
        final_result = recognize(stream_buffer_.data(), stream_buffer_.size());
    }

    // 清空缓冲区
    stream_buffer_.clear();
    partial_result_.clear();

    LOG_DEBUG("Ended streaming ASR");
    return final_result;
}

} // namespace asr
} // namespace voice_assistant
