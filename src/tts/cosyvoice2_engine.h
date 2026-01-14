#ifndef VOICE_ASSISTANT_COSYVOICE2_ENGINE_H
#define VOICE_ASSISTANT_COSYVOICE2_ENGINE_H

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>

namespace voice_assistant {
namespace tts {

/**
 * @brief CosyVoice2 TTS引擎 - 支持Zero-Shot和Streaming
 *
 * 基于CosyVoice2-0.5B模型，实现完整的TTS pipeline:
 * 1. Frontend: 文本处理和特征提取
 * 2. LLM: 生成speech token序列
 * 3. Flow: speech token转mel频谱
 * 4. HiFT: mel频谱转波形
 */
class CosyVoice2Engine {
public:
    using StreamCallback = std::function<void(const float* audio, size_t num_samples)>;

    struct Config {
        std::string model_dir;              // 模型目录，包含所有ONNX模型

        // 模型文件路径（相对于model_dir）
        std::string tokenizer_path = "tokenizer.json";
        std::string campplus_model = "campplus.onnx";
        std::string speech_tokenizer_model = "speech_tokenizer_v2.onnx";
        std::string llm_model = "llm.onnx";
        std::string flow_encoder_model = "flow_encoder.onnx";
        std::string flow_decoder_model = "flow_decoder.onnx";
        std::string hift_model = "hift.onnx";

        // 运行时配置
        int sample_rate = 24000;            // CosyVoice2使用24kHz
        int num_threads = 4;
        bool use_gpu = true;
        int gpu_device_id = 0;

        // Streaming配置
        bool enable_streaming = true;
        int chunk_size = 25;                // token chunk size
        int token_frame_rate = 25;          // token帧率
        int token_mel_ratio = 2;            // token到mel的比例

        // 生成参数
        float speed = 1.0f;
        float temperature = 0.8f;
        int top_k = 25;
        float top_p = 0.8f;
    };

    explicit CosyVoice2Engine(const Config& config);
    ~CosyVoice2Engine();

    // 禁止拷贝
    CosyVoice2Engine(const CosyVoice2Engine&) = delete;
    CosyVoice2Engine& operator=(const CosyVoice2Engine&) = delete;

    /**
     * @brief 初始化引擎
     * @return 成功返回true
     */
    bool initialize();

    /**
     * @brief Zero-Shot语音合成（离线模式）
     * @param tts_text 待合成文本
     * @param prompt_text Prompt文本（用于zero-shot）
     * @param prompt_audio_16k Prompt音频（16kHz, float32）
     * @param prompt_audio_samples Prompt音频样本数
     * @return 合成的音频（24kHz, float32）
     */
    std::vector<float> synthesizeZeroShot(
        const std::string& tts_text,
        const std::string& prompt_text,
        const float* prompt_audio_16k,
        size_t prompt_audio_samples);

    /**
     * @brief Zero-Shot语音合成（流式模式）
     * @param tts_text 待合成文本
     * @param prompt_text Prompt文本
     * @param prompt_audio_16k Prompt音频（16kHz, float32）
     * @param prompt_audio_samples Prompt音频样本数
     * @param callback 音频流回调
     */
    void synthesizeZeroShotStream(
        const std::string& tts_text,
        const std::string& prompt_text,
        const float* prompt_audio_16k,
        size_t prompt_audio_samples,
        StreamCallback callback);

    /**
     * @brief 检查是否初始化
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief Frontend: 提取文本token
     */
    std::vector<int64_t> extractTextToken(const std::string& text);

    /**
     * @brief Frontend: 提取说话人embedding
     */
    std::vector<float> extractSpeakerEmbedding(const float* audio_16k, size_t num_samples);

    /**
     * @brief Frontend: 提取Speech token
     */
    std::vector<int64_t> extractSpeechToken(const float* audio_16k, size_t num_samples);

    /**
     * @brief Frontend: 提取Mel频谱特征
     */
    std::vector<float> extractMelSpectrogram(const float* audio_24k, size_t num_samples);

    /**
     * @brief 音频重采样：16kHz -> 24kHz
     */
    std::vector<float> resample16kTo24k(const float* audio_16k, size_t num_samples);

    /**
     * @brief LLM推理：生成speech token序列
     */
    std::vector<int64_t> llmInference(
        const std::vector<int64_t>& text_token,
        const std::vector<int64_t>& prompt_text_token,
        const std::vector<int64_t>& prompt_speech_token,
        const std::vector<float>& embedding);

    /**
     * @brief Flow推理：token转mel频谱
     */
    std::vector<float> flowInference(
        const std::vector<int64_t>& speech_token,
        const std::vector<int64_t>& prompt_token,
        const std::vector<float>& prompt_feat,
        const std::vector<float>& embedding);

    /**
     * @brief HiFT推理：mel频谱转波形
     */
    std::vector<float> hiftInference(const std::vector<float>& mel_spectrogram);

    /**
     * @brief Streaming worker线程
     */
    void streamingWorker(
        const std::vector<int64_t>& text_token,
        const std::vector<int64_t>& prompt_text_token,
        const std::vector<int64_t>& prompt_speech_token,
        const std::vector<int64_t>& flow_prompt_token,
        const std::vector<float>& prompt_feat,
        const std::vector<float>& llm_embedding,
        const std::vector<float>& flow_embedding,
        StreamCallback callback);

    Config config_;
    bool initialized_;

    // ONNX Runtime sessions
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;

    std::unique_ptr<Ort::Session> campplus_session_;
    std::unique_ptr<Ort::Session> speech_tokenizer_session_;
    std::unique_ptr<Ort::Session> llm_session_;
    std::unique_ptr<Ort::Session> flow_encoder_session_;
    std::unique_ptr<Ort::Session> flow_decoder_session_;
    std::unique_ptr<Ort::Session> hift_session_;

    Ort::MemoryInfo memory_info_;

    // Tokenizer（需要实现或使用第三方库）
    class Tokenizer;
    std::unique_ptr<Tokenizer> tokenizer_;

    // Streaming相关
    std::mutex mutex_;
    std::queue<std::vector<float>> audio_queue_;
};

} // namespace tts
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_COSYVOICE2_ENGINE_H
