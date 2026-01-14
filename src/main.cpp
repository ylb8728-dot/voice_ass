#include "audio/audio_input.h"
#include "audio/audio_output.h"
#include "audio/vad_detector.h"
#include "asr/asr_engine.h"
#include "query/query_engine.h"
#include "tts/tts_engine.h"
#include "utils/logger.h"
#include "utils/config.h"
#include "utils/thread_pool.h"

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>

using namespace voice_assistant;

// 全局退出标志
std::atomic<bool> g_running(true);

void signalHandler(int signal) {
    (void)signal;
    LOG_INFO("Received signal, shutting down...");
    g_running = false;
}

/**
 * @brief 语音助手主类
 */
class VoiceAssistant {
public:
    VoiceAssistant()
        : thread_pool_(4) {
    }

    ~VoiceAssistant() {
        stop();
    }

    bool initialize(const std::string& config_path) {
        LOG_INFO("Initializing Voice Assistant...");

        // 加载配置
        config_ = std::make_unique<utils::Config>();
        if (!config_->loadFromFile(config_path)) {
            LOG_ERROR("Failed to load config file");
            return false;
        }

        // 初始化日志
        std::string log_level_str = config_->getString("system.log_level", "info");
        utils::LogLevel log_level = utils::LogLevel::INFO;
        if (log_level_str == "trace") log_level = utils::LogLevel::TRACE;
        else if (log_level_str == "debug") log_level = utils::LogLevel::DEBUG;
        else if (log_level_str == "warn") log_level = utils::LogLevel::WARN;
        else if (log_level_str == "error") log_level = utils::LogLevel::ERROR;

        utils::Logger::getInstance().setLevel(log_level);

        std::string log_path = config_->getString("system.log_path", "");
        if (!log_path.empty()) {
            utils::Logger::getInstance().setLogFile(log_path);
        }

        // 初始化音频输入
        int input_sample_rate = config_->getInt("audio.input.sample_rate", 16000);
        int input_channels = config_->getInt("audio.input.channels", 1);
        int input_buffer_size = config_->getInt("audio.input.buffer_size", 256);

        audio_input_ = std::make_unique<audio::AudioInput>(
            input_sample_rate, input_channels, input_buffer_size);

        std::string input_device = config_->getString("audio.input.device", "");
        if (!audio_input_->initialize(input_device)) {
            LOG_ERROR("Failed to initialize audio input");
            return false;
        }

        // 初始化音频输出
        int output_sample_rate = config_->getInt("audio.output.sample_rate", 22050);
        int output_channels = config_->getInt("audio.output.channels", 1);
        int output_buffer_size = config_->getInt("audio.output.buffer_size", 512);

        audio_output_ = std::make_unique<audio::AudioOutput>(
            output_sample_rate, output_channels, output_buffer_size);

        std::string output_device = config_->getString("audio.output.device", "");
        if (!audio_output_->initialize(output_device)) {
            LOG_ERROR("Failed to initialize audio output");
            return false;
        }

        // 初始化 VAD
        audio::VADDetector::Config vad_config;
        vad_config.energy_threshold = config_->getDouble("asr.vad_threshold", 0.5f);
        vad_detector_ = std::make_unique<audio::VADDetector>(vad_config);

        // 初始化 ASR
        asr::ASREngine::Config asr_config;
        asr_config.model_path = config_->getString("asr.model_path");
        asr_config.language = config_->getString("asr.language", "中文");
        asr_config.num_threads = config_->getInt("asr.num_threads", 4);
        asr_config.use_gpu = config_->getBool("asr.use_gpu", true);
        asr_config.gpu_device_id = config_->getInt("asr.gpu_device_id", 0);

        asr_engine_ = std::make_unique<asr::ASREngine>(asr_config);
        if (!asr_engine_->initialize()) {
            LOG_ERROR("Failed to initialize ASR engine");
            return false;
        }

        // 初始化查询引擎
        query::QueryEngine::Config query_config;

        std::string query_mode = config_->getString("query.mode", "hybrid");
        if (query_mode == "local") {
            query_config.mode = query::QueryEngine::Mode::LOCAL_ONLY;
        } else if (query_mode == "llm") {
            query_config.mode = query::QueryEngine::Mode::LLM_ONLY;
        } else {
            query_config.mode = query::QueryEngine::Mode::HYBRID;
        }

        query_config.db_config.db_path = config_->getString("query.database.path");
        query_config.db_config.enable_fts = config_->getBool("query.database.enable_fts", true);

        query_config.llm_type = config_->getString("query.llm.type", "local");
        query_config.local_llm_config.model_path = config_->getString("query.llm.model_path");
        query_config.local_llm_config.temperature = config_->getDouble("query.llm.temperature", 0.7f);

        query_engine_ = std::make_unique<query::QueryEngine>(query_config);
        if (!query_engine_->initialize()) {
            LOG_ERROR("Failed to initialize query engine");
            return false;
        }

        // 初始化 TTS
        tts::TTSEngine::Config tts_config;
        tts_config.model_dir = config_->getString("tts.model_dir");
        tts_config.speaker = config_->getString("tts.speaker", "中文女");
        tts_config.enable_streaming = config_->getBool("tts.enable_streaming", true);
        tts_config.num_threads = config_->getInt("tts.num_threads", 4);
        tts_config.use_gpu = config_->getBool("tts.use_gpu", true);
        tts_config.gpu_device_id = config_->getInt("tts.gpu_device_id", 0);

        tts_engine_ = std::make_unique<tts::TTSEngine>(tts_config);
        if (!tts_engine_->initialize()) {
            LOG_ERROR("Failed to initialize TTS engine");
            return false;
        }

        LOG_INFO("Voice Assistant initialized successfully");
        return true;
    }

    void run() {
        LOG_INFO("Starting Voice Assistant...");
        LOG_INFO("Listening... (Press Ctrl+C to exit)");

        // 启动音频输入
        audio_input_->start([this](const float* audio, size_t length) {
            this->onAudioInput(audio, length);
        });

        // 主循环
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        LOG_INFO("Voice Assistant stopped");
    }

    void stop() {
        if (audio_input_) {
            audio_input_->stop();
        }

        if (audio_output_) {
            audio_output_->stop();
        }
    }

private:
    void onAudioInput(const float* audio, size_t length) {
        // VAD 检测
        bool is_speech = vad_detector_->isSpeech(audio, length);

        if (is_speech) {
            // 累积音频
            std::lock_guard<std::mutex> lock(audio_buffer_mutex_);
            audio_buffer_.insert(audio_buffer_.end(), audio, audio + length);
        } else {
            // 静音段，检查是否有累积的音频
            std::vector<float> buffer_copy;
            {
                std::lock_guard<std::mutex> lock(audio_buffer_mutex_);
                if (audio_buffer_.size() > 16000) {  // 至少 1 秒音频
                    buffer_copy = audio_buffer_;
                    audio_buffer_.clear();
                }
            }

            if (!buffer_copy.empty()) {
                // 异步处理
                thread_pool_.enqueue([this, buffer_copy]() {
                    this->processAudio(buffer_copy);
                });
            }
        }
    }

    void processAudio(const std::vector<float>& audio) {
        LOG_INFO("Processing audio segment (", audio.size(), " samples)");

        // 1. ASR 识别
        std::string text = asr_engine_->recognize(audio);
        if (text.empty()) {
            LOG_WARN("ASR returned empty text");
            return;
        }

        LOG_INFO("User said: ", text);

        // 2. 查询
        std::string response = query_engine_->query(text);
        if (response.empty()) {
            response = "抱歉，我没有找到相关信息。";
        }

        LOG_INFO("Response: ", response);

        // 3. TTS 合成
        tts_engine_->synthesizeStream(response, [this](const float* audio_chunk, size_t length) {
            // 播放音频
            std::vector<float> chunk(audio_chunk, audio_chunk + length);
            audio_output_->playAsync(chunk);
        });
    }

    std::unique_ptr<utils::Config> config_;
    std::unique_ptr<audio::AudioInput> audio_input_;
    std::unique_ptr<audio::AudioOutput> audio_output_;
    std::unique_ptr<audio::VADDetector> vad_detector_;
    std::unique_ptr<asr::ASREngine> asr_engine_;
    std::unique_ptr<query::QueryEngine> query_engine_;
    std::unique_ptr<tts::TTSEngine> tts_engine_;

    utils::ThreadPool thread_pool_;

    std::vector<float> audio_buffer_;
    std::mutex audio_buffer_mutex_;
};

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 注册信号处理
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 解析命令行参数
    std::string config_path = "config/config.yaml";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                     << "Options:\n"
                     << "  --config PATH    Config file path (default: config/config.yaml)\n"
                     << "  --list-devices   List audio devices\n"
                     << "  --help, -h       Show this help message\n";
            return 0;
        } else if (arg == "--list-devices") {
            std::cout << "Input devices:\n";
            auto input_devices = audio::AudioInput::listDevices();
            for (size_t i = 0; i < input_devices.size(); ++i) {
                std::cout << "  [" << i << "] " << input_devices[i] << "\n";
            }

            std::cout << "\nOutput devices:\n";
            auto output_devices = audio::AudioOutput::listDevices();
            for (size_t i = 0; i < output_devices.size(); ++i) {
                std::cout << "  [" << i << "] " << output_devices[i] << "\n";
            }
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    // 创建并运行语音助手
    try {
        VoiceAssistant assistant;

        if (!assistant.initialize(config_path)) {
            LOG_ERROR("Failed to initialize voice assistant");
            return 1;
        }

        assistant.run();

    } catch (const std::exception& e) {
        LOG_CRITICAL("Fatal error: ", e.what());
        return 1;
    }

    LOG_INFO("Goodbye!");
    return 0;
}
