#ifndef VOICE_ASSISTANT_AUDIO_OUTPUT_H
#define VOICE_ASSISTANT_AUDIO_OUTPUT_H

#include <portaudio.h>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace voice_assistant {
namespace audio {

/**
 * @brief 音频输出类，基于 PortAudio
 *
 * 负责将音频数据通过扬声器播放
 */
class AudioOutput {
public:
    /**
     * @brief 构造函数
     * @param sample_rate 采样率 (默认 22050 Hz)
     * @param channels 声道数 (默认 1 单声道)
     * @param frames_per_buffer 每次回调的帧数 (默认 512)
     */
    AudioOutput(int sample_rate = 22050,
                int channels = 1,
                unsigned long frames_per_buffer = 512);

    ~AudioOutput();

    // 禁止拷贝
    AudioOutput(const AudioOutput&) = delete;
    AudioOutput& operator=(const AudioOutput&) = delete;

    /**
     * @brief 初始化音频输出
     * @param device_name 设备名称，空字符串使用默认设备
     * @return 成功返回 true
     */
    bool initialize(const std::string& device_name = std::string());

    /**
     * @brief 同步播放音频
     * @param audio_data 音频数据
     */
    void play(const std::vector<float>& audio_data);

    /**
     * @brief 异步播放音频（加入队列）
     * @param audio_data 音频数据
     */
    void playAsync(const std::vector<float>& audio_data);

    /**
     * @brief 停止播放并清空队列
     */
    void stop();

    /**
     * @brief 等待所有队列中的音频播放完成
     */
    void waitFinish();

    /**
     * @brief 检查是否正在播放
     */
    bool isPlaying() const { return is_playing_; }

    /**
     * @brief 列出所有可用的输出设备
     */
    static std::vector<std::string> listDevices();

    /**
     * @brief 获取默认输出设备名称
     */
    static std::string getDefaultDevice();

private:
    /**
     * @brief PortAudio 回调函数
     */
    static int paCallback(const void* input_buffer,
                         void* output_buffer,
                         unsigned long frames_per_buffer,
                         const PaStreamCallbackTimeInfo* time_info,
                         PaStreamCallbackFlags status_flags,
                         void* user_data);

    /**
     * @brief 查找设备索引
     */
    int findDeviceIndex(const std::string& device_name);

    /**
     * @brief 异步播放线程
     */
    void playbackThread();

    PaStream* stream_;
    int sample_rate_;
    int channels_;
    unsigned long frames_per_buffer_;

    // 当前播放缓冲区
    std::vector<float> current_buffer_;
    size_t current_position_;
    std::mutex buffer_mutex_;

    // 异步播放队列
    std::queue<std::vector<float>> play_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // 播放状态
    std::atomic<bool> is_playing_;
    std::atomic<bool> stop_thread_;
    std::thread playback_thread_;
};

} // namespace audio
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_AUDIO_OUTPUT_H
