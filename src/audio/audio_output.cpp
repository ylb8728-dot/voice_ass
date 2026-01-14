#include "audio_output.h"
#include "../utils/logger.h"
#include <cstring>
#include <algorithm>

namespace voice_assistant {
namespace audio {

AudioOutput::AudioOutput(int sample_rate, int channels, unsigned long frames_per_buffer)
    : stream_(nullptr)
    , sample_rate_(sample_rate)
    , channels_(channels)
    , frames_per_buffer_(frames_per_buffer)
    , current_position_(0)
    , is_playing_(false)
    , stop_thread_(false) {
}

AudioOutput::~AudioOutput() {
    stop();

    stop_thread_ = true;
    queue_cv_.notify_all();

    if (playback_thread_.joinable()) {
        playback_thread_.join();
    }

    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }

    Pa_Terminate();
}

bool AudioOutput::initialize(const std::string& device_name) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        LOG_ERROR("PortAudio initialization failed: ", Pa_GetErrorText(err));
        return false;
    }

    int device_index = findDeviceIndex(device_name);
    if (device_index < 0) {
        LOG_ERROR("Audio output device not found: ", device_name.empty() ? "default" : device_name);
        return false;
    }

    PaStreamParameters output_params;
    std::memset(&output_params, 0, sizeof(output_params));
    output_params.device = device_index;
    output_params.channelCount = channels_;
    output_params.sampleFormat = paFloat32;
    output_params.suggestedLatency =
        Pa_GetDeviceInfo(device_index)->defaultLowOutputLatency;
    output_params.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(
        &stream_,
        nullptr,  // no input
        &output_params,
        sample_rate_,
        frames_per_buffer_,
        paClipOff,
        &AudioOutput::paCallback,
        this
    );

    if (err != paNoError) {
        LOG_ERROR("Failed to open audio stream: ", Pa_GetErrorText(err));
        return false;
    }

    // 启动异步播放线程
    playback_thread_ = std::thread(&AudioOutput::playbackThread, this);

    LOG_INFO("Audio output initialized: ", sample_rate_, "Hz, ",
             channels_, " channel(s), device: ", device_name.empty() ? "default" : device_name);
    return true;
}

void AudioOutput::play(const std::vector<float>& audio_data) {
    if (!stream_) {
        LOG_ERROR("Audio output not initialized");
        return;
    }

    if (audio_data.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        current_buffer_ = audio_data;
        current_position_ = 0;
    }

    if (!is_playing_) {
        PaError err = Pa_StartStream(stream_);
        if (err != paNoError) {
            LOG_ERROR("Failed to start audio stream: ", Pa_GetErrorText(err));
            return;
        }
        is_playing_ = true;
    }

    // 等待播放完成
    while (current_position_ < current_buffer_.size()) {
        Pa_Sleep(10);
    }

    if (is_playing_) {
        Pa_StopStream(stream_);
        is_playing_ = false;
    }
}

void AudioOutput::playAsync(const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        play_queue_.push(audio_data);
    }
    queue_cv_.notify_one();
}

void AudioOutput::stop() {
    if (stream_ && is_playing_) {
        Pa_StopStream(stream_);
        is_playing_ = false;
    }

    // 清空队列
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        std::queue<std::vector<float>> empty;
        std::swap(play_queue_, empty);
    }

    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        current_buffer_.clear();
        current_position_ = 0;
    }
}

void AudioOutput::waitFinish() {
    while (!play_queue_.empty() || current_position_ < current_buffer_.size()) {
        Pa_Sleep(10);
    }
}

std::vector<std::string> AudioOutput::listDevices() {
    std::vector<std::string> devices;

    Pa_Initialize();

    int num_devices = Pa_GetDeviceCount();
    for (int i = 0; i < num_devices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info->maxOutputChannels > 0) {
            devices.push_back(info->name);
        }
    }

    Pa_Terminate();

    return devices;
}

std::string AudioOutput::getDefaultDevice() {
    Pa_Initialize();

    PaDeviceIndex device = Pa_GetDefaultOutputDevice();
    std::string name = "default";

    if (device != paNoDevice) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(device);
        if (info) {
            name = info->name;
        }
    }

    Pa_Terminate();

    return name;
}

int AudioOutput::paCallback(
    const void* input_buffer,
    void* output_buffer,
    unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags,
    void* user_data) {

    (void)input_buffer;
    (void)time_info;
    (void)status_flags;

    AudioOutput* self = static_cast<AudioOutput*>(user_data);
    float* out = static_cast<float*>(output_buffer);

    std::lock_guard<std::mutex> lock(self->buffer_mutex_);

    size_t frames_to_copy = std::min(
        static_cast<size_t>(frames_per_buffer),
        self->current_buffer_.size() - self->current_position_
    );

    if (frames_to_copy > 0) {
        std::memcpy(out,
                   &self->current_buffer_[self->current_position_],
                   frames_to_copy * sizeof(float));
        self->current_position_ += frames_to_copy;
    }

    // 填充剩余部分为静音
    if (frames_to_copy < frames_per_buffer) {
        std::memset(&out[frames_to_copy], 0,
                   (frames_per_buffer - frames_to_copy) * sizeof(float));
    }

    return paContinue;
}

int AudioOutput::findDeviceIndex(const std::string& device_name) {
    if (device_name.empty()) {
        return Pa_GetDefaultOutputDevice();
    }

    int num_devices = Pa_GetDeviceCount();
    for (int i = 0; i < num_devices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info->maxOutputChannels > 0 && device_name == info->name) {
            return i;
        }
    }

    return -1;
}

void AudioOutput::playbackThread() {
    while (!stop_thread_) {
        std::vector<float> audio_data;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !play_queue_.empty() || stop_thread_;
            });

            if (stop_thread_) {
                break;
            }

            if (!play_queue_.empty()) {
                audio_data = std::move(play_queue_.front());
                play_queue_.pop();
            }
        }

        if (!audio_data.empty()) {
            play(audio_data);
        }
    }
}

} // namespace audio
} // namespace voice_assistant
