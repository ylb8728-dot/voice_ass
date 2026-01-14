#include "audio_input.h"
#include "../utils/logger.h"
#include <cstring>
#include <stdexcept>

namespace voice_assistant {
namespace audio {

AudioInput::AudioInput(int sample_rate, int channels, unsigned long frames_per_buffer)
    : stream_(nullptr)
    , sample_rate_(sample_rate)
    , channels_(channels)
    , frames_per_buffer_(frames_per_buffer)
    , is_active_(false) {
}

AudioInput::~AudioInput() {
    stop();

    if (stream_) {
        Pa_CloseStream(stream_);
        stream_ = nullptr;
    }

    Pa_Terminate();
}

bool AudioInput::initialize(const std::string& device_name) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        LOG_ERROR("PortAudio initialization failed: ", Pa_GetErrorText(err));
        return false;
    }

    int device_index = findDeviceIndex(device_name);
    if (device_index < 0) {
        LOG_ERROR("Audio input device not found: ", device_name.empty() ? "default" : device_name);
        return false;
    }

    PaStreamParameters input_params;
    std::memset(&input_params, 0, sizeof(input_params));
    input_params.device = device_index;
    input_params.channelCount = channels_;
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency =
        Pa_GetDeviceInfo(device_index)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(
        &stream_,
        &input_params,
        nullptr,  // no output
        sample_rate_,
        frames_per_buffer_,
        paClipOff,
        &AudioInput::paCallback,
        this
    );

    if (err != paNoError) {
        LOG_ERROR("Failed to open audio stream: ", Pa_GetErrorText(err));
        return false;
    }

    LOG_INFO("Audio input initialized: ", sample_rate_, "Hz, ",
             channels_, " channel(s), device: ", device_name.empty() ? "default" : device_name);
    return true;
}

bool AudioInput::start(AudioCallback callback) {
    if (!stream_) {
        LOG_ERROR("Audio input not initialized");
        return false;
    }

    if (is_active_) {
        LOG_WARN("Audio input already active");
        return true;
    }

    callback_ = callback;

    PaError err = Pa_StartStream(stream_);
    if (err != paNoError) {
        LOG_ERROR("Failed to start audio stream: ", Pa_GetErrorText(err));
        return false;
    }

    is_active_ = true;
    LOG_INFO("Audio input started");
    return true;
}

void AudioInput::stop() {
    if (!is_active_) {
        return;
    }

    if (stream_) {
        Pa_StopStream(stream_);
    }

    is_active_ = false;
    LOG_INFO("Audio input stopped");
}

std::vector<std::string> AudioInput::listDevices() {
    std::vector<std::string> devices;

    Pa_Initialize();

    int num_devices = Pa_GetDeviceCount();
    for (int i = 0; i < num_devices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info->maxInputChannels > 0) {
            devices.push_back(info->name);
        }
    }

    Pa_Terminate();

    return devices;
}

std::string AudioInput::getDefaultDevice() {
    Pa_Initialize();

    PaDeviceIndex device = Pa_GetDefaultInputDevice();
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

int AudioInput::paCallback(
    const void* input_buffer,
    void* output_buffer,
    unsigned long frames_per_buffer,
    const PaStreamCallbackTimeInfo* time_info,
    PaStreamCallbackFlags status_flags,
    void* user_data) {

    (void)output_buffer;
    (void)time_info;
    (void)status_flags;

    AudioInput* self = static_cast<AudioInput*>(user_data);

    if (self->callback_ && input_buffer) {
        const float* in = static_cast<const float*>(input_buffer);
        self->callback_(in, frames_per_buffer);
    }

    return paContinue;
}

int AudioInput::findDeviceIndex(const std::string& device_name) {
    if (device_name.empty()) {
        return Pa_GetDefaultInputDevice();
    }

    int num_devices = Pa_GetDeviceCount();
    for (int i = 0; i < num_devices; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info->maxInputChannels > 0 && device_name == info->name) {
            return i;
        }
    }

    return -1;
}

} // namespace audio
} // namespace voice_assistant
