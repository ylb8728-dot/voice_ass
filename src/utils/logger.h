#ifndef VOICE_ASSISTANT_LOGGER_H
#define VOICE_ASSISTANT_LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>

namespace voice_assistant {
namespace utils {

enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& getInstance();

    void setLevel(LogLevel level);
    void setLogFile(const std::string& path);

    void log(LogLevel level, const std::string& message);

    template<typename... Args>
    void trace(Args&&... args) { log(LogLevel::TRACE, format(std::forward<Args>(args)...)); }

    template<typename... Args>
    void debug(Args&&... args) { log(LogLevel::DEBUG, format(std::forward<Args>(args)...)); }

    template<typename... Args>
    void info(Args&&... args) { log(LogLevel::INFO, format(std::forward<Args>(args)...)); }

    template<typename... Args>
    void warn(Args&&... args) { log(LogLevel::WARN, format(std::forward<Args>(args)...)); }

    template<typename... Args>
    void error(Args&&... args) { log(LogLevel::ERROR, format(std::forward<Args>(args)...)); }

    template<typename... Args>
    void critical(Args&&... args) { log(LogLevel::CRITICAL, format(std::forward<Args>(args)...)); }

private:
    Logger();
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template<typename T>
    std::string format(T&& arg) {
        std::ostringstream oss;
        oss << std::forward<T>(arg);
        return oss.str();
    }

    template<typename T, typename... Args>
    std::string format(T&& first, Args&&... args) {
        std::ostringstream oss;
        oss << std::forward<T>(first);
        ((oss << std::forward<Args>(args)), ...);
        return oss.str();
    }

    std::string levelToString(LogLevel level);
    std::string getCurrentTime();

    LogLevel min_level_;
    std::ofstream log_file_;
    std::mutex mutex_;
    bool console_output_;
};

// 全局日志宏
#define LOG_TRACE(...) voice_assistant::utils::Logger::getInstance().trace(__VA_ARGS__)
#define LOG_DEBUG(...) voice_assistant::utils::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) voice_assistant::utils::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARN(...) voice_assistant::utils::Logger::getInstance().warn(__VA_ARGS__)
#define LOG_ERROR(...) voice_assistant::utils::Logger::getInstance().error(__VA_ARGS__)
#define LOG_CRITICAL(...) voice_assistant::utils::Logger::getInstance().critical(__VA_ARGS__)

} // namespace utils
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_LOGGER_H
