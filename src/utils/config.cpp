#include "config.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace voice_assistant {
namespace utils {

// 简单的配置存储实现（不依赖 yaml-cpp）
class Config::Impl {
public:
    std::map<std::string, std::string> values_;

    bool parseYamlFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        std::vector<std::string> section_stack;

        while (std::getline(file, line)) {
            // 移除注释
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            // 去除首尾空白
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.empty()) continue;

            // 计算缩进级别
            size_t indent = 0;
            for (char c : line) {
                if (c == ' ') indent++;
                else break;
            }
            int level = indent / 2;

            // 解析键值对
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);

                std::string value = line.substr(colon_pos + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                // 更新层级栈
                while (section_stack.size() > static_cast<size_t>(level)) {
                    section_stack.pop_back();
                }

                if (!value.empty()) {
                    // 移除引号
                    if ((value.front() == '"' && value.back() == '"') ||
                        (value.front() == '\'' && value.back() == '\'')) {
                        value = value.substr(1, value.length() - 2);
                    }

                    // 构建完整键名
                    std::string full_key;
                    for (const auto& section : section_stack) {
                        full_key += section + ".";
                    }
                    full_key += key;

                    values_[full_key] = value;
                } else {
                    // 新的 section
                    section_stack.push_back(key);
                }
            }
        }

        return true;
    }

    std::string get(const std::string& key, const std::string& default_value) const {
        auto it = values_.find(key);
        if (it != values_.end()) {
            return it->second;
        }
        return default_value;
    }
};

Config::Config() : impl_(std::make_unique<Impl>()) {}

Config::~Config() = default;

bool Config::loadFromFile(const std::string& path) {
    LOG_INFO("Loading config from: ", path);

    if (!impl_->parseYamlFile(path)) {
        LOG_ERROR("Failed to load config file: ", path);
        return false;
    }

    LOG_INFO("Config loaded successfully, ", impl_->values_.size(), " entries");
    return true;
}

std::string Config::getString(const std::string& key, const std::string& default_value) const {
    return impl_->get(key, default_value);
}

int Config::getInt(const std::string& key, int default_value) const {
    std::string value = impl_->get(key, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

double Config::getDouble(const std::string& key, double default_value) const {
    std::string value = impl_->get(key, "");
    if (value.empty()) {
        return default_value;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return default_value;
    }
}

bool Config::getBool(const std::string& key, bool default_value) const {
    std::string value = impl_->get(key, "");
    if (value.empty()) {
        return default_value;
    }
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value == "true" || value == "yes" || value == "1";
}

std::vector<std::string> Config::getStringList(const std::string& key) const {
    // 简单实现：假设列表用逗号分隔
    std::string value = impl_->get(key, "");
    std::vector<std::string> result;

    if (value.empty()) {
        return result;
    }

    std::istringstream iss(value);
    std::string item;
    while (std::getline(iss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            result.push_back(item);
        }
    }

    return result;
}

bool Config::has(const std::string& key) const {
    return impl_->values_.find(key) != impl_->values_.end();
}

void Config::set(const std::string& key, const std::string& value) {
    impl_->values_[key] = value;
}

} // namespace utils
} // namespace voice_assistant
