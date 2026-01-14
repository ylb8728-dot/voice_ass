#ifndef VOICE_ASSISTANT_CONFIG_H
#define VOICE_ASSISTANT_CONFIG_H

#include <string>
#include <map>
#include <memory>
#include <vector>

namespace voice_assistant {
namespace utils {

/**
 * @brief 配置管理类
 *
 * 支持从 YAML 文件加载配置，提供类型安全的访问接口
 */
class Config {
public:
    Config();
    ~Config();

    /**
     * @brief 从 YAML 文件加载配置
     * @param path YAML 文件路径
     * @return 成功返回 true
     */
    bool loadFromFile(const std::string& path);

    /**
     * @brief 获取字符串配置
     * @param key 配置键（支持点号分隔的嵌套路径，如 "asr.model_path"）
     * @param default_value 默认值
     * @return 配置值
     */
    std::string getString(const std::string& key, const std::string& default_value = "") const;

    /**
     * @brief 获取整数配置
     */
    int getInt(const std::string& key, int default_value = 0) const;

    /**
     * @brief 获取浮点数配置
     */
    double getDouble(const std::string& key, double default_value = 0.0) const;

    /**
     * @brief 获取布尔配置
     */
    bool getBool(const std::string& key, bool default_value = false) const;

    /**
     * @brief 获取字符串数组配置
     */
    std::vector<std::string> getStringList(const std::string& key) const;

    /**
     * @brief 检查配置项是否存在
     */
    bool has(const std::string& key) const;

    /**
     * @brief 设置配置值
     */
    void set(const std::string& key, const std::string& value);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace utils
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_CONFIG_H
