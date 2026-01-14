#ifndef VOICE_ASSISTANT_TEXT_FRONTEND_H
#define VOICE_ASSISTANT_TEXT_FRONTEND_H

#include <string>
#include <vector>
#include <map>

namespace voice_assistant {
namespace tts {

/**
 * @brief 文本前端处理器
 *
 * 处理文本归一化、分词和音素转换
 */
class TextFrontend {
public:
    TextFrontend();
    ~TextFrontend();

    /**
     * @brief 文本归一化
     * @param text 原始文本
     * @return 归一化后的文本
     */
    std::string normalize(const std::string& text);

    /**
     * @brief 文本转 token
     * @param text 文本
     * @return token ID 序列
     */
    std::vector<int64_t> textToTokens(const std::string& text);

    /**
     * @brief 加载词典
     * @param vocab_path 词典文件路径
     * @return 成功返回 true
     */
    bool loadVocabulary(const std::string& vocab_path);

private:
    /**
     * @brief 数字转中文
     */
    std::string numberToChinese(const std::string& number);

    /**
     * @brief 简单分词
     */
    std::vector<std::string> tokenize(const std::string& text);

    /**
     * @brief 词转 ID
     */
    int64_t wordToId(const std::string& word);

    std::map<std::string, int64_t> vocab_;
    int64_t unk_id_;
};

} // namespace tts
} // namespace voice_assistant

#endif // VOICE_ASSISTANT_TEXT_FRONTEND_H
