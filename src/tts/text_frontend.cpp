#include "text_frontend.h"
#include "../utils/logger.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>

namespace voice_assistant {
namespace tts {

TextFrontend::TextFrontend()
    : unk_id_(0) {
}

TextFrontend::~TextFrontend() = default;

std::string TextFrontend::normalize(const std::string& text) {
    std::string result = text;

    // 转换为小写（仅英文）
    // 注意：中文不需要转换

    // 数字转中文
    std::regex number_regex("\\d+");
    std::smatch match;
    std::string temp = result;
    result.clear();

    while (std::regex_search(temp, match, number_regex)) {
        result += match.prefix();
        result += numberToChinese(match.str());
        temp = match.suffix();
    }
    result += temp;

    // 移除特殊符号（保留基本标点）
    std::string cleaned;
    for (char c : result) {
        if (std::isalnum(static_cast<unsigned char>(c)) ||
            std::isspace(static_cast<unsigned char>(c)) ||
            c == ',' || c == '.' || c == '!' || c == '?' ||
            (c & 0x80)) {  // UTF-8 多字节字符（包括中文标点）
            cleaned += c;
        }
    }

    return cleaned;
}

std::vector<int64_t> TextFrontend::textToTokens(const std::string& text) {
    std::vector<int64_t> tokens;

    // 简单的字符级分词（中文）
    std::string normalized = normalize(text);

    size_t i = 0;
    while (i < normalized.size()) {
        // 处理 UTF-8 字符
        int char_len = 1;
        unsigned char c = normalized[i];

        if ((c & 0x80) == 0) {
            // ASCII
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }

        std::string word = normalized.substr(i, char_len);
        tokens.push_back(wordToId(word));

        i += char_len;
    }

    return tokens;
}

bool TextFrontend::loadVocabulary(const std::string& vocab_path) {
    LOG_INFO("Loading vocabulary from: ", vocab_path);

    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open vocabulary file: ", vocab_path);
        return false;
    }

    vocab_.clear();
    std::string line;
    int64_t id = 0;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocab_[line] = id++;
        }
    }

    unk_id_ = vocab_.size();

    LOG_INFO("Loaded ", vocab_.size(), " words");
    return true;
}

std::string TextFrontend::numberToChinese(const std::string& number) {
    // 简化实现：仅支持整数
    static const char* digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
    static const char* units[] = {"", "十", "百", "千"};

    if (number.empty()) {
        return "";
    }

    // 简单处理：逐位转换
    std::string result;
    for (size_t i = 0; i < number.size(); ++i) {
        int digit = number[i] - '0';
        if (digit >= 0 && digit <= 9) {
            result += digits[digit];

            // 添加单位
            int unit_index = number.size() - 1 - i;
            if (unit_index > 0 && unit_index < 4 && digit != 0) {
                result += units[unit_index];
            }
        }
    }

    return result;
}

std::vector<std::string> TextFrontend::tokenize(const std::string& text) {
    std::vector<std::string> tokens;

    // 简单的空格分词
    std::istringstream iss(text);
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

int64_t TextFrontend::wordToId(const std::string& word) {
    auto it = vocab_.find(word);
    if (it != vocab_.end()) {
        return it->second;
    }
    return unk_id_;
}

} // namespace tts
} // namespace voice_assistant
