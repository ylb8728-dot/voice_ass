#!/bin/bash
# 测试CosyVoice2导出脚本

set -e  # 遇到错误立即退出

echo "================================"
echo "CosyVoice2 ONNX Export Test"
echo "================================"

# 检查CosyVoice模型是否存在
MODEL_DIR="CosyVoice/pretrained_models/CosyVoice2-0.5B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    echo "Please download CosyVoice2-0.5B model first"
    exit 1
fi

# 检查配置文件
CONFIG_FILE="$MODEL_DIR/cosyvoice2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Model directory found"
echo "✓ Config file found"
echo ""

# 创建输出目录
OUTPUT_DIR="models/cosyvoice2_onnx"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 运行导出
echo "Starting export..."
echo "================================"

python scripts/export_cosyvoice2_to_onnx.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --opset-version 18 \
    --skip-llm

echo ""
echo "================================"
echo "Export completed!"
echo "================================"

# 检查导出的文件
echo ""
echo "Exported files:"
ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "No ONNX files found"

# 验证ONNX文件
echo ""
echo "Verifying ONNX models..."
for file in "$OUTPUT_DIR"/*.onnx; do
    if [ -f "$file" ]; then
        echo "  Checking $(basename $file)..."
        python -c "
import onnx
import sys
try:
    model = onnx.load('$file')
    onnx.checker.check_model(model)
    print('    ✓ Valid')
except Exception as e:
    print(f'    ✗ Invalid: {e}')
    sys.exit(1)
"
    fi
done

echo ""
echo "================================"
echo "All tests passed!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Copy pre-built ONNX models:"
echo "   cp $MODEL_DIR/campplus.onnx $OUTPUT_DIR/"
echo "   cp $MODEL_DIR/speech_tokenizer_v2.onnx $OUTPUT_DIR/"
echo ""
echo "2. Export LLM (optional):"
echo "   python scripts/export_llm_to_onnx.py --model-dir $MODEL_DIR --output-dir models/llm_onnx --mode components"
echo ""
echo "3. Build C++ test program:"
echo "   cd src/tts && mkdir build && cd build && cmake .. && make"
echo ""
