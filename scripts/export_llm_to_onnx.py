#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice2 LLM导出脚本 - 处理自回归生成逻辑

策略：将LLM拆分为两个ONNX模型
1. Encoder (Prefill): 处理初始输入序列，生成KV cache
2. Decoder (Decode): 单步解码，使用KV cache增量生成

这种设计允许在C++中实现高效的自回归生成循环。
"""

import argparse
import os
import sys
import logging
import torch
import onnx
import onnxruntime as ort
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

# Add CosyVoice to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..', 'CosyVoice'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'CosyVoice', 'third_party', 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2


class LLMEncoderWrapper(torch.nn.Module):
    """
    LLM Encoder包装器 - Prefill阶段

    输入：
        - lm_input: [batch, seq_len, hidden_dim] - 初始输入序列embedding
        - attention_mask: [batch, seq_len] - 注意力mask

    输出：
        - hidden_states: [batch, seq_len, hidden_dim] - 最后一层hidden states
        - past_key_values: List of (key, value) pairs for each layer
    """
    def __init__(self, llm_model):
        super().__init__()
        self.model = llm_model

    def forward(self, lm_input, attention_mask):
        """Prefill阶段：处理完整输入序列"""
        outputs = self.model(
            inputs_embeds=lm_input,
            attention_mask=attention_mask,
            use_cache=True,  # 生成KV cache
            return_dict=True
        )

        # 返回hidden states和past_key_values
        return outputs.last_hidden_state, outputs.past_key_values


class LLMDecoderWrapper(torch.nn.Module):
    """
    LLM Decoder包装器 - Decode阶段

    输入：
        - input_embeds: [batch, 1, hidden_dim] - 新token的embedding
        - attention_mask: [batch, past_len + 1] - 注意力mask
        - past_key_values: KV cache from previous step

    输出：
        - hidden_states: [batch, 1, hidden_dim] - 新token的hidden state
        - present_key_values: Updated KV cache
    """
    def __init__(self, llm_model):
        super().__init__()
        self.model = llm_model

    def forward(self, input_embeds, attention_mask, past_key_values):
        """Decode阶段：单步增量解码"""
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )

        return outputs.last_hidden_state, outputs.past_key_values


class SimplifiedLLMWrapper(torch.nn.Module):
    """
    简化的LLM包装器 - 不使用KV cache（适用于较短序列）

    输入：
        - lm_input: [batch, seq_len, hidden_dim]
        - attention_mask: [batch, seq_len]

    输出：
        - logits: [batch, seq_len, vocab_size]
    """
    def __init__(self, llm, llm_decoder, speech_token_size):
        super().__init__()
        self.llm = llm
        self.llm_decoder = llm_decoder
        self.speech_token_size = speech_token_size

    def forward(self, lm_input, attention_mask):
        """前向传播：处理完整序列并输出logits"""
        outputs = self.llm(
            inputs_embeds=lm_input,
            attention_mask=attention_mask,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        logits = self.llm_decoder(hidden_states)

        return logits


def export_llm_simplified(model, output_path, opset_version=18, max_seq_len=512):
    """
    导出简化版LLM（不使用KV cache）

    适用于：
    - 快速原型开发
    - 序列长度较短的场景
    - 不需要流式生成的场景
    """
    logging.info("Exporting simplified LLM model (no KV cache)...")

    llm_module = model.model.llm

    # 获取Qwen2模型
    if hasattr(llm_module, 'llm') and hasattr(llm_module.llm, 'model'):
        qwen_model = llm_module.llm.model
        llm_decoder = llm_module.llm_decoder
        speech_token_size = llm_module.speech_token_size
    else:
        logging.error("Cannot find Qwen2 model in LLM structure")
        return False

    # 创建包装器
    wrapper = SimplifiedLLMWrapper(qwen_model, llm_decoder, speech_token_size)
    wrapper.eval()

    device = model.model.device
    batch_size = 1
    seq_len = 100
    hidden_dim = 896  # llm_input_size from config

    # 创建dummy输入
    dummy_lm_input = torch.randn(batch_size, seq_len, hidden_dim,
                                 dtype=torch.float32, device=device)
    dummy_attention_mask = torch.ones(batch_size, seq_len,
                                      dtype=torch.int64, device=device)

    try:
        logging.info(f"  Input shape: {dummy_lm_input.shape}")
        logging.info(f"  Attention mask shape: {dummy_attention_mask.shape}")

        torch.onnx.export(
            wrapper,
            (dummy_lm_input, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['lm_input', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'lm_input': {0: 'batch', 1: 'seq_len'},
                'attention_mask': {0: 'batch', 1: 'seq_len'},
                'logits': {0: 'batch', 1: 'seq_len'},
            }
        )

        logging.info(f"✓ Simplified LLM exported to {output_path}")
        verify_onnx_model(output_path)
        return True

    except Exception as e:
        logging.error(f"Failed to export simplified LLM: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_llm_with_kv_cache(model, encoder_path, decoder_path, opset_version=18):
    """
    导出带KV cache的LLM（拆分为encoder和decoder）

    这是推荐的导出方式，支持高效的自回归生成。

    注意：由于Qwen2模型的复杂性，KV cache的导出比较困难。
    这里提供框架，实际实现可能需要：
    1. 使用transformers的特殊导出工具
    2. 使用ONNX Runtime Generate扩展
    3. 使用TensorRT-LLM
    """
    logging.info("Exporting LLM with KV cache (encoder + decoder)...")

    llm_module = model.model.llm

    if hasattr(llm_module, 'llm') and hasattr(llm_module.llm, 'model'):
        qwen_model = llm_module.llm.model
    else:
        logging.error("Cannot find Qwen2 model in LLM structure")
        return False

    logging.warning("KV cache export is complex and may require specialized tools")
    logging.info("Recommended approaches:")
    logging.info("  1. Use Optimum (Hugging Face) for ONNX export with past_key_values")
    logging.info("  2. Use TensorRT-LLM for optimized inference")
    logging.info("  3. Use vLLM backend (CosyVoice2 already supports this)")

    # TODO: 实现完整的KV cache导出
    # 需要处理past_key_values的复杂结构

    return False


def export_llm_components(model, output_dir, opset_version=18):
    """
    导出LLM的各个组件

    这些组件可以在C++中组合使用，实现完整的LLM推理。
    """
    logging.info("Exporting LLM components...")

    llm_module = model.model.llm
    device = model.model.device

    # 1. 导出text_embedding
    logging.info("Exporting text_embedding...")
    text_embedding = llm_module.text_embedding
    text_embedding.eval()

    dummy_text_token = torch.randint(0, 1000, (1, 50), dtype=torch.int32, device=device)

    try:
        torch.onnx.export(
            text_embedding,
            dummy_text_token.long(),
            os.path.join(output_dir, 'llm_text_embedding.onnx'),
            export_params=True,
            opset_version=opset_version,
            input_names=['text_token'],
            output_names=['text_embedding'],
            dynamic_axes={
                'text_token': {0: 'batch', 1: 'seq_len'},
                'text_embedding': {0: 'batch', 1: 'seq_len'},
            }
        )
        logging.info("  ✓ text_embedding exported")
    except Exception as e:
        logging.warning(f"  ✗ text_embedding export failed: {e}")

    # 2. 导出speech_embedding
    logging.info("Exporting speech_embedding...")
    speech_embedding = llm_module.speech_embedding
    speech_embedding.eval()

    dummy_speech_token = torch.randint(0, 6561, (1, 100), dtype=torch.int32, device=device)

    try:
        torch.onnx.export(
            speech_embedding,
            dummy_speech_token.long(),
            os.path.join(output_dir, 'llm_speech_embedding.onnx'),
            export_params=True,
            opset_version=opset_version,
            input_names=['speech_token'],
            output_names=['speech_embedding'],
            dynamic_axes={
                'speech_token': {0: 'batch', 1: 'seq_len'},
                'speech_embedding': {0: 'batch', 1: 'seq_len'},
            }
        )
        logging.info("  ✓ speech_embedding exported")
    except Exception as e:
        logging.warning(f"  ✗ speech_embedding export failed: {e}")

    # 3. 导出spk_embed_affine_layer
    logging.info("Exporting spk_embed_affine_layer...")
    spk_affine = llm_module.spk_embed_affine_layer
    spk_affine.eval()

    dummy_spk_emb = torch.randn(1, 192, dtype=torch.float32, device=device)

    try:
        torch.onnx.export(
            spk_affine,
            dummy_spk_emb,
            os.path.join(output_dir, 'llm_spk_affine.onnx'),
            export_params=True,
            opset_version=opset_version,
            input_names=['spk_embedding'],
            output_names=['spk_embedding_projected'],
        )
        logging.info("  ✓ spk_embed_affine_layer exported")
    except Exception as e:
        logging.warning(f"  ✗ spk_embed_affine_layer export failed: {e}")

    # 4. 导出llm_decoder (output projection)
    logging.info("Exporting llm_decoder...")
    llm_decoder = llm_module.llm_decoder
    llm_decoder.eval()

    dummy_hidden = torch.randn(1, 50, 896, dtype=torch.float32, device=device)

    try:
        torch.onnx.export(
            llm_decoder,
            dummy_hidden,
            os.path.join(output_dir, 'llm_decoder.onnx'),
            export_params=True,
            opset_version=opset_version,
            input_names=['hidden_states'],
            output_names=['logits'],
            dynamic_axes={
                'hidden_states': {0: 'batch', 1: 'seq_len'},
                'logits': {0: 'batch', 1: 'seq_len'},
            }
        )
        logging.info("  ✓ llm_decoder exported")
    except Exception as e:
        logging.warning(f"  ✗ llm_decoder export failed: {e}")

    # 5. 导出llm_embedding (SOS/EOS and Task ID)
    logging.info("Exporting llm_embedding...")
    llm_embedding = llm_module.llm_embedding
    llm_embedding.eval()

    # 保存权重为numpy
    llm_emb_weight = llm_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(output_dir, 'llm_embedding_weight.npy'), llm_emb_weight)
    logging.info(f"  ✓ llm_embedding weights saved (shape: {llm_emb_weight.shape})")

    logging.info("✓ LLM components exported")
    return True


def verify_onnx_model(onnx_path):
    """验证ONNX模型"""
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info(f"  ✓ ONNX model {os.path.basename(onnx_path)} is valid")
    except Exception as e:
        logging.error(f"  ✗ ONNX model validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export CosyVoice2 LLM to ONNX')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to CosyVoice2 model directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for ONNX models')
    parser.add_argument('--opset-version', type=int, default=18,
                       help='ONNX opset version')
    parser.add_argument('--mode', type=str, default='simplified',
                       choices=['simplified', 'kv_cache', 'components'],
                       help='Export mode: simplified (no KV cache), kv_cache (with KV cache), components (individual parts)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(levelname)s] %(message)s')

    logging.info("="*60)
    logging.info("CosyVoice2 LLM ONNX Export")
    logging.info("="*60)
    logging.info(f"Model directory: {args.model_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Export mode: {args.mode}")
    logging.info("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    logging.info("Loading CosyVoice2 model...")
    try:
        model = CosyVoice2(args.model_dir)
        logging.info("✓ Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return 1

    # 导出
    try:
        if args.mode == 'simplified':
            output_path = os.path.join(args.output_dir, 'llm_simplified.onnx')
            success = export_llm_simplified(model, output_path, args.opset_version)

        elif args.mode == 'kv_cache':
            encoder_path = os.path.join(args.output_dir, 'llm_encoder.onnx')
            decoder_path = os.path.join(args.output_dir, 'llm_decoder_step.onnx')
            success = export_llm_with_kv_cache(model, encoder_path, decoder_path, args.opset_version)

        elif args.mode == 'components':
            success = export_llm_components(model, args.output_dir, args.opset_version)

        if success:
            logging.info("="*60)
            logging.info("✓ Export completed successfully!")
            logging.info("="*60)
            return 0
        else:
            logging.error("Export failed")
            return 1

    except Exception as e:
        logging.error(f"Export error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
