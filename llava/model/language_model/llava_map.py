"""
Extended LLaVA model for Map Detection

Key modifications:
1. Add scene queries from Q-Former (768 tokens)
2. Add learnable instance+point queries (1050 tokens)
3. Support custom attention mask
4. Extract features for detection

Author: Auto-generated for Map Detection
Date: 2025-01
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from ..map_queries import MapInstancePointQueries, MapAttentionMask, MapQueryExtractor


class LlavaMapDetectionModel(LlavaLlamaForCausalLM):
    """
    Extended LLaVA for map detection with custom queries and attention.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Add map detection components
        self.map_queries = MapInstancePointQueries(
            num_instances=50,
            num_points=20,
            embed_dim=config.hidden_size
        )
        
        self.use_custom_mask = True  # Enable custom structured attention mask
        
    def forward_with_map(
        self,
        text_embeds: torch.FloatTensor,
        scene_tokens: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_map_features: bool = False,
    ) -> Union[CausalLMOutputWithPast, dict]:
        """
        Forward pass with map detection.
        
        Args:
            text_embeds: (B, text_len, hidden_size) - 文本嵌入（可能已包含 scene tokens）
            scene_tokens: (B, 768, hidden_size) or None - Scene tokens（如果已嵌入 text_embeds 则为 None）
            labels: (B, text_len) for language modeling loss
            return_map_features: Whether to return map features
        
        Returns:
            If return_map_features=True:
                dict with:
                    - instance_features: (B, 50, hidden_size)
                    - point_features: (B, 50, 20, hidden_size)
                    - query_outputs: (B, 1050, hidden_size) - 保持 LLM 原始输出顺序
                    - scene_tokens: (B, 768, hidden_size)
            Else:
                CausalLMOutputWithPast
        """
        # ===== 输入验证 =====
        # 1. 检查 text_embeds 维度
        if text_embeds.dim() != 3:
            raise ValueError(
                f"text_embeds 应该是 3D tensor (B, L, H), "
                f"但收到 {text_embeds.dim()}D tensor: {text_embeds.shape}"
            )
        
        # 2. 检查 hidden_size 是否匹配 LLM
        expected_hidden_size = self.config.hidden_size
        if text_embeds.shape[2] != expected_hidden_size:
            raise ValueError(
                f"text_embeds hidden_size 不匹配: "
                f"期望 {expected_hidden_size}, 收到 {text_embeds.shape[2]}"
            )
        
        # 3. 检查 scene_tokens（如果提供）
        if scene_tokens is not None:
            if scene_tokens.dim() != 3:
                raise ValueError(
                    f"scene_tokens 应该是 3D tensor (B, N, H), "
                    f"但收到 {scene_tokens.dim()}D tensor: {scene_tokens.shape}"
                )
            if scene_tokens.shape[0] != text_embeds.shape[0]:
                raise ValueError(
                    f"Batch size 不匹配: "
                    f"text_embeds batch={text_embeds.shape[0]}, "
                    f"scene_tokens batch={scene_tokens.shape[0]}"
                )
            if scene_tokens.shape[2] != expected_hidden_size:
                raise ValueError(
                    f"scene_tokens hidden_size 不匹配: "
                    f"期望 {expected_hidden_size}, 收到 {scene_tokens.shape[2]}"
                )
        
        # 4. 检查 labels（如果提供）
        if labels is not None:
            if labels.dim() != 2:
                raise ValueError(
                    f"labels 应该是 2D tensor (B, L), "
                    f"但收到 {labels.dim()}D tensor: {labels.shape}"
                )
            if labels.shape[0] != text_embeds.shape[0]:
                raise ValueError(
                    f"labels batch size 不匹配: "
                    f"text_embeds batch={text_embeds.shape[0]}, "
                    f"labels batch={labels.shape[0]}"
                )
            if labels.shape[1] != text_embeds.shape[1]:
                raise ValueError(
                    f"labels 长度应该和 text_embeds 一致: "
                    f"labels length={labels.shape[1]}, "
                    f"text_embeds length={text_embeds.shape[1]}"
                )
        
        # ===== 主逻辑 =====
        batch_size = text_embeds.shape[0]
        text_len = text_embeds.shape[1]
        
        # Generate map queries and move to same device/dtype as text_embeds
        map_query_embeds = self.map_queries(batch_size)  # (B, 1050, hidden_size)
        map_query_embeds = map_query_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype)
        
        # Concatenate: [text+scene] + [map_queries]
        # If scene_tokens is None, it means scene tokens are already in text_embeds
        if scene_tokens is not None:
            scene_len = scene_tokens.shape[1]
            inputs_embeds = torch.cat([text_embeds, scene_tokens, map_query_embeds], dim=1)
        else:
            scene_len = 0  # Scene tokens already included in text_embeds
            inputs_embeds = torch.cat([text_embeds, map_query_embeds], dim=1)
        
        # Extend labels
        if labels is not None:
            scene_labels = torch.full(
                (batch_size, scene_len),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            query_labels = torch.full(
                (batch_size, 1050),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([labels, scene_labels, query_labels], dim=1)
        
        # Create attention mask
        if self.use_custom_mask:
            # Create 4D custom attention mask: (B, 1, seq, seq)
            # Format: 1.0 = can attend, 0.0 = cannot attend
            # transformers will convert this to additive mask internally
            attention_mask = MapAttentionMask.create_mask(
                batch_size=batch_size,
                text_len=text_len,
                scene_len=scene_len,
                num_instances=50,
                num_points=20,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            # Note: transformers expects (B, 1, Q, K) with 1=attend, 0=mask
            # It will internally convert to additive mask: (1.0 - mask) * -inf
        else:
            # Standard 2D mask for causal attention
            total_len = inputs_embeds.shape[1]
            attention_mask = torch.ones(batch_size, total_len, device=inputs_embeds.device)
        
        # Forward through LLM
        outputs = super(LlavaLlamaForCausalLM, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract map features if requested
        if return_map_features:
            hidden_states = outputs.hidden_states[-1]
            
            # 计算 prefix 长度（text + scene）
            prefix_len = text_len + scene_len
            
            # 【优化】提取 LLM 输出的原始 query outputs，保持原始顺序
            # 原始顺序: [Inst0, P0_1..P0_20, Inst1, P1_1..P1_20, ..., Inst49, P49_1..P49_20]
            # 这个顺序与 MapInstancePointQueries 生成的顺序一致
            query_outputs = hidden_states[:, prefix_len:, :]  # (B, 1050, hidden_size)
            
            # 同时提取分离的 instance 和 point features（用于兼容性）
            instance_features, point_features = MapQueryExtractor.extract_features(
                llm_output=hidden_states,
                text_len=text_len,
                scene_len=scene_len,
                num_instances=50,
                num_points=20
            )
            
            # 提取 Scene Tokens (经过 LLM 处理后的)
            # 注意：这个值目前不使用（map_llava_model.py 使用原始 Q-Former 输出）
            # 但保留以供将来可能的需求
            if scene_len > 0:
                # scene tokens 是单独传入的情况
                scene_start = text_len
                scene_end = text_len + scene_len
            else:
                # scene tokens 已经嵌入到 text_embeds 中
                # 这种情况下无法准确提取 LLM 处理后的 scene tokens
                # 设置为 None 表示不可用
                scene_start = None
                scene_end = None
            
            # 提取经过 LLM 处理后的 scene tokens
            if scene_start is not None:
                scene_tokens_processed = hidden_states[:, scene_start:scene_end, :]
            else:
                scene_tokens_processed = None  # 表示 scene tokens 已嵌入 text，无法单独提取
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'instance_features': instance_features,  # (B, 50, hidden_size) - 分离的实例特征
                'point_features': point_features,        # (B, 50, 20, hidden_size) - 分离的点特征
                'query_outputs': query_outputs,          # (B, 1050, hidden_size) - 【新增】原始顺序
                'scene_tokens': scene_tokens_processed,  # (B, 768, hidden_size) or None
            }
        
        return outputs

