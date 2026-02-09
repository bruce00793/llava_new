"""
Complete End-to-End LLaVA Map Detection Model

Flow:
    Images (6 views) → Q-Former → Scene Tokens (768)
                                        ↓
    Text Prompt → Embed → Text Embeds
                                        ↓
    Learnable Queries (1050) → [Text + Scene + Queries]
                                        ↓
                                    LLM Forward
                                        ↓
                Extract Instance/Point Features + Scene Tokens
                  - instance_features: (B, 50, 4096)
                  - point_features: (B, 50, 20, 4096)
                  - scene_tokens: (B, 768, 4096)
                                        ↓
                    ┌───────────────────────────────────────────┐
                    │   Map-Scene Interaction Layer (新增！)     │
                    │                                           │
                    │   Map Features ←─Cross-Attention─→ Scene  │
                    │   (让 Map Queries 直接从图像提取信息)        │
                    └───────────────────────────────────────────┘
                                        ↓
                    Map Decoder (Instance-Conditioned Point Prediction)
                      - inst_reduced + pt_reduced → concat → PointHead
                                        ↓
                            Predictions (logits, points, bbox)

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer

from .qformer import QFormer, build_qformer
from .language_model.llava_map import LlavaMapDetectionModel
from .map_decoder import MapDecoder
from .map_config import MapDetectionConfig, DEFAULT_MAP_CONFIG
from .map_scene_interaction import MapSceneInteractionLayer, build_map_scene_interaction

# LoRA support
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA fine-tuning will not be available.")
    print("Install with: pip install peft")


class LLaVAMapDetector(nn.Module):
    """
    Complete end-to-end model for map detection using LLaVA architecture.
    
    Components:
    1. Q-Former: 6 camera images → 768 scene tokens
    2. LLM: text + scene + 1050 queries → hidden states
    3. Decoder: hidden states → predictions
    """
    
    def __init__(
        self,
        qformer_config: dict,
        llm_path: str = "lmsys/vicuna-7b-v1.5",
        map_config: MapDetectionConfig = None,
        freeze_llm: bool = True,
        qformer_pretrained_path: Optional[str] = None,
        use_lora: bool = True,           # 默认启用 LoRA 微调
        lora_r: int = 32,                 # 增加 rank 以提供足够学习能力
        lora_alpha: int = 64,             # 保持 alpha/r = 2
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = None,
    ):
        """
        Args:
            qformer_config: Config dict for Q-Former
            llm_path: Path to pretrained LLM
            map_config: Map detection config
            freeze_llm: Whether to freeze LLM parameters (ignored if use_lora=True)
            qformer_pretrained_path: Path to BLIP-2 pretrained Q-Former weights (optional)
            use_lora: Whether to use LoRA fine-tuning for LLM (default: True)
            lora_r: LoRA rank (default: 32, 增加以适应空间理解任务)
            lora_alpha: LoRA alpha scaling factor (default: 64, 保持 alpha/r=2)
            lora_dropout: LoRA dropout (default: 0.1)
            lora_target_modules: Which modules to apply LoRA 
                (default: ["q_proj", "k_proj", "v_proj", "o_proj"] - 只微调 Attention 层)
        """
        self.use_lora = use_lora
        super().__init__()
        
        self.config = map_config or DEFAULT_MAP_CONFIG
        
        # 1. Q-Former for multi-view encoding
        print(f"\n{'='*60}")
        print(f"Initializing Q-Former...")
        print(f"{'='*60}")
        self.qformer = build_qformer(qformer_config)
        # Move Q-Former to GPU
        # Note: Keep FP32 for numerical stability, will be cast to FP16 via autocast if needed
        self.qformer = self.qformer.cuda()
        
        # Load pretrained Q-Former weights if provided
        if qformer_pretrained_path is not None:
            self._load_qformer_pretrained(qformer_pretrained_path)
        else:
            print(f"⚠️  Q-Former initialized from scratch (random weights)")
            print(f"   Tip: Use qformer_pretrained_path='blip2' for better performance")
        
        # 2. LLM with map queries
        print(f"\n{'='*60}")
        print(f"Loading LLM: {llm_path}")
        print(f"{'='*60}")
        
        # 检测是否在分布式训练环境中
        # 如果是分布式训练，不使用 device_map="auto"（会分布到多GPU导致问题）
        # 而是先加载到 CPU，后续由 .cuda() 和 DDP 处理
        import os
        is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
        is_single_gpu = torch.cuda.device_count() == 1
        
        # 【关键】使用 BF16 而非 FP16 加载 LLM
        # BF16 指数范围与 FP32 相同（max ~3.4e38），反向传播梯度不会溢出
        # FP16 max 仅 65504，7B 模型 32 层反向传播梯度必然溢出
        # RTX 4090 (Ada Lovelace, compute capability 8.9) 完全支持 BF16
        llm_dtype = torch.bfloat16
        self.llm_dtype = llm_dtype
        print(f"  LLM dtype: {llm_dtype} (BF16 prevents gradient overflow in backward pass)")
        
        if is_distributed or is_single_gpu:
            print(f"  Mode: {'Distributed' if is_distributed else 'Single GPU'} - loading to CPU first")
            self.llm = LlavaMapDetectionModel.from_pretrained(
                llm_path,
                torch_dtype=llm_dtype,
                device_map=None,
                low_cpu_mem_usage=True,
            )
        else:
            print(f"  Mode: Multi-GPU Auto - using device_map='auto'")
            self.llm = LlavaMapDetectionModel.from_pretrained(
                llm_path,
                torch_dtype=llm_dtype,
                device_map="auto",
            )
        
        # Fix: Convert map_queries to FP32 for stable training
        # (FP16 parameters can overflow during optimizer updates)
        print(f"Converting Map Queries to FP32 for training stability...")
        self.llm.map_queries = self.llm.map_queries.float()
        
        # Re-initialize with proper values in FP32
        with torch.no_grad():
            device = self.llm.map_queries.instance_content.device
            
            # Re-init instance content
            self.llm.map_queries.instance_content.data = torch.randn(
                self.llm.map_queries.instance_content.shape,
                device=device, dtype=torch.float32
            ) * 0.02
            
            # Re-init point content
            self.llm.map_queries.point_content.data = torch.randn(
                self.llm.map_queries.point_content.shape,
                device=device, dtype=torch.float32
            ) * 0.02
        
        # Verify
        print(f"✅ Map Queries: dtype={self.llm.map_queries.instance_content.dtype}, "
              f"no_nan={not torch.isnan(self.llm.map_queries.instance_content).any()}")
        
        print(f"✅ LLM loaded successfully!")
        
        # 3. Tokenizer for text (from local path)
        print(f"\nLoading tokenizer from local path...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, local_files_only=True)
        print(f"✅ Tokenizer loaded from local: {llm_path}")
        
        # 4. Map-Scene Interaction Layer (新增！)
        # 在 LLM 输出后、Decoder 之前，让 Map Features 直接和 Scene Tokens 交互
        print(f"\n{'='*60}")
        print(f"Initializing Map-Scene Interaction Layer...")
        print(f"{'='*60}")
        self.map_scene_interaction = build_map_scene_interaction(
            input_dim=4096,      # LLM hidden size
            embed_dim=256,       # 交互层维度
            num_heads=8,         # 注意力头数
            num_layers=6,        # 6 层交互（与 MapTR Decoder 对齐）
            ffn_dim=1024,        # FFN 维度
            dropout=0.1,
        )
        self.map_scene_interaction = self.map_scene_interaction.cuda()
        print(f"✅ Map-Scene Interaction Layer initialized (6 layers)")
        
        # 5. Decoder for predictions
        print(f"\n{'='*60}")
        print(f"Initializing Map Decoder...")
        print(f"{'='*60}")
        self.decoder = MapDecoder(self.config)
        # Move Decoder to GPU
        self.decoder = self.decoder.cuda()
        print(f"✅ Map Decoder initialized (random weights)")
        
        # 6. Loss function (在 __init__ 中创建，而不是 forward 中动态创建)
        # dir_loss 按实例数归一化，量级约 9.5
        # weight_dir=0.25 是折中方案：方向损失贡献约 5-8%，有意义但不主导训练
        from .map_loss import MapDetectionLoss, HungarianMatcher
        self.criterion = MapDetectionLoss(
            num_classes=3,
            weight_cls=2.0,
            weight_pts=5.0,
            weight_dir=0.25,  # 折中方案（MapTR用0.005几乎无作用，2.0会主导训练）
        )
        self._aux_matcher = HungarianMatcher(cost_class=2.0, cost_points=5.0)
        print(f"✅ Loss function initialized")
        
        # 7. LoRA or Freeze LLM
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA. Install with: pip install peft")
            
            print(f"\n{'='*60}")
            print(f"Applying LoRA to LLM...")
            print(f"{'='*60}")
            
            # Default target modules for LLaMA-based models
            # 针对地图检测任务优化的 LoRA 配置：
            # - q_proj: Map Queries 如何查询 Scene Tokens（核心）
            # - k_proj: Scene Tokens 如何被索引（重要）
            # - v_proj: Scene Tokens 提供什么信息（核心）
            # - o_proj: Attention 输出投影（重要）
            # 注：不包含 MLP 层，因为检测任务主要依赖 Attention 机制
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            
            # Apply LoRA to LLM
            self.llm = get_peft_model(self.llm, lora_config)
            
            # Make sure map_queries are trainable
            for param in self.llm.base_model.model.map_queries.parameters():
                param.requires_grad = True
            
            # 【关键修复】将 LoRA 参数转换为 FP32
            # 原因：LLM 以 FP16 加载，LoRA 继承 FP16。
            # 在 FP16 下，GradScaler 缩放后的梯度极易溢出（FP16 max=65504）
            # 导致每个梯度步都出现 NaN/Inf。
            # 转为 FP32 后，梯度范围扩大到 3.4e38，彻底解决溢出问题。
            lora_param_count = 0
            for name, param in self.llm.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    param.data = param.data.float()
                    lora_param_count += 1
            print(f"✅ Converted {lora_param_count} LoRA parameters to FP32")
            
            print(f"✅ LoRA applied to LLM!")
            print(f"   - LoRA rank (r): {lora_r}")
            print(f"   - LoRA alpha: {lora_alpha}")
            print(f"   - LoRA dropout: {lora_dropout}")
            print(f"   - Target modules: {lora_target_modules}")
            self.llm.print_trainable_parameters()
            
        elif freeze_llm:
            print(f"\n{'='*60}")
            print(f"Freezing LLM backbone parameters...")
            print(f"{'='*60}")
            for param in self.llm.model.parameters():
                param.requires_grad = False
            # Only train map_queries
            for param in self.llm.map_queries.parameters():
                param.requires_grad = True
            print(f"✅ LLM frozen, only training:")
            print(f"   - Q-Former")
            print(f"   - Map Queries (1050 learnable queries)")
            print(f"   - Map-Scene Interaction Layer")
            print(f"   - Map Decoder")
        else:
            print(f"\n{'='*60}")
            print(f"Full LLM fine-tuning enabled (not recommended)")
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"✅ LLaVAMapDetector initialized successfully!")
        print(f"{'='*60}")
        self._print_trainable_params()
    
    def _load_qformer_pretrained(self, pretrained_path: str):
        """
        Load pretrained Q-Former weights from BLIP-2 or custom checkpoint.
        
        Args:
            pretrained_path: 
                - 'blip2': Load from Salesforce BLIP-2
                - Local path: Load from local checkpoint
        """
        import os
        
        if pretrained_path == 'blip2':
            print(f"📥 Loading BLIP-2 pretrained Q-Former...")
            
            # Use local BLIP-2 path if available
            local_blip2_path = "/home/cly/auto/llava_test/LLaVA/blip2-opt-2.7b"
            
            try:
                from transformers import Blip2Model
                
                # Check if local BLIP-2 exists
                if os.path.exists(local_blip2_path):
                    print(f"   Loading from local: {local_blip2_path}")
                    blip2 = Blip2Model.from_pretrained(
                        local_blip2_path,
                        torch_dtype=torch.float32,
                        local_files_only=True,
                    )
                else:
                    print(f"   ⚠️ Local BLIP-2 not found, trying remote: Salesforce/blip2-opt-2.7b")
                    blip2 = Blip2Model.from_pretrained(
                        "Salesforce/blip2-opt-2.7b",
                        torch_dtype=torch.float32,
                    )
                
                # Extract Q-Former components
                qformer_state = {}
                for name, param in blip2.named_parameters():
                    if 'qformer' in name or 'query_tokens' in name:
                        # Remove prefix
                        new_name = name.replace('qformer.', '')
                        new_name = new_name.replace('language_model.', '')
                        qformer_state[new_name] = param.data
                
                # Load into our Q-Former (partial loading, ignore size mismatch)
                missing, unexpected = self.qformer.load_state_dict(qformer_state, strict=False)
                
                print(f"✅ BLIP-2 Q-Former loaded!")
                print(f"   Loaded parameters: {len(qformer_state)}")
                if len(missing) > 0:
                    print(f"   Missing keys (will use random init): {len(missing)}")
                if len(unexpected) > 0:
                    print(f"   Unexpected keys (ignored): {len(unexpected)}")
                
                del blip2  # Free memory
                
            except Exception as e:
                print(f"⚠️  Failed to load BLIP-2 weights: {e}")
                print(f"   Falling back to random initialization")
        
        elif os.path.exists(pretrained_path):
            print(f"📥 Loading Q-Former from local checkpoint...")
            print(f"   Path: {pretrained_path}")
            
            try:
                # Check if it's a directory (HuggingFace model format)
                if os.path.isdir(pretrained_path):
                    print(f"   Detected HuggingFace model directory, using from_pretrained...")
                    from transformers import Blip2Model
                    
                    # Load BLIP-2 model from local directory
                    blip2 = Blip2Model.from_pretrained(
                        pretrained_path,
                        torch_dtype=torch.float32,
                    )
                    
                    # Extract Q-Former components
                    qformer_state = {}
                    for name, param in blip2.named_parameters():
                        if 'qformer' in name or 'query_tokens' in name:
                            new_name = name.replace('qformer.', '')
                            new_name = new_name.replace('language_model.', '')
                            qformer_state[new_name] = param.data
                    
                    # Load into our Q-Former (partial loading)
                    missing, unexpected = self.qformer.load_state_dict(qformer_state, strict=False)
                    
                    print(f"✅ BLIP-2 Q-Former loaded from local directory!")
                    print(f"   Loaded parameters: {len(qformer_state)}")
                    if len(missing) > 0:
                        print(f"   Missing keys (will use random init): {len(missing)}")
                    if len(unexpected) > 0:
                        print(f"   Unexpected keys (ignored): {len(unexpected)}")
                    
                    del blip2  # Free memory
                else:
                    # It's a single file checkpoint
                    state_dict = torch.load(pretrained_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'qformer' in state_dict:
                        state_dict = state_dict['qformer']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                    
                    missing, unexpected = self.qformer.load_state_dict(state_dict, strict=False)
                    
                    print(f"✅ Q-Former checkpoint loaded!")
                    if len(missing) > 0:
                        print(f"   Missing keys: {len(missing)}")
                    if len(unexpected) > 0:
                        print(f"   Unexpected keys: {len(unexpected)}")
                    
            except Exception as e:
                print(f"⚠️  Failed to load checkpoint: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Falling back to random initialization")
        
        else:
            print(f"⚠️  Pretrained path not found: {pretrained_path}")
            print(f"   Using random initialization")
    
    def _print_trainable_params(self):
        """Print trainable parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nParameter Statistics:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        return_loss: bool = False,
        gt_labels: Optional[torch.Tensor] = None,
        gt_points: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        cam_intrinsics: Optional[torch.Tensor] = None,
        cam_extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of complete model.
        
        Args:
            images: (B, 6, 3, 448, 800) - 6 camera views (H=448, W=800)
            text_ids: (B, L) - Tokenized text with IMAGE_TOKEN_INDEX=-200
            return_loss: Whether to compute loss (requires GT)
            cam_intrinsics: (B, 6, 3, 3) - Camera intrinsic matrices (optional, for 3D pos encoding)
            cam_extrinsics: (B, 6, 4, 4) - Camera extrinsic matrices (optional, for 3D pos encoding)
            gt_labels: (B, M) - Ground truth class labels
            gt_points: (B, M, 20, 2) - Ground truth points
            gt_masks: (B, M) - Valid GT mask
        
        Returns:
            dict with keys:
                - pred_logits: (B, 50, 3) classification logits
                - pred_points: (B, 50, 20, 2) point coordinates
                - instance_features: (B, 50, 4096) (optional)
                - point_features: (B, 50, 20, 4096) (optional)
                - loss_dict: dict of losses (if return_loss=True)
        """
        batch_size = images.shape[0]
        
        # ===== Step 1: Q-Former - Images to Scene Tokens =====
        # Pass camera parameters for 3D position encoding if available
        scene_tokens = self.qformer(
            images, 
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics
        )  # (B, 768, 4096)
        
        # 【安全检查】确认 Q-Former 输出正常（修复 autocast 后不应再出现 NaN）
        if torch.isnan(scene_tokens).any() or torch.isinf(scene_tokens).any():
            print(f"❌ [Forward] Q-Former output still contains NaN/Inf after autocast fix! "
                  f"This indicates a deeper issue.", flush=True)
        
        # Convert to match LLM precision (BF16)
        scene_tokens = scene_tokens.to(self.llm_dtype)
        
        # ===== Step 2: Embed Text and Replace IMAGE_TOKEN =====
        # Handle IMAGE_TOKEN_INDEX (-200) which is a placeholder for scene tokens
        from llava.constants import IMAGE_TOKEN_INDEX
        
        # Replace IMAGE_TOKEN_INDEX with a valid token ID (0 = pad token) temporarily
        text_ids_safe = text_ids.clone()
        image_token_mask = (text_ids == IMAGE_TOKEN_INDEX)
        text_ids_safe[image_token_mask] = 0  # Use pad token temporarily
        
        # Get text embeddings
        # Note: 需要处理 LoRA 包装后的访问路径
        if self.use_lora:
            # LoRA 包装后路径: base_model.model.model.embed_tokens
            embed_tokens = self.llm.base_model.model.model.embed_tokens
        else:
            # 原始路径: model.embed_tokens
            embed_tokens = self.llm.model.embed_tokens
        text_embeds_temp = embed_tokens(text_ids_safe)  # (B, L, 4096)
        
        # Replace the embeddings at IMAGE_TOKEN positions with scene_tokens
        # Create new list to hold embeddings with varying lengths
        text_embeds_list = []
        expected_length = None
        
        for b in range(batch_size):
            image_positions = torch.where(image_token_mask[b])[0]
            if len(image_positions) > 0:
                # Replace IMAGE_TOKEN embedding with scene_tokens
                pos = image_positions[0].item()
                # Concatenate: text[:pos] + scene_tokens + text[pos+1:]
                new_embeds = torch.cat([
                    text_embeds_temp[b, :pos],
                    scene_tokens[b],  # Insert 768 scene tokens
                    text_embeds_temp[b, pos+1:]
                ], dim=0)
            else:
                # 没有 IMAGE_TOKEN，直接使用原始 embeddings
                # 这种情况不应该发生，添加警告
                import warnings
                warnings.warn(f"Batch {b} has no IMAGE_TOKEN! This may cause issues.")
                new_embeds = text_embeds_temp[b]
            
            # 检查长度一致性
            if expected_length is None:
                expected_length = new_embeds.shape[0]
            else:
                if new_embeds.shape[0] != expected_length:
                    raise ValueError(
                        f"Batch {b} has different embedding length {new_embeds.shape[0]} "
                        f"vs expected {expected_length}. "
                        f"Ensure all samples have IMAGE_TOKEN at the same position."
                    )
            
            text_embeds_list.append(new_embeds)
        
        # Stack back into tensor (all should have same length now)
        # 注：768 scene tokens 替换 1 个 IMAGE_TOKEN = 净增 767 个 tokens
        text_embeds = torch.stack(text_embeds_list, dim=0)  # (B, L+767, 4096)
        
        # ===== Step 3: LLM Forward with Map Queries =====
        # This will:
        # - Add 1050 learnable queries
        # - Concatenate [text_with_scene, queries]
        # - Forward through LLM
        # - Extract instance and point features from query positions
        # Note: text_embeds now includes scene_tokens, so we pass it directly
        # and set scene_tokens=None to avoid double-adding
        
        # 处理 LoRA 模式：PEFT 包装后需要通过 base_model 访问自定义方法
        if self.use_lora:
            # LoRA 模式：通过 base_model 调用 forward_with_map
            llm_output = self.llm.base_model.forward_with_map(
                text_embeds=text_embeds,  # Already includes scene tokens
                scene_tokens=None,  # Don't add scene tokens again
                return_map_features=True,
            )
        else:
            # 非 LoRA 模式：直接调用
            llm_output = self.llm.forward_with_map(
                text_embeds=text_embeds,  # Already includes scene tokens
                scene_tokens=None,  # Don't add scene tokens again
                return_map_features=True,
            )
        
        # 【优化】使用 query_outputs 保持 LLM 原始输出顺序
        # 原始顺序: [Inst0, P0_1..P0_20, Inst1, P1_1..P1_20, ..., Inst49, P49_1..P49_20]
        query_outputs = llm_output['query_outputs']  # (B, 1050, 4096) - 保持原始顺序
        
        # 【安全检查】确认 LLM 输出正常
        if torch.isnan(query_outputs).any() or torch.isinf(query_outputs).any():
            print(f"❌ [Forward] LLM query_outputs contains NaN/Inf!", flush=True)
        
        # 获取维度信息
        B = query_outputs.shape[0]
        N_inst = 50   # 实例数量
        N_pts = 20    # 每个实例的点数量
        queries_per_inst = 1 + N_pts  # 21 (1 instance query + 20 point queries)
        H = query_outputs.shape[2]  # hidden_size (4096)
        
        # ===== Step 4: Map-Scene Interaction (新增！) =====
        # 让 Map Features 直接和 Scene Tokens 做 Cross-Attention
        # 
        # 【重要设计决策】
        # 1. 使用**原始 scene tokens**（Q-Former 输出），而非 LLM 处理后的
        # 2. 【优化】保持 LLM 输出的原始顺序送入 Map-Scene Interaction
        #    - 原始顺序: [Inst0, P0_1..P0_20, Inst1, P1_1..P1_20, ...]
        #    - 好处: 同一实例的 instance 和 points 在序列中相邻，
        #           Self-Attention 时更容易建立局部关联
        #
        # Map-Scene Interaction: Cross-Attention
        # 【关键】使用原始 scene_tokens（Q-Former 直接输出），不使用 LLM 处理后的
        scene_tokens_for_interaction = scene_tokens  # 使用 Q-Former 原始输出
        
        # 确保 dtype 一致（转为 FP32 以保证数值稳定性）
        map_features_combined = query_outputs.to(dtype=torch.float32)
        scene_tokens_for_interaction = scene_tokens_for_interaction.to(dtype=torch.float32)
        
        # Cross-Attention: Map Features 从 Scene Tokens 提取视觉信息
        enhanced_map_features = self.map_scene_interaction(
            map_features=map_features_combined,
            scene_tokens=scene_tokens_for_interaction,
        )  # (B, 1050, 4096)
        
        # 【安全检查】确认 Map-Scene Interaction 输出正常
        if torch.isnan(enhanced_map_features).any() or torch.isinf(enhanced_map_features).any():
            print(f"❌ [Forward] Map-Scene Interaction output contains NaN/Inf!", flush=True)
        
        # 【优化】从增强后的特征中按原始顺序重新提取 instance 和 point features
        # 原始顺序: [Inst0, P0_1..P0_20, Inst1, P1_1..P1_20, ...]
        instance_features_list = []
        point_features_list = []
        
        for i in range(N_inst):
            start_idx = i * queries_per_inst
            # Instance query 位于每组的第一个位置
            inst_feat = enhanced_map_features[:, start_idx:start_idx+1, :]  # (B, 1, H)
            instance_features_list.append(inst_feat)
            # Point queries 位于 instance 之后的 20 个位置
            point_feat = enhanced_map_features[:, start_idx+1:start_idx+queries_per_inst, :]  # (B, 20, H)
            point_features_list.append(point_feat)
        
        # 拼接成最终形状
        instance_features = torch.cat(instance_features_list, dim=1)  # (B, 50, H)
        point_features = torch.stack(point_features_list, dim=1)      # (B, 50, 20, H)
        
        # ===== Step 5: Decode to Predictions =====
        # Move features to decoder's device and dtype
        decoder_device = next(self.decoder.parameters()).device
        decoder_dtype = next(self.decoder.parameters()).dtype
        instance_features = instance_features.to(device=decoder_device, dtype=decoder_dtype)
        point_features = point_features.to(device=decoder_device, dtype=decoder_dtype)
        
        # 注：不再 clamp Decoder 输入，避免阻断梯度流
        
        # Instance-Conditioned Point Prediction
        # Uses both instance_features and point_features
        decoder_output = self.decoder(instance_features, point_features)
        
        pred_logits = decoder_output['class_logits']  # (B, 50, 3)
        pred_points = decoder_output['points']        # (B, 50, 20, 2)
        pred_bbox = decoder_output['bbox']            # (B, 50, 4)
        
        # 【安全检查】确认 Decoder 输出正常
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print(f"❌ [Forward] Decoder pred_logits contains NaN/Inf!", flush=True)
        if torch.isnan(pred_points).any() or torch.isinf(pred_points).any():
            print(f"❌ [Forward] Decoder pred_points contains NaN/Inf!", flush=True)
        
        # Build output dict
        output = {
            'pred_logits': pred_logits,
            'pred_points': pred_points,
            'pred_bbox': pred_bbox,
            'instance_features': instance_features,
            'point_features': point_features,
        }
        
        # ===== Step 6: Compute Loss (if requested) =====
        if return_loss:
            if gt_labels is None or gt_points is None or gt_masks is None:
                raise ValueError("GT data required for loss computation")
            
            # 【关键修复】损失计算必须在 FP32 下执行！
            # 原因：autocast 会把某些操作降为 FP16（如 F.binary_cross_entropy_with_logits），
            # 导致大 loss 值在反向传播时产生 FP16 梯度溢出。
            # 使用 autocast(enabled=False) 确保所有损失计算都在 FP32 下。
            with torch.cuda.amp.autocast(enabled=False):
                # 确保所有输入都是 FP32
                pred_logits_f32 = pred_logits.float()
                pred_points_f32 = pred_points.float()
                
                # Prepare GT lists (Loss expects lists)
                gt_labels_list = []
                gt_points_list = []
                gt_masks_list = []
                
                for b in range(batch_size):
                    mask = gt_masks[b]  # (M,)
                    num_valid = int(mask.sum().item())  # 确保是整数，避免浮点切片错误
                    
                    if num_valid > 0:
                        gt_labels_list.append(gt_labels[b, :num_valid])
                        gt_points_list.append(gt_points[b, :num_valid].float())
                        gt_masks_list.append(torch.ones(num_valid, 20, dtype=torch.bool, device=mask.device))
                    else:
                        # Empty GT
                        gt_labels_list.append(torch.empty(0, dtype=torch.long, device=mask.device))
                        gt_points_list.append(torch.empty(0, 20, 2, dtype=torch.float32, device=mask.device))
                        gt_masks_list.append(torch.empty(0, 20, dtype=torch.bool, device=mask.device))
                
                # Compute main loss (final points)
                total_loss, loss_dict = self.criterion(
                    pred_logits=pred_logits_f32,
                    pred_lines=pred_points_f32,
                    gt_labels=gt_labels_list,
                    gt_lines=gt_points_list,
                    gt_masks=gt_masks_list,
                )
                
                # ========== 辅助损失：监督初始点和中间层（完整监督）==========
                # 与 MapTR 一致：对每个中间层计算完整的 cls + pts + dir 损失
                if 'init_points' in decoder_output and 'intermediate_points' in decoder_output:
                    intermediate_points = decoder_output['intermediate_points']
                    
                    # intermediate_points 包含: [init, layer1, layer2, layer3, layer4, layer5, layer6(=final)]
                    # 我们监督除最终层外的所有中间层 (共 6 个)
                    # 【修改】辅助损失权重：统一权重，避免梯度冲突导致训练不稳定
                    # 递增权重 [0.1-0.6] 会导致浅层监督不足、深层过强，辅助损失爆炸（6.7倍主损失）
                    # MapTR 3层用统一1.0，我们6层用统一0.5，总权重3.0倍（对齐MapTR）
                    num_aux = len(intermediate_points) - 1  # 不包括最终层
                    aux_weights = [0.5 for _ in range(num_aux)]  # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    
                    aux_loss_total = 0.0
                    for i, aux_pts in enumerate(intermediate_points[:-1]):
                        # 计算完整的辅助损失（cls + pts + dir），与 MapTR 一致
                        aux_loss_dict = self._compute_aux_full_loss(
                            pred_logits=pred_logits_f32,  # 分类使用最终层的 logits
                            pred_points=aux_pts.float(),
                            gt_labels_list=gt_labels_list,
                            gt_points_list=gt_points_list,
                            gt_masks_list=gt_masks_list,
                        )
                        
                        # 加权求和：与主损失使用相同的权重比例
                        aux_layer_loss = (
                            self.criterion.weight_cls * aux_loss_dict['cls'] +
                            self.criterion.weight_pts * aux_loss_dict['pts'] +
                            self.criterion.weight_dir * aux_loss_dict['dir']
                        )
                        aux_loss_total = aux_loss_total + aux_weights[i] * aux_layer_loss
                        
                        # 记录各项损失（用于监控）
                        loss_dict[f'loss_aux_{i}_cls'] = aux_loss_dict['cls'].detach()
                        loss_dict[f'loss_aux_{i}_pts'] = aux_loss_dict['pts'].detach()
                        loss_dict[f'loss_aux_{i}_dir'] = aux_loss_dict['dir'].detach()
                    
                    total_loss = total_loss + aux_loss_total
                    loss_dict['loss_aux_total'] = aux_loss_total.detach() if isinstance(aux_loss_total, torch.Tensor) else torch.tensor(aux_loss_total)
            
            output['loss_dict'] = loss_dict
            output['loss'] = total_loss
        
        return output
    
    def _compute_aux_full_loss(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels_list: list,
        gt_points_list: list,
        gt_masks_list: list,
    ) -> Dict[str, torch.Tensor]:
        """
        计算完整的辅助损失（cls + pts + dir），与 MapTR 设计一致。
        
        对每个中间层都计算完整的三项损失，提供更强的监督信号。
        
        Args:
            pred_logits: [B, N, 3] 分类预测（使用最终层的 logits）
            pred_points: [B, N, P, 2] 中间层的点预测
            gt_labels_list: List[Tensor] 真实标签
            gt_points_list: List[Tensor] 真实点坐标
            gt_masks_list: List[Tensor] 点有效掩码
            
        Returns:
            Dict with 'cls', 'pts', 'dir' losses
        """
        B = pred_points.shape[0]
        N = pred_points.shape[1]
        device = pred_points.device
        dtype = pred_points.dtype
        
        # 使用 matcher 获取匹配关系
        indices = self._aux_matcher(
            pred_logits, pred_points, 
            gt_labels_list, gt_points_list, gt_masks_list
        )
        
        # ========== 1. 分类损失 ==========
        target_classes = torch.full((B, N), 3, dtype=torch.long, device=device)  # 3 = background
        num_total_pos = 0
        
        for b, (pred_idx, gt_idx, _) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = gt_labels_list[b][gt_idx].to(device)
                num_total_pos += len(pred_idx)
        
        avg_factor = max(num_total_pos, 1.0)
        pred_logits_flat = pred_logits.reshape(-1, 3)
        target_classes_flat = target_classes.reshape(-1)
        loss_cls = self.criterion.focal_loss(pred_logits_flat, target_classes_flat, avg_factor=avg_factor)
        
        # ========== 2. 点距离损失 ==========
        all_pts_loss = []
        for b, (pred_idx, gt_idx, best_gt) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred = pred_points[b, pred_idx]
            matched_gt = best_gt.to(device=device, dtype=dtype)
            matched_mask = gt_masks_list[b][gt_idx].to(device=device)
            
            diff = torch.abs(matched_pred - matched_gt)
            diff_masked = diff * matched_mask.unsqueeze(-1)
            all_pts_loss.append(diff_masked.sum())
        
        if num_total_pos == 0:
            loss_pts = pred_points.sum() * 0.0
        else:
            loss_pts = sum(all_pts_loss) / avg_factor
        
        # ========== 3. 方向损失 ==========
        all_dir_loss = []
        for b, (pred_idx, gt_idx, best_gt) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            matched_pred = pred_points[b, pred_idx]
            matched_gt = best_gt.to(device=device, dtype=dtype)
            matched_mask = gt_masks_list[b][gt_idx].to(device=device)
            
            # 反归一化到物理坐标
            pred_denorm = matched_pred.clone()
            pred_denorm[..., 0] = matched_pred[..., 0] * 15.0
            pred_denorm[..., 1] = matched_pred[..., 1] * 30.0
            gt_denorm = matched_gt.clone()
            gt_denorm[..., 0] = matched_gt[..., 0] * 15.0
            gt_denorm[..., 1] = matched_gt[..., 1] * 30.0
            
            # 计算方向向量
            pred_dirs = pred_denorm[:, 1:] - pred_denorm[:, :-1]
            gt_dirs = gt_denorm[:, 1:] - gt_denorm[:, :-1]
            
            # 【根本修复】使用 sqrt(x^2 + eps) 代替 .norm()
            # .norm() 在零点梯度为 NaN (0/0)，torch.where 无法屏蔽（NaN*0=NaN）
            eps_sq = 1e-6
            pred_len = torch.sqrt((pred_dirs ** 2).sum(dim=-1, keepdim=True) + eps_sq)
            gt_len = torch.sqrt((gt_dirs ** 2).sum(dim=-1, keepdim=True) + eps_sq)
            
            # 安全归一化
            pred_dirs_norm = pred_dirs / pred_len
            gt_dirs_norm = gt_dirs / gt_len
            
            # 余弦相似度损失
            cosine_sim = (pred_dirs_norm * gt_dirs_norm).sum(dim=-1).clamp(-1, 1)
            dir_loss = 1.0 - cosine_sim
            
            # 边掩码（用 raw squared length 判断，避免 .norm()）
            edge_mask = matched_mask[:, :-1] & matched_mask[:, 1:]
            raw_pred_len_sq = (pred_dirs ** 2).sum(dim=-1)
            raw_gt_len_sq = (gt_dirs ** 2).sum(dim=-1)
            valid_edge = (raw_pred_len_sq > 1e-4) & (raw_gt_len_sq > 1e-4)
            final_mask = edge_mask & valid_edge
            
            dir_loss_masked = dir_loss * final_mask.to(dir_loss.device)
            all_dir_loss.append(dir_loss_masked.sum())
        
        if num_total_pos == 0:
            loss_dir = pred_points.sum() * 0.0
        else:
            # 按实例数归一化（与 MapTR 一致）
            loss_dir = sum(all_dir_loss) / avg_factor
        
        return {
            'cls': loss_cls,
            'pts': loss_pts,
            'dir': loss_dir,
        }
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        score_threshold: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction.
        
        Args:
            images: (B, 6, 3, 448, 800) - 6 camera views (H=448, W=800)
            text_ids: (B, L)
            score_threshold: Confidence threshold
        
        Returns:
            dict with predictions
        """
        self.eval()
        
        output = self.forward(images, text_ids, return_loss=False)
        
        # Post-process: filter by score
        pred_logits = output['pred_logits']  # (B, 50, 3)
        pred_points = output['pred_points']  # (B, 50, 20, 2)
        
        # Get scores and labels
        pred_probs = torch.softmax(pred_logits, dim=-1)  # (B, 50, 3)
        pred_scores, pred_labels = pred_probs.max(dim=-1)  # (B, 50)
        
        # Filter by threshold
        batch_predictions = []
        for b in range(pred_logits.shape[0]):
            valid_mask = pred_scores[b] >= score_threshold
            
            batch_predictions.append({
                'labels': pred_labels[b][valid_mask],
                'scores': pred_scores[b][valid_mask],
                'points': pred_points[b][valid_mask],
            })
        
        return batch_predictions


def build_map_detector(
    qformer_config_path: str = None,
    llm_path: str = "lmsys/vicuna-7b-v1.5",
    freeze_llm: bool = True,
    qformer_pretrained: str = None,
    use_lora: bool = True,            # 默认启用 LoRA 微调
    lora_r: int = 32,                  # 增加 rank 以提供足够学习能力
    lora_alpha: int = 64,              # 保持 alpha/r = 2
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[list] = None,
) -> LLaVAMapDetector:
    """
    Build complete map detector model.
    
    Args:
        qformer_config_path: Path to Q-Former config (None = use default)
        llm_path: Path to LLM (default: Vicuna-7B)
        freeze_llm: Whether to freeze LLM (default: True, ignored if use_lora=True)
        qformer_pretrained: Q-Former pretrained weights
            - None: Random initialization (not recommended)
            - 'blip2': Load from BLIP-2 (recommended)
            - '/path/to/checkpoint.pth': Load from local file
        use_lora: Whether to use LoRA fine-tuning (default: True, 推荐用于地图检测)
        lora_r: LoRA rank (default: 32, 增加以适应空间理解任务)
        lora_alpha: LoRA alpha (default: 64, 保持 alpha/r=2)
        lora_dropout: LoRA dropout (default: 0.1)
        lora_target_modules: Target modules for LoRA 
            (default: ["q_proj", "k_proj", "v_proj", "o_proj"] - 只微调 Attention 层)
    
    Returns:
        LLaVAMapDetector model
    
    Example:
        >>> # Freeze LLM (default, fast training)
        >>> model = build_map_detector(freeze_llm=True)
        
        >>> # LoRA fine-tuning (recommended for better performance)
        >>> model = build_map_detector(
        ...     use_lora=True,
        ...     lora_r=16,
        ...     lora_alpha=32,
        ... )
        
        >>> # Full fine-tuning (not recommended, requires lots of memory)
        >>> model = build_map_detector(freeze_llm=False)
    """
    # Default Q-Former config
    if qformer_config_path is None:
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': 768,
            'num_decoder_layers': 6,
            'llm_hidden_size': 4096,
            # Enhanced 3D Position Encoding (ABC方案)
            'depth_num': 32,        # 32个深度假设（更密集的深度采样）
            'depth_start': 1.0,     # 最小深度 1米
            'depth_max': 60.0,      # 最大深度 60米
            'use_lid': True,        # 方案B: LID深度分布 (近密远疏)
            # pc_range 格式：[x_min, y_min, z_min, x_max, y_max, z_max]
            # 与 MapConfig 保持一致！MapTR 使用此范围
            'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        }
    else:
        import json
        with open(qformer_config_path, 'r') as f:
            qformer_config = json.load(f)
    
    model = LLaVAMapDetector(
        qformer_config=qformer_config,
        llm_path=llm_path,
        freeze_llm=freeze_llm,
        qformer_pretrained_path=qformer_pretrained,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    
    return model


if __name__ == "__main__":
    print("Testing LLaVAMapDetector...")
    
    # Build model
    model = build_map_detector(freeze_llm=True)
    
    # Test input (H=448 divisible by 32, W=800)
    batch_size = 2
    images = torch.randn(batch_size, 6, 3, 448, 800)
    text_ids = torch.randint(0, 32000, (batch_size, 100))
    
    # GT data
    gt_labels = torch.randint(0, 3, (batch_size, 10))
    gt_points = torch.randn(batch_size, 10, 20, 2)
    gt_masks = torch.ones(batch_size, 10, dtype=torch.bool)
    
    print(f"\nForward pass (with loss)...")
    output = model(
        images=images,
        text_ids=text_ids,
        return_loss=True,
        gt_labels=gt_labels,
        gt_points=gt_points,
        gt_masks=gt_masks,
    )
    
    print(f"\nOutput keys: {output.keys()}")
    print(f"  pred_logits: {output['pred_logits'].shape}")
    print(f"  pred_points: {output['pred_points'].shape}")
    print(f"  loss: {output['loss'].item():.4f}")
    
    print(f"\n✅ Test passed!")

