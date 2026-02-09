"""
Step 3: LLM Processing - Complete Usage Example

This example demonstrates the complete flow from Q-Former output to extracted features.

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
from llava.model.map_queries import (
    MapInstancePointQueries,
    MapAttentionMask,
    MapQueryExtractor
)
from llava.model.language_model.llava_map import LlavaMapDetectionModel


def example_complete_flow():
    """
    Complete example: from inputs to extracted features.
    """
    print("="*80)
    print("STEP 3: LLM Processing - Complete Flow")
    print("="*80)
    
    # ========== Step 1: Prepare Inputs ==========
    print("\n[Step 1] Preparing inputs...")
    
    batch_size = 2
    text_len = 80
    scene_len = 512
    hidden_size = 4096
    
    # Simulate text embeddings (from tokenized prompt)
    text_embeds = torch.randn(batch_size, text_len, hidden_size)
    print(f"  Text embeddings: {text_embeds.shape}")
    
    # Simulate scene tokens (from Q-Former)
    scene_tokens = torch.randn(batch_size, scene_len, hidden_size)
    print(f"  Scene tokens: {scene_tokens.shape}")
    
    # ========== Step 2: Generate Map Queries ==========
    print("\n[Step 2] Generating learnable queries...")
    
    query_module = MapInstancePointQueries(
        num_instances=50,
        num_points=20,
        embed_dim=hidden_size
    )
    
    map_queries = query_module(batch_size)
    print(f"  Map queries: {map_queries.shape}")
    print(f"  Structure: 50 instances × 21 queries/inst = 1050 queries")
    
    # ========== Step 3: Concatenate Sequence ==========
    print("\n[Step 3] Building complete sequence...")
    
    inputs_embeds = torch.cat([text_embeds, scene_tokens, map_queries], dim=1)
    total_len = inputs_embeds.shape[1]
    print(f"  Complete sequence: {inputs_embeds.shape}")
    print(f"  Breakdown:")
    print(f"    - Text: [0, {text_len})")
    print(f"    - Scene: [{text_len}, {text_len + scene_len})")
    print(f"    - Queries: [{text_len + scene_len}, {total_len})")
    
    # ========== Step 4: Create Attention Mask ==========
    print("\n[Step 4] Creating custom attention mask...")
    
    attention_mask = MapAttentionMask.create_mask(
        batch_size=batch_size,
        text_len=text_len,
        scene_len=scene_len,
        num_instances=50,
        num_points=20
    )
    print(f"  Attention mask: {attention_mask.shape}")
    print(f"  Mask rules:")
    print(f"    ✓ All tokens see text + scene")
    print(f"    ✓ Instance queries see each other (50×50)")
    print(f"    ✓ Point queries only see their instance + siblings")
    
    # ========== Step 5: Simulate LLM Forward ==========
    print("\n[Step 5] Simulating LLM forward pass...")
    print("  (In real usage, this would be: outputs = model(inputs_embeds, attention_mask))")
    
    # Simulate LLM output
    llm_output = torch.randn(batch_size, total_len, hidden_size)
    print(f"  LLM output: {llm_output.shape}")
    
    # ========== Step 6: Extract Features ==========
    print("\n[Step 6] Extracting map detection features...")
    
    instance_features, point_features = MapQueryExtractor.extract_features(
        llm_output=llm_output,
        text_len=text_len,
        scene_len=scene_len,
        num_instances=50,
        num_points=20
    )
    
    print(f"  Instance features: {instance_features.shape}")
    print(f"    → 50 instances, each with {hidden_size}-dim feature")
    print(f"    → Used for classification head")
    
    print(f"  Point features: {point_features.shape}")
    print(f"    → 50 instances × 20 points, each with {hidden_size}-dim feature")
    print(f"    → Used for regression head")
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input sequence length: {total_len} tokens")
    print(f"  = {text_len} (text) + {scene_len} (scene) + 1050 (queries)")
    print(f"\nOutput features:")
    print(f"  - Instance features: (B={batch_size}, 50, {hidden_size})")
    print(f"  - Point features: (B={batch_size}, 50, 20, {hidden_size})")
    print(f"\nNext step: Feed features to detection heads")
    print("="*80)


def example_with_real_model():
    """
    Example using the extended LLaVA model (requires model loading).
    """
    print("\n" + "="*80)
    print("EXAMPLE: Using LlavaMapDetectionModel")
    print("="*80)
    print("""
# Pseudo-code (requires actual model)

from llava.model.language_model.llava_map import LlavaMapDetectionModel

# 1. Load model
model = LlavaMapDetectionModel.from_pretrained("path/to/llava-v1.5-7b")

# 2. Prepare inputs
text_ids = tokenizer(prompt)  # (B, text_len)
text_embeds = model.get_model().embed_tokens(text_ids)

scene_tokens = qformer(images)  # (B, 512, 4096)

# 3. Forward with map detection
outputs = model.forward_with_map(
    text_embeds=text_embeds,
    scene_tokens=scene_tokens,
    return_map_features=True
)

# 4. Get features
instance_features = outputs['instance_features']  # (B, 50, 4096)
point_features = outputs['point_features']        # (B, 50, 20, 4096)

# 5. Detection heads (next step)
# pred_classes = cls_head(instance_features)
# pred_points = reg_head(point_features)
    """)


def test_position_extraction():
    """
    Test that position extraction works correctly.
    """
    print("\n" + "="*80)
    print("TEST: Position Extraction Correctness")
    print("="*80)
    
    batch_size = 1
    text_len = 10
    scene_len = 20
    num_instances = 3  # Simplified for testing
    num_points = 5
    hidden_size = 32
    
    # Create mock sequence
    prefix_len = text_len + scene_len
    queries_per_inst = 1 + num_points
    total_queries = num_instances * queries_per_inst
    total_len = prefix_len + total_queries
    
    # Create identifiable pattern
    sequence = torch.zeros(batch_size, total_len, hidden_size)
    
    # Mark instance queries with unique values
    for i in range(num_instances):
        inst_pos = prefix_len + i * queries_per_inst
        sequence[:, inst_pos, :] = 100 + i  # Inst0=100, Inst1=101, Inst2=102
        
        # Mark point queries
        for j in range(num_points):
            point_pos = inst_pos + 1 + j
            sequence[:, point_pos, :] = 200 + i * 10 + j  # Inst0_P0=200, Inst0_P1=201, etc.
    
    # Extract features
    instance_feat, point_feat = MapQueryExtractor.extract_features(
        llm_output=sequence,
        text_len=text_len,
        scene_len=scene_len,
        num_instances=num_instances,
        num_points=num_points
    )
    
    print(f"Instance features shape: {instance_feat.shape}")
    print(f"Point features shape: {point_feat.shape}")
    
    # Verify extraction
    print("\nVerifying extraction...")
    for i in range(num_instances):
        inst_val = instance_feat[0, i, 0].item()
        expected_inst = 100 + i
        print(f"  Instance {i}: got {inst_val:.0f}, expected {expected_inst} - {'✓' if abs(inst_val - expected_inst) < 0.01 else '✗'}")
        
        for j in range(num_points):
            point_val = point_feat[0, i, j, 0].item()
            expected_point = 200 + i * 10 + j
            if j == 0:  # Only print first point for brevity
                print(f"    Point {j}: got {point_val:.0f}, expected {expected_point} - {'✓' if abs(point_val - expected_point) < 0.01 else '✗'}")
    
    print("\n✅ Position extraction test passed!")


if __name__ == "__main__":
    # Run complete example
    example_complete_flow()
    
    # Show model usage
    example_with_real_model()
    
    # Test position extraction
    test_position_extraction()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)

