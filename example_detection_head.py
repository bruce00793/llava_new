"""
Example: Using MapDetectionHead

This demonstrates how to use the detection head in the complete pipeline.

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
from llava.model.map_detection_head import MapDetectionHead, build_detection_head


def example_basic_usage():
    """Basic usage of MapDetectionHead."""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Create detection head
    detection_head = MapDetectionHead(
        hidden_size=4096,
        num_classes=3,
        intermediate_dim=1024,
        bottleneck_dim=256,
        dropout=0.1
    )
    
    print(f"\n✓ Created detection head with {detection_head.get_num_params():,} parameters")
    
    # Prepare inputs (from LLM output)
    batch_size = 4
    instance_features = torch.randn(batch_size, 50, 4096)
    point_features = torch.randn(batch_size, 50, 20, 4096)
    
    print(f"\nInputs:")
    print(f"  instance_features: {instance_features.shape}")
    print(f"  point_features: {point_features.shape}")
    
    # Forward
    pred_classes, pred_points = detection_head(instance_features, point_features)
    
    print(f"\nOutputs:")
    print(f"  pred_classes: {pred_classes.shape}  # Logits for 3 classes")
    print(f"  pred_points: {pred_points.shape}   # Coordinates in [-1, 1]")
    
    # Apply softmax for probabilities
    pred_probs = torch.softmax(pred_classes, dim=-1)
    print(f"\nPredicted probabilities shape: {pred_probs.shape}")
    print(f"Points value range: [{pred_points.min():.3f}, {pred_points.max():.3f}]")


def example_with_llm_output():
    """Example showing integration with LLM output."""
    print("\n" + "="*80)
    print("Example 2: Integration with LLM")
    print("="*80)
    
    # Simulate LLM forward
    from llava.model.language_model.llava_map import LlavaMapDetectionModel
    from llava.model.map_detection_head import MapDetectionHead
    
    print("\nComplete pipeline:")
    print("  1. Images → Q-Former → scene_tokens")
    print("  2. Text + scene_tokens + map_queries → LLM")
    print("  3. LLM output → extract features")
    print("  4. Features → detection_head → predictions")
    
    # Step 3 & 4 (assuming we have LLM outputs)
    batch_size = 2
    
    # Simulated LLM outputs
    instance_features = torch.randn(batch_size, 50, 4096)
    point_features = torch.randn(batch_size, 50, 20, 4096)
    
    print(f"\n[Step 3] Extracted features from LLM:")
    print(f"  instance_features: {instance_features.shape}")
    print(f"  point_features: {point_features.shape}")
    
    # Detection head
    detection_head = MapDetectionHead()
    pred_classes, pred_points = detection_head(instance_features, point_features)
    
    print(f"\n[Step 4] Detection head predictions:")
    print(f"  pred_classes: {pred_classes.shape}")
    print(f"  pred_points: {pred_points.shape}")
    
    # Post-process
    pred_labels = pred_classes.argmax(dim=-1)  # [B, 50]
    pred_scores = torch.softmax(pred_classes, dim=-1).max(dim=-1)[0]  # [B, 50]
    
    print(f"\n[Post-process]:")
    print(f"  Predicted labels: {pred_labels.shape}")
    print(f"  Confidence scores: {pred_scores.shape}")
    print(f"  Sample scores: {pred_scores[0, :5]}")


def example_inference():
    """Example showing inference (without training)."""
    print("\n" + "="*80)
    print("Example 3: Inference Mode")
    print("="*80)
    
    # Build model
    detection_head = build_detection_head({
        'hidden_size': 4096,
        'num_classes': 3,
        'dropout': 0.0  # No dropout in inference
    })
    
    detection_head.eval()  # Set to eval mode
    
    print("\n✓ Model in eval mode (dropout disabled)")
    
    with torch.no_grad():
        # Prepare inputs
        instance_features = torch.randn(1, 50, 4096)
        point_features = torch.randn(1, 50, 20, 4096)
        
        # Forward
        pred_classes, pred_points = detection_head(instance_features, point_features)
        
        # Get predictions
        pred_probs = torch.softmax(pred_classes, dim=-1)[0]  # [50, 3]
        pred_labels = pred_probs.argmax(dim=-1)  # [50]
        pred_scores = pred_probs.max(dim=-1)[0]  # [50]
        pred_coords = pred_points[0]  # [50, 20, 2]
        
        print(f"\nPredictions for 1 sample:")
        print(f"  Classes shape: {pred_labels.shape}")
        print(f"  Scores shape: {pred_scores.shape}")
        print(f"  Coordinates shape: {pred_coords.shape}")
        
        # Filter by confidence
        threshold = 0.5
        valid_mask = pred_scores > threshold
        num_valid = valid_mask.sum().item()
        
        print(f"\nFiltering (confidence > {threshold}):")
        print(f"  Valid predictions: {num_valid} / 50")
        
        if num_valid > 0:
            valid_labels = pred_labels[valid_mask]
            valid_scores = pred_scores[valid_mask]
            valid_coords = pred_coords[valid_mask]
            
            print(f"  Valid labels: {valid_labels[:5]}")
            print(f"  Valid scores: {valid_scores[:5]}")
            print(f"  Valid coords shape: {valid_coords.shape}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "="*80)
    print("Example 4: Custom Configuration")
    print("="*80)
    
    # Lightweight config
    lightweight_config = {
        'hidden_size': 4096,
        'num_classes': 3,
        'intermediate_dim': 512,   # Smaller
        'bottleneck_dim': 128,     # Smaller
        'dropout': 0.1
    }
    
    # Heavy config
    heavy_config = {
        'hidden_size': 4096,
        'num_classes': 3,
        'intermediate_dim': 2048,  # Larger
        'bottleneck_dim': 512,     # Larger
        'dropout': 0.1
    }
    
    # Build models
    lightweight_head = build_detection_head(lightweight_config)
    heavy_head = build_detection_head(heavy_config)
    standard_head = build_detection_head()  # Default
    
    print("\nModel comparison:")
    print(f"  Lightweight: {lightweight_head.get_num_params():,} params")
    print(f"  Standard:    {standard_head.get_num_params():,} params")
    print(f"  Heavy:       {heavy_head.get_num_params():,} params")
    
    # Test all
    instance_features = torch.randn(1, 50, 4096)
    point_features = torch.randn(1, 50, 20, 4096)
    
    for name, head in [('Lightweight', lightweight_head), 
                       ('Standard', standard_head), 
                       ('Heavy', heavy_head)]:
        pred_cls, pred_pts = head(instance_features, point_features)
        print(f"\n{name} model:")
        print(f"  Output shapes: {pred_cls.shape}, {pred_pts.shape}")


def example_parameter_analysis():
    """Analyze model parameters."""
    print("\n" + "="*80)
    print("Example 5: Parameter Analysis")
    print("="*80)
    
    detection_head = MapDetectionHead()
    
    print("\nLayer-by-layer parameters:")
    print("\nClassification Head:")
    for i, (name, param) in enumerate(detection_head.cls_head.named_parameters()):
        print(f"  {name:30s} {str(param.shape):20s} {param.numel():>10,} params")
    
    print("\nRegression Head:")
    for i, (name, param) in enumerate(detection_head.reg_head.named_parameters()):
        print(f"  {name:30s} {str(param.shape):20s} {param.numel():>10,} params")
    
    cls_params = sum(p.numel() for p in detection_head.cls_head.parameters())
    reg_params = sum(p.numel() for p in detection_head.reg_head.parameters())
    
    print(f"\nSummary:")
    print(f"  Classification head: {cls_params:,} params")
    print(f"  Regression head:     {reg_params:,} params")
    print(f"  Total:               {detection_head.get_num_params():,} params")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_with_llm_output()
    example_inference()
    example_custom_config()
    example_parameter_analysis()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)

