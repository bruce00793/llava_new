#!/usr/bin/env python3
"""
Quick mAP Evaluation Script

Evaluates trained model using the new MapEvaluator module.
Outputs mAP following MapTR protocol.

Usage:
    python eval_map.py --checkpoint path/to/checkpoint.pth
"""

import os
# Force single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset
from llava.model.map_eval import MapEvaluator, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Map Detection Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataroot', type=str,
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--gt-cache', type=str,
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache',
                        help='Path to GT cache')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to evaluate (None = all)')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Score threshold for predictions')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results JSON')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("Map Detection mAP Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # Build model
    print("Loading model...")
    llm_path = '/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5'
    
    model = build_map_detector(
        llm_path=llm_path,
        freeze_llm=True,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle module prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, local_files_only=True)
    
    # Load CLIP image processor from local path
    clip_path = '/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336'
    if os.path.exists(clip_path):
        image_processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
        print(f"✅ CLIP loaded from local: {clip_path}")
    else:
        raise FileNotFoundError(f"CLIP model not found at: {clip_path}. Please ensure local models are available.")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = MapDetectionDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    print(f"✅ Loaded {len(val_dataset)} validation samples")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )
    
    # Create evaluator
    evaluator = MapEvaluator(
        score_threshold=args.score_threshold,
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("Running evaluation...")
    print(f"{'='*60}")
    
    results = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        evaluator=evaluator,
        max_samples=args.max_samples,
        verbose=True,
    )
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {args.save_results}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Overall mAP: {results.get('mAP', 0.0)*100:.2f}%")
    print(f"  mAP@0.5m:    {results.get('mAP@0.5m', 0.0)*100:.2f}%")
    print(f"  mAP@1.0m:    {results.get('mAP@1.0m', 0.0)*100:.2f}%")
    print(f"  mAP@1.5m:    {results.get('mAP@1.5m', 0.0)*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Compare with MapTR reference
    print("📊 MapTR Reference (24 epochs, nuScenes):")
    print("  - MapTR-tiny:  ~50.3% mAP")
    print("  - MapTR-nano:  ~46.3% mAP")
    print()
    
    return results


if __name__ == '__main__':
    main()
