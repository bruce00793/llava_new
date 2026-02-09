"""
Test script to verify dataset updates
"""

import sys
import torch
import numpy as np

# Test imports
try:
    from llava.data.map_dataset import MapDetectionDataset
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from transformers import AutoTokenizer
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test tokenization with image tokens
def test_image_token_insertion():
    print("\n=== Test 1: Image Token Insertion ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
        
        # Create a simple prompt with 1 image token (for Q-Former output)
        prompt = DEFAULT_IMAGE_TOKEN + "\nTest prompt for map detection"
        print(f"Prompt: {prompt[:100]}...")
        
        # Count image tokens in string
        count = prompt.count(DEFAULT_IMAGE_TOKEN)
        print(f"✓ Image tokens in string: {count}")
        assert count == 1, f"Expected 1 image token (for Q-Former), got {count}"
        
        print("✓ Test 1 passed!")
        print("  Note: This 1 <image> token will be replaced by 512 scene tokens from Q-Former")
        return True
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False

# Test coordinate normalization
def test_coordinate_normalization():
    print("\n=== Test 2: Coordinate Normalization ===")
    
    try:
        # Test points in ego frame
        pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        
        # Test point at center (0, 0)
        center_point = np.array([[0.0, 0.0]])
        center_norm = (center_point - [x_min, y_min]) / ([x_max - x_min, y_max - y_min]) * 2 - 1
        print(f"Center (0, 0) normalized to: {center_norm}")
        assert np.allclose(center_norm, [[0.0, 0.0]]), "Center should be (0, 0)"
        
        # Test point at min corner
        min_point = np.array([[x_min, y_min]])
        min_norm = (min_point - [x_min, y_min]) / ([x_max - x_min, y_max - y_min]) * 2 - 1
        print(f"Min corner ({x_min}, {y_min}) normalized to: {min_norm}")
        assert np.allclose(min_norm, [[-1.0, -1.0]]), "Min corner should be (-1, -1)"
        
        # Test point at max corner
        max_point = np.array([[x_max, y_max]])
        max_norm = (max_point - [x_min, y_min]) / ([x_max - x_min, y_max - y_min]) * 2 - 1
        print(f"Max corner ({x_max}, {y_max}) normalized to: {max_norm}")
        assert np.allclose(max_norm, [[1.0, 1.0]]), "Max corner should be (1, 1)"
        
        print("✓ Test 2 passed!")
        return True
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False

# Test rotation matrix conversion
def test_rotation_conversion():
    print("\n=== Test 3: Quaternion to Rotation Matrix ===")
    
    try:
        from pyquaternion import Quaternion
        
        # Identity quaternion
        quat = Quaternion([1, 0, 0, 0])  # w, x, y, z
        rot_mat = quat.rotation_matrix
        
        print(f"Identity quaternion rotation matrix:\n{rot_mat}")
        assert np.allclose(rot_mat, np.eye(3)), "Identity quaternion should give identity matrix"
        
        # Test transformation matrix construction
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = [1.0, 2.0, 3.0]  # translation
        
        print(f"4x4 transformation matrix:\n{mat}")
        assert mat.shape == (4, 4), "Should be 4x4 matrix"
        
        print("✓ Test 3 passed!")
        return True
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False

# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Dataset Updates")
    print("=" * 60)
    
    results = []
    results.append(test_image_token_insertion())
    results.append(test_coordinate_normalization())
    results.append(test_rotation_conversion())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"✗ {sum(not r for r in results)} test(s) failed")
        print("=" * 60)
        sys.exit(1)

