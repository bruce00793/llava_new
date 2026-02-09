"""
Test script to verify the new English prompt
"""

from llava.constants import DEFAULT_IMAGE_TOKEN

# The default prompt used in map_dataset.py
prompt = (
    "You are driving a car with 6 surrounding cameras viewing the environment. "
    "Please identify three types of map elements visible in the images:\n"
    "1. Lane dividers - dashed or solid lines separating traffic lanes\n"
    "2. Road boundaries - curbs, barriers, or edges defining road limits\n"
    "3. Pedestrian crossings - zebra crossing patterns for pedestrians\n"
    "Detect all visible map elements in the BEV coordinate space."
)

# With <image> token
prompt_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

print("="*80)
print("NEW ENGLISH PROMPT (without <image> token):")
print("="*80)
print(prompt)
print()

print("="*80)
print("COMPLETE PROMPT (with <image> token):")
print("="*80)
print(prompt_with_image)
print()

print("="*80)
print("COMPARISON:")
print("="*80)
print(f"Old (Chinese): 请帮我识别图中的车道线、道路边界、人行横道三类物体")
print(f"New (English): {prompt}")
print()

print("="*80)
print("KEY IMPROVEMENTS:")
print("="*80)
print("✅ Changed from Chinese to English")
print("✅ Added detailed description of each class")
print("✅ Mentioned 6 surrounding cameras")
print("✅ Mentioned BEV coordinate space")
print("✅ More professional and informative")
print()

# Estimate token length
print("="*80)
print("ESTIMATED LENGTH:")
print("="*80)
print(f"Character count: {len(prompt)} chars")
print(f"Word count: {len(prompt.split())} words")
print(f"Estimated tokens: ~{len(prompt.split()) * 1.3:.0f} tokens (rough estimate)")

