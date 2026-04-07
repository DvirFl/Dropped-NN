import torch
import os
from pathlib import Path

pieces_dir = 'pieces'

print("="*80)
print("INSPECTING LAYER SHAPES IN 'pieces' FOLDER")
print("="*80)

# List all .pth files in pieces directory
pth_files = sorted([f for f in os.listdir(pieces_dir) if f.endswith('.pth')])

print(f"\nFound {len(pth_files)} .pth files\n")

for pth_file in pth_files:
    filepath = os.path.join(pieces_dir, pth_file)
    
    try:
        state_dict = torch.load(filepath, map_location='cpu')
        
        print(f"{pth_file}:")
        print(f"  Total parameters: {len(state_dict)}")
        
        for key, tensor in state_dict.items():
            print(f"    {key}: {tensor.shape}")
        
        # Calculate total parameters
        total_params = sum(t.numel() for t in state_dict.values())
        print(f"  Total elements: {total_params:,}")
        print()
        
    except Exception as e:
        print(f"{pth_file}: ERROR - {e}\n")

print("="*80)
print("EXPECTED SHAPES FOR [48, 96] CONFIGURATION")
print("="*80)
print("Each Block should have:")
print("  inp.weight: torch.Size([96, 48])  # hidden_dim x in_dim")
print("  out.weight: torch.Size([48, 96])  # in_dim x hidden_dim")
print("Last layer should have:")
print("  layer.weight: torch.Size([1, 48])  # out_dim x in_dim")
print()

print("="*80)
print("SUMMARY")
print("="*80)

# Create a summary table
summary_data = []
for pth_file in pth_files:
    filepath = os.path.join(pieces_dir, pth_file)
    try:
        state_dict = torch.load(filepath, map_location='cpu')
        total_params = sum(t.numel() for t in state_dict.values())
        shapes = [str(t.shape) for t in state_dict.values()]
        summary_data.append((pth_file, len(state_dict), total_params, shapes))
    except:
        pass

print(f"\n{'File':<15} {'Params':<8} {'Shapes'}")
print("-" * 80)
for fname, num_params, total_params, shapes in summary_data:
    shapes_str = ", ".join(shapes)
    print(f"{fname:<15} {num_params:<8} {shapes_str}")

print("\n" + "="*80)
