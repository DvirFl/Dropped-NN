import os
import pandas as pd
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Load CSV data
CSV_FILE = 'historical_data.csv' # Replace with your actual CSV file name
df = pd.read_csv(CSV_FILE)

# Extract inputs (measurement_0 to measurement_47)
X = torch.tensor(df[[f'measurement_{i}' for i in range(48)]].values, dtype=torch.float32)
pred_target = torch.tensor(df['pred'].values, dtype=torch.float32)

# Load the known anchor (LastLayer)
last_layer_state = torch.load('pieces/piece_85.pth', map_location='cpu')

# Separate files into candidate pools based on shapes
pool_inp = []
pool_out = []

for i in range(97):
    if i == 85: 
        continue # Skip the exit anchor
    filename = f"pieces/piece_{i}.pth"
    state = torch.load(filename, map_location='cpu')
    
    if state['weight'].shape == torch.Size([96, 48]):
        pool_inp.append((filename, state))
    elif state['weight'].shape == torch.Size([48, 96]):
        pool_out.append((filename, state))

print(f"Loaded {len(pool_inp)} input projections and {len(pool_out)} output projections.")

# --- STEP 1: Pairing using Diagonal Dominance Ratio ---
cost_matrix = np.zeros((48, 48))

for i, (inp_name, inp_state) in enumerate(pool_inp):
    W_in = inp_state['weight']
    for j, (out_name, out_state) in enumerate(pool_out):
        W_out = out_state['weight']
        
        # Calculate matrix product
        prod = torch.matmul(W_out, W_in)
        trace = torch.trace(prod).abs().item()
        frob_norm = torch.linalg.matrix_norm(prod, ord='fro').item()
        
        # Maximize the ratio, which means minimizing the negative ratio
        cost_matrix[i, j] = - (trace / frob_norm)

# Solve optimal bipartite matching (The Hungarian Algorithm)
row_idx, col_idx = linear_sum_assignment(cost_matrix)

paired_blocks = []
for r, c in zip(row_idx, col_idx):
    paired_blocks.append({
        'inp_file': pool_inp[r][0],
        'out_file': pool_out[c][0],
        'inp_state': pool_inp[r][1],
        'out_state': pool_out[c][1],
    })

print("Successfully matched all 48 pairs.")

# --- STEP 2: Ordering via Delta-Norm Heuristic ---
# Blocks with smaller residual contributions tend to belong earlier in the network.
X_subset = X[:500] # Use a subset for speed

for block in paired_blocks:
    W_in = block['inp_state']['weight']
    b_in = block['inp_state'].get('bias', None)
    W_out = block['out_state']['weight']
    b_out = block['out_state'].get('bias', None)
    
    # Calculate Block(x) perturbation: out(ReLU(inp(x)))
    z = torch.nn.functional.linear(X_subset, W_in, b_in)
    z = torch.nn.functional.relu(z)
    perturbation = torch.nn.functional.linear(z, W_out, b_out)
    
    # Measure the average L2 norm of the signal change
    delta_norm = torch.linalg.vector_norm(perturbation, dim=1).mean().item()
    block['delta_norm'] = delta_norm

# Sort by ascending delta norm 
ordered_blocks = sorted(paired_blocks, key=lambda x: x['delta_norm'])


# Also show the original detailed output (commented out)
# for k, block in enumerate(ordered_blocks):
#     print(f"Block {k+1:02d}: Inp -> {block['inp_file']}, Out -> {block['out_file']}")
    
def compute_total_mse(blocks, X_subset, target_subset, last_layer_state):
    """Simulates the forward pass through all blocks and computes MSE."""
    x = X_subset.clone()
    
    # Pass through all blocks in current order
    for block in blocks:
        residual = x
        W_in = block['inp_state']['weight']
        b_in = block['inp_state'].get('bias', None)
        W_out = block['out_state']['weight']
        b_out = block['out_state'].get('bias', None)
        
        z = torch.nn.functional.linear(x, W_in, b_in)
        z = torch.nn.functional.relu(z)
        x = torch.nn.functional.linear(z, W_out, b_out)
        x = x + residual
        
    # Final layer
    W_last = last_layer_state['weight']
    b_last = last_layer_state.get('bias', None)
    predictions = torch.nn.functional.linear(x, W_last, b_last).squeeze()
    
    # Calculate Mean Squared Error
    mse = torch.mean((predictions - target_subset) ** 2).item()
    return mse

# Use a reasonably large subset of 2,000 rows for accuracy and speed
subset_size = 2000
X_test = X[:subset_size]
y_test = pred_target[:subset_size]

# Calculate initial error
best_mse = compute_total_mse(ordered_blocks, X_test, y_test, last_layer_state)
print(f"\nInitial MSE with heuristic ordering: {best_mse:.6f}")

# --- Step 3: Adjacent Swap Refinement ---
improved = True
pass_num = 1

while improved:
    improved = False
    print(f"Starting refinement pass {pass_num}...")
    
    for i in range(len(ordered_blocks) - 1):
        # Create a copy of the list and swap adjacent blocks
        candidate_blocks = list(ordered_blocks)
        candidate_blocks[i], candidate_blocks[i+1] = candidate_blocks[i+1], candidate_blocks[i]
        
        # Test the swap
        candidate_mse = compute_total_mse(candidate_blocks, X_test, y_test, last_layer_state)
        
        # If the swap lowers the error, we keep it!
        if candidate_mse < best_mse:
            best_mse = candidate_mse
            ordered_blocks = candidate_blocks
            improved = True
            print(f"  -> Swapped block {i} and {i+1}. New MSE: {best_mse:.6f}")
            
    pass_num += 1

print("\n--- Final Perfect Layer Sequence ---")
print("\n--- Identified Layer Sequence ---")
# Extract numbers from file names
layer_numbers = []
for k, block in enumerate(ordered_blocks):
    # Extract number from inp_file (e.g., "pieces/piece_85.pth" -> "85")
    inp_num = block['inp_file'].split('_')[-1].split('.')[0]
    out_num = block['out_file'].split('_')[-1].split('.')[0]
    layer_numbers.extend([inp_num, out_num])

# Create comma-separated string of numbers
numbers_string = ','.join(layer_numbers)
print(f"Layer sequence numbers: {numbers_string}")

print(f"\nFinal MSE achieved: {best_mse:.10f}")