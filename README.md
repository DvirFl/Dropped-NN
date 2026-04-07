# Summary of Jane Street Puzzle - I dropped a neural net

## Overview
This project solves a Jane Street puzzle involving reconstructing the correct order of layers in a neural network. The neural network appears to be a residual network with 48 blocks, each consisting of an input projection and an output projection, followed by a final output layer.

You can find the puzzle in https://huggingface.co/spaces/jane-street/droppedaneuralnet

## Files Description

### inspect_pieces.py
This script inspects the 97 PyTorch model pieces stored in the `pieces/` directory (piece_0.pth to piece_96.pth). It analyzes each piece's state dictionary to determine:
- Parameter counts and tensor shapes
- Classification into input projections (96×48 weights), output projections (48×96 weights), or the final layer (1×48 weights)
- Expected shapes for a [48, 96] hidden dimension configuration

The script confirms there are 48 input projections, 48 output projections, and 1 final layer (piece_85.pth).

### hueristic_layer_search.py
This script reconstructs the neural network by solving two key problems:
1. **Pairing**: Matches input and output projections into 48 residual blocks using the Hungarian algorithm, optimizing for diagonal dominance in the matrix product.
2. **Ordering**: Determines the correct sequence of blocks using a delta-norm heuristic (blocks with smaller signal perturbations appear earlier) and refines via adjacent swaps to minimize mean squared error (MSE) against historical data.

The final output provides the layer sequence as a comma-separated list of piece numbers, achieving a very low MSE (indicating correct reconstruction).

## Neural Network Architecture
- **Input**: 48-dimensional measurements
- **Hidden**: 96-dimensional representations
- **Blocks**: 48 residual blocks, each with ReLU activation
- **Output**: 1-dimensional prediction
- **Total Parameters**: ~460,000 (estimated from shapes)

## Solution Approach
1. Load and preprocess historical training data
2. Identify and pair projection layers using the Hongarian Algorithm
3. Order blocks using signal perturbation heuristics
4. Refine ordering through local search optimization
5. Validate against prediction accuracy</content>
<parameter name="filePath">/home/dvir/Downloads/historical_data_and_pieces/summary.md
