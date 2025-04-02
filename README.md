# Karcher Merge

## Overview
`Karcher_merge.py` is a Python script for merging model weights using the Karcher mean, a concept from Riemannian geometry. It supports both `.safetensors` and `.bin` formats and allows merging up to 100 model weights.

## Features
- Supports `.safetensors` and `.bin` model formats
- Uses Karcher mean for weighted averaging of model parameters
- Customizable weight distribution via `--alphas`
- Runs on `cpu` or `cuda`
- Optionally copies extra non-weight files from the first model’s directory

## Installation
Ensure you have Python 3 and install dependencies:
```bash
pip install torch safetensors
```

## Usage
```bash
python Karcher_merge.py --models modelA.safetensors modelB.bin \
    --alphas 0.4 0.6 --output merged.safetensors \
    --device cuda --karcher-iter 10 --karcher-tol 1e-5
```

### Arguments
| Argument | Description |
|----------|-------------|
| `--models` | List of model weight files to merge (2-100 files) |
| `--alphas` | Weight coefficients for merging (default: equal weights) |
| `--device` | Compute device: `cpu` or `cuda` (default: `cpu`) |
| `--output` | Output filename (default: `merged.safetensors`) |
| `--copy-extra-files` | Copy additional non-weight files from first model |
| `--karcher-iter` | Maximum iterations for Karcher mean computation (default: 10) |
| `--karcher-tol` | Convergence tolerance for Karcher mean algorithm (default: `1e-5`) |

## How It Works
The script implements the Karcher mean method to merge model weights iteratively:

1. **Normalize and align tensors**: Ensures tensors have compatible shapes.
2. **Compute Karcher mean**: Uses Riemannian gradient descent to find the mean of tensors in the tangent space.

   Given tensors \( T_1, T_2, \dots, T_n \) and weights \( \alpha_1, \alpha_2, \dots, \alpha_n \), the Karcher mean \( T^* \) minimizes the sum of squared Riemannian distances:
   
   ```math
   T^* = \arg\min_T \sum_{i=1}^{n} \alpha_i d^2(T, T_i)
   ```
   
   where \( d(T, T_i) \) is the Riemannian distance between tensors, given by:
   
   ```math
   d(T, T_i) = \| \log(T^{-1} T_i) \|
   ```
   
   The algorithm iteratively updates \( T^* \) using gradient descent along the manifold:
   
   ```math
   T_{k+1} = \exp_{T_k} \left( \sum_{i=1}^{n} \alpha_i \log_{T_k} (T_i) \right)
   ```
   
3. **Rescale merged tensors**: Applies global scaling based on original tensor norms:
   
   ```math
   s = \sum_{i=1}^{n} \alpha_i \|T_i\|
   ```
   
   The final merged tensor is computed as:
   
   ```math
   T^* = s \cdot U
   ```
   
   where \( U \) is the unit-weighted mean computed in the tangent space.

4. **Save output**: Writes merged weights to a `.safetensors` file.

## Notes
- Ensure models have compatible architectures before merging.
- Large models may require substantial memory.
- Different `--alphas` values will influence how model weights are blended.

## License
This project is open-source under the MIT License.

## 算法由葉佐俊（dc ID:win100）版權所有
