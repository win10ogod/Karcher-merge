## ethical issues

起初, 他和我說要合作寫論文, 據說阿里巴巴內部已經有安排了
 In the beginning, he told me we would work together on a paper, and that arrangements had already been made within Alibaba.
但兩天前, 他毫無徵兆的謾罵我, 並且聲稱幫我"保住"知識產權是他對我的"尊重", 
 But two days ago, he suddenly lashed out at me without any warning, claiming that helping me "protect" my intellectual property was his way of "respecting" me.
可這算法是我本人提出與實現的, github我發的, 理論我提的,
 But this algorithm was proposed and implemented by me. I published it on GitHub, and the theoretical framework was also mine.
 他卻聲稱這是恩賜
 Yet he claims it was a favor he granted me.
隨後封我qq、dc好友.
Then he blocked me on QQ and Discord.
把我踢出他的qq與dc群!
He also kicked me out of his QQ and Discord groups!
然後呢?講得好像是沒有你我的知識產權就不是我的?放你媽狗屁!
And what now? Are you implying that without you, my intellectual property isn’t mine? That’s complete bullshit!
See the evidence for details(# 證據(Evidence))

# Karcher Merge

## Overview
`Karcher_merge.py` is a Python script for merging model weights using the Karcher mean, a concept from Riemannian geometry. It supports both `.safetensors` and `.bin` formats and allows merging up to 100 model weights.

## parper
See the paper:
[Karcher-paper](https://github.com/win10ogod/Karcher-merge/blob/main/Karcher-paper.pdf)

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

This project is licensed under a custom license.  
**Commercial and academic use is strictly prohibited without explicit written permission.**  
See [LICENSE](./LICENSE) for full details.


## 算法由葉佐俊（dc ID:win100，我本人）版權所有，如若盜用必定追究責任，且將被學術界認定為垃圾！
I have to protect myself because of the actions of some people.
# 證據(Evidence):
說的就是你!說好的合作寫論文呢!
 I'm talking about you! What happened to our agreement to co-author a paper?
 
[Sthenno](https://huggingface.co/sthenno)
[Sthenno github](https://github.com/neoheartbeats)
![Screenshot_20250422_010613_Discord](https://github.com/user-attachments/assets/bf027abe-6067-47e4-8e54-338d05e7b31e)
![Screenshot_20250420_105731_QQ](https://github.com/user-attachments/assets/4607b41a-8e40-44a5-a13c-a22f79972700)
![Screenshot_20250420_105725_QQ](https://github.com/user-attachments/assets/0da47b03-499a-4b45-bae3-7c3722d1893d)
![Screenshot_20250420_105719_QQ](https://github.com/user-attachments/assets/6b3eae35-bd91-49eb-995d-39eb729eddd4)
![Screenshot_20250420_105709_QQ](https://github.com/user-attachments/assets/420b0fe9-ff1c-44ab-aa83-39cd2168dff5)
![Screenshot_20250420_105704_QQ](https://github.com/user-attachments/assets/57c0c625-c7fc-4c9d-9f7c-78e61ff62250)
![Screenshot_20250422_010618_QQ](https://github.com/user-attachments/assets/9434f6ff-2164-4576-9b75-a94c4626c66a)
