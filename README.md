## 原創性聲明(Statement of Originality)
本人聲明，Karcher-merge包括idea及code均為本人（在o3-mini的輔助下）的個人智力成果。

Karcher-merge的目的是更好的模型合併，其核心思想是將神經網絡權重視為黎曼流形上的點，利用Karcher平均（也稱為Fréchet平均或幾何平均）來計算它們的內在平均值，同時考慮權重張量的方向和幅度。該方法將每個權重張量分解為方向分量（位於單位球面上）和幅度分量，通過叠代算法計算方向的Karcher平均，並使用加權算術平均計算合並幅度。這種幾何方法能有效處理正交或近正交權重，保持權重張量的方向信息和適當縮放，最後可以達到超越model stock等傳統模型合併方法的效果。本人之所以再次公開此聲明是因為sthenno（qq號為：1829321520）企圖借合作之名竊取本人成果來達成與阿里巴巴（qwen團隊？）的論文合作，在騙取我的代碼和論文後，sthenno(qq號為：1829321520)將我在各種社交媒體上拉黑（詳情見附件1，2），導致本人無法正常聲索本人智力成果的所有權。

本人從某些途徑(Sthenno本人)獲知，該論文已通過了阿里的內部流程。因此，本人迫不得已把已有代碼和論文主要思路公開，望周知。 

希望阿里巴巴相關工作人員看見本文後可盡快聯系本人（jmes100010@gmail.com，qq:3695314497）。

I hereby declare that Karcher-merge, including both the idea and code, is my personal intellectual achievement (with the assistance of o3-mini). The purpose of Karcher-merge is to achieve better model merging, with its core concept being the treatment of neural network weights as points on a Riemannian manifold, utilizing the Karcher mean (also known as Fréchet mean or geometric mean) to compute their intrinsic average while considering both the direction and magnitude of weight tensors. This method decomposes each weight tensor into a directional component (lying on the unit sphere) and a magnitude component, computing the Karcher mean of directions through an iterative algorithm and using weighted arithmetic mean for merged magnitude. This geometric approach effectively handles orthogonal or nearly orthogonal weights, preserves directional information in weight tensors with appropriate scaling, and ultimately achieves results that surpass traditional model averaging methods such as model stock. I am making this public declaration again because sthenno (QQ number: 1829321520) attempted to steal my work under the pretense of collaboration to establish a paper partnership with Alibaba (Qwen team?). After fraudulently obtaining my code and paper, sthenno (QQ number: 1829321520) blocked me on various social media platforms (see attachments 1 and 2 for details), preventing me from properly claiming ownership of my intellectual achievement.
I have learned through certain channels that this paper has passed through Alibaba's internal process. 
Therefore, I am compelled to make the existing code and the main ideas of the paper public. 
Please take note. I hope that relevant staff members at Alibaba who see this article can contact me as soon as possible (via email, phone).

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
 
明明是Sthenno意圖竊盜成果,如今事情即將鬧大了,他卻慌了.
It was obviously Sthenno's intention to steal the results.
Now things are about to get serious,
But he panicked.
![螢幕擷取畫面 2025-04-22 215242](https://github.com/user-attachments/assets/16713644-095c-41de-a7cf-8156ae9d0872)

## Sthenno related content

[Sthenno](https://huggingface.co/sthenno)
[Sthenno github](https://github.com/neoheartbeats)
![Screenshot_20250422_010613_Discord](https://github.com/user-attachments/assets/bf027abe-6067-47e4-8e54-338d05e7b31e)
![Screenshot_20250420_105731_QQ](https://github.com/user-attachments/assets/4607b41a-8e40-44a5-a13c-a22f79972700)
![Screenshot_20250420_105725_QQ](https://github.com/user-attachments/assets/0da47b03-499a-4b45-bae3-7c3722d1893d)
![Screenshot_20250420_105719_QQ](https://github.com/user-attachments/assets/6b3eae35-bd91-49eb-995d-39eb729eddd4)
![Screenshot_20250420_105709_QQ](https://github.com/user-attachments/assets/420b0fe9-ff1c-44ab-aa83-39cd2168dff5)
![Screenshot_20250420_105704_QQ](https://github.com/user-attachments/assets/57c0c625-c7fc-4c9d-9f7c-78e61ff62250)
![Screenshot_20250422_010618_QQ](https://github.com/user-attachments/assets/9434f6ff-2164-4576-9b75-a94c4626c66a)
