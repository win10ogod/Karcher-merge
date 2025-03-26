#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Karcher_merge.py

This script is used to merge 1-5 model weights (supports .safetensors and .bin formats).
Implements weight fusion based on the Riemannian (Karcher) mean concept.

Usage example:
  python Karcher_merge.py --models modelA.safetensors modelB.bin --alphas 0.4 0.6 --output merged.safetensors --device cuda --karcher-iter 10 --karcher-tol 1e-5
"""

import argparse
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

###############################################################################
# General utilities: Reading, saving, and tensor alignment (resizing)
###############################################################################

class BinDataHandler:
    """Simple wrapper for .bin files providing get_tensor() and keys()."""
    def __init__(self, data):
        self.data = data

    def keys(self):
        return list(self.data.keys())

    def get_tensor(self, key):
        return self.data[key]

def read_tensors(file_path, device="cpu"):
    """
    Reads .safetensors or .bin files based on extension,
    returns (handler, key_set).
    """
    if file_path.endswith(".safetensors"):
        f = safe_open(file_path, framework="pt", device=device)
        return f, set(f.keys())
    elif file_path.endswith(".bin"):
        data = torch.load(file_path, map_location=torch.device(device))
        f = BinDataHandler(data)
        return f, set(data.keys())
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_safetensors(tensor_dict, output_path):
    """Saves merged weights as safetensors file."""
    metadata = {"format": "pt"}
    save_file(tensor_dict, output_path, metadata=metadata)

def resize_tensors(t1, t2):
    """
    Aligns tensors using zero-pad if last two dimensions don't match
    (for demonstration, adjust as needed).
    """
    if len(t1.shape) < 2 or len(t2.shape) < 2:
        return t1, t2

    h1, w1 = t1.shape[-2], t1.shape[-1]
    h2, w2 = t2.shape[-2], t2.shape[-1]

    if w1 < w2:
        pad_w = w2 - w1
        t1 = F.pad(t1, (0, pad_w, 0, 0))
    elif w2 < w1:
        pad_w = w1 - w2
        t2 = F.pad(t2, (0, pad_w, 0, 0))

    if h1 < h2:
        pad_h = h2 - h1
        t1 = F.pad(t1, (0, 0, 0, pad_h))
    elif h2 < h1:
        pad_h = h1 - h2
        t2 = F.pad(t2, (0, 0, 0, pad_h))

    return t1, t2

def copy_extra_files_if_needed(args):
    """
    If --copy-extra-files is specified, copies non-weight files
    (not ending with .bin, .safetensors or .pt) from the first model's
    directory to the output directory.
    """
    if not args.copy_extra_files:
        return
    first_model = args.models[0]
    first_dir = os.path.dirname(first_model) if os.path.isfile(first_model) else first_model
    out_dir = os.path.dirname(args.output) or "."
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(first_dir) and os.path.isdir(out_dir):
        for fname in os.listdir(first_dir):
            fpath = os.path.join(first_dir, fname)
            if (os.path.isfile(fpath) and not fname.startswith(".") and
                not fname.endswith(".bin") and not fname.endswith(".safetensors") and not fname.endswith(".pt")):
                tgt = os.path.join(out_dir, fname)
                print(f"[Info] Copying extra file: {fpath} -> {tgt}")
                shutil.copyfile(fpath, tgt)

###############################################################################
# Algorithm: Karcher Mean-based Weight Fusion
###############################################################################

def karcher_merge_tensors(tensors, alphas, max_iter=10, tol=1e-5):
    """
    Fixed Karcher mean merging function
    Main fixes: Ensures trigonometric functions receive tensor inputs
    """
    if len(tensors) == 1:
        return tensors[0]
    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))
            
    # Select non-zero weight vectors
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])
    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: Normalized weighted arithmetic mean
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)
    
    # Iterative Karcher mean computation
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Flatten tensor for dot product calculation
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            theta_val = theta.item()
            if theta_val < tol:
                continue
            else:
                # Ensure tensor operations
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)
        
        # Convert norm_T to tensor
        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break
            
        # Use tensor for trigonometric calculations
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)
        
        # Ensure u is a unit vector
        u_norm = torch.linalg.norm(u.float())
        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)
            
    # Global scale: Weighted sum of original tensor norms (including zero vectors)
    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n
    return s * u

###############################################################################
# Main Program
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Model merging script (Karcher mean fusion algorithm)"
    )
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model files (.safetensors or .bin), supports 2-100 models")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Global weights for each model (equal weights if not specified)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Computing device (cpu or cuda)")
    parser.add_argument("--output", type=str, default="merged.safetensors",
                        help="Output filename (safetensors)")
    parser.add_argument("--copy-extra-files", action="store_true",
                        help="Copy non-weight files from first model's directory to output")
    parser.add_argument("--karcher-iter", type=int, default=10,
                        help="Maximum iterations for Karcher mean algorithm")
    parser.add_argument("--karcher-tol", type=float, default=1e-5,
                        help="Convergence tolerance for Karcher mean algorithm")
    args = parser.parse_args()

    N = len(args.models)
    if N < 2 or N > 100:
        raise ValueError("This script only supports merging 2-100 models.")

    handlers = []
    all_keys = []
    for mpath in args.models:
        h, keys = read_tensors(mpath, device=args.device)
        handlers.append((h, keys))
        all_keys.append(keys)
    common_keys = set.intersection(*all_keys)

    def normalize_alphas(arr):
        s = sum(arr)
        if abs(s) < 1e-9:
            raise ValueError("alphas sum to 0")
        return [x / s for x in arr]

    if args.alphas is None:
        alphas = [1.0 / N] * N
    else:
        if len(args.alphas) != N:
            raise ValueError("Number of --alphas must match number of models")
        alphas = normalize_alphas(args.alphas)

    merge_func = lambda tensors, alphas: karcher_merge_tensors(tensors, alphas,
                                                            max_iter=args.karcher_iter,
                                                            tol=args.karcher_tol)

    merged_tensors = {}
    for key in sorted(common_keys):
        weight_list = []
        for (handler, _) in handlers:
            w = handler.get_tensor(key)
            weight_list.append(w)
        for i in range(1, N):
            w0, wi = weight_list[0], weight_list[i]
            w0r, wir = resize_tensors(w0, wi)
            weight_list[0] = w0r
            weight_list[i] = wir
        shape0 = weight_list[0].shape
        if not all(w.shape == shape0 for w in weight_list):
            print(f"[Warning] key={key} weight shapes inconsistent, skipping layer")
            continue
        d0 = weight_list[0].device
        dt0 = weight_list[0].dtype
        for i in range(N):
            if weight_list[i].device != d0:
                weight_list[i] = weight_list[i].to(d0)
            if weight_list[i].dtype != dt0:
                weight_list[i] = weight_list[i].to(dt0)
        merged = merge_func(weight_list, alphas)
        merged_tensors[key] = merged

    print(f"[Info] Merged {len(merged_tensors)} layers")
    save_safetensors(merged_tensors, args.output)
    print(f"[Info] Output file: {args.output}")
    copy_extra_files_if_needed(args)

if __name__ == "__main__":
    main()