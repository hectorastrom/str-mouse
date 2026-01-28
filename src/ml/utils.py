# @Time    : 2026-01-14 17:49
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : utils.py

# Contains some logic commonly used across scripts for the models (mainly
# inference and checkpoint loading)

import torch as t
import time
import os
import re


CHECKPOINTS_ROOT = "checkpoints"
FINETUNE_DIR = os.path.join(CHECKPOINTS_ROOT, "mouse_finetune")
PRETRAIN_DIR = os.path.join(CHECKPOINTS_ROOT, "emnist_pretrain")


def find_best_checkpoint(ckpt_type: str = "finetune") -> tuple[str, int]:
    """
    Find the best checkpoint folder with the highest class count.
    
    Args:
        ckpt_type: Either "finetune" or "pretrain"
    
    Returns:
        Tuple of (folder_name, num_classes) for the best checkpoint.
        For example: ("best_finetune-63-class", 63)
    
    Raises:
        FileNotFoundError: If no matching checkpoint folders are found.
    """
    if ckpt_type == "finetune":
        ckpt_dir = FINETUNE_DIR
        pattern = re.compile(r"^best_finetune-(\d+)-class$")
    elif ckpt_type == "pretrain":
        ckpt_dir = PRETRAIN_DIR
        pattern = re.compile(r"^best_pretrain-(\d+)-class$")
    else:
        raise ValueError(f"Unknown ckpt_type: {ckpt_type}. Use 'finetune' or 'pretrain'.")
    
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    best_folder = None
    best_num_classes = -1
    
    for folder in os.listdir(ckpt_dir):
        match = pattern.match(folder)
        if match:
            num_classes = int(match.group(1))
            if num_classes > best_num_classes:
                best_num_classes = num_classes
                best_folder = folder
    
    if best_folder is None:
        raise FileNotFoundError(
            f"No checkpoint folders matching pattern found in {ckpt_dir}. "
            f"Expected folders like 'best_{ckpt_type}-N-class'."
        )
    
    return best_folder, best_num_classes


def get_checkpoint_num_classes(folder_name: str) -> int:
    """
    Extract the number of classes from a checkpoint folder name.
    
    Args:
        folder_name: Checkpoint folder name (e.g., "best_finetune-63-class")
    
    Returns:
        Number of classes, or None if pattern doesn't match.
    """
    match = re.search(r"-(\d+)-class$", folder_name)
    if match:
        return int(match.group(1))
    return None


def forward_pass(model, img : t.Tensor, char_map : dict, log_time : bool = False):
    """
    Run an inference pass with a model given a 4D tensor image input (B, C, H,
    W)
    
    Returns a list of probabilities assigned to each character (length of model
    logits output dim), and the predicted char according to the char_map. If
    log_time is true, the inference time will also be returned.
    """
    assert img.ndim == 4, "Image must have batch and channel dimensions!"
    
    start_time = time.time()
    with t.no_grad():
        input_tensor = img.to(model.device)
        logits = model(input_tensor)
        char_probs = t.softmax(logits, dim=1).squeeze(0) # (num_classes, )
        predicted_idx = t.argmax(logits, dim=1).item()
    end_time = time.time()
    predicted_char = char_map[predicted_idx]
    
    if log_time:
        return char_probs, predicted_char, end_time - start_time
    
    return char_probs, predicted_char,