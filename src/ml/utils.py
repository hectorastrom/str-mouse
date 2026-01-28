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


def discover_all_checkpoints(ckpt_type: str = "finetune") -> dict[int, str]:
    """
    Discover all checkpoint folders matching the best_*-N-class pattern.
    
    Args:
        ckpt_type: Either "finetune" or "pretrain"
    
    Returns:
        Dict mapping num_classes -> full checkpoint path (including best.ckpt).
        For example: {53: "checkpoints/mouse_finetune/best_finetune-53-class/best.ckpt", ...}
    
    Raises:
        ValueError: If ckpt_type is invalid.
    """
    if ckpt_type == "finetune":
        ckpt_dir = FINETUNE_DIR
        pattern = re.compile(r"^best_finetune-(\d+)-class$")
    elif ckpt_type == "pretrain":
        ckpt_dir = PRETRAIN_DIR
        pattern = re.compile(r"^best_pretrain-(\d+)-class$")
    else:
        raise ValueError(f"Unknown ckpt_type: {ckpt_type}. Use 'finetune' or 'pretrain'.")
    
    checkpoints = {}
    
    if not os.path.isdir(ckpt_dir):
        return checkpoints
    
    for folder in os.listdir(ckpt_dir):
        match = pattern.match(folder)
        if match:
            num_classes = int(match.group(1))
            ckpt_path = os.path.join(ckpt_dir, folder, "best.ckpt")
            if os.path.isfile(ckpt_path):
                checkpoints[num_classes] = ckpt_path
    
    return checkpoints


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
    checkpoints = discover_all_checkpoints(ckpt_type)
    
    if not checkpoints:
        if ckpt_type == "finetune":
            ckpt_dir = FINETUNE_DIR
        else:
            ckpt_dir = PRETRAIN_DIR
        raise FileNotFoundError(
            f"No checkpoint folders matching pattern found in {ckpt_dir}. "
            f"Expected folders like 'best_{ckpt_type}-N-class'."
        )
    
    best_num_classes = max(checkpoints.keys())
    # Extract folder name from path
    best_folder = f"best_{ckpt_type}-{best_num_classes}-class"
    
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