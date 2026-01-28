# @Time    : 2026-01-13 21:22
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : main.py

# Goal: Use the finetuned StrokeNet to try and decode the original secret
# message inside mouse_velocities.csv

from src.data.utils import build_img, build_char_map, build_inverse_char_map
from src.ml.utils import forward_pass, find_best_checkpoint, get_checkpoint_num_classes, FINETUNE_DIR
import torch as t
from torchvision.transforms.functional import to_pil_image
from src.ml.architectures.cnn import StrokeNet
from src.decode.decipher import integrate_file
import argparse
import matplotlib.pyplot as plt


def main():
    # Find best checkpoint as default
    default_ckpt, default_num_classes = find_best_checkpoint("finetune")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=default_ckpt,
        help=f"Finetune checkpoint to use from {FINETUNE_DIR} directory (default: {default_ckpt})"
    )
    args = parser.parse_args()
    
    # Determine num_classes from checkpoint name, fall back to default
    num_classes = get_checkpoint_num_classes(args.ckpt)
    if num_classes is None:
        num_classes = default_num_classes
        print(f"Warning: Could not determine num_classes from '{args.ckpt}', using {num_classes}")
    
    char_map = build_char_map()
    inverse_char_map = build_inverse_char_map()
    chunks = integrate_file("mouse_velocities.csv", silent=True)
    complete_str = ""
    # load finetuned model
    ckpt_path = f"{FINETUNE_DIR}/{args.ckpt}/best.ckpt"
    print(f"Loading checkpoint: {ckpt_path} ({num_classes} classes)")
    model = StrokeNet.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
    )
    model.eval()

    # ground truth string for comparison (set to expected decoded message)
    actual_string = "SECRET MESSAGE "

    # 0-25: a-z, 26-51: A-Z, 52: space
    def char_to_idx(c):
        if c == " ":
            return 52
        elif c.isupper():
            return ord(c) - ord("A") + 26
        else:  # lowercase
            return ord(c) - ord("a")

    actual_string_idx = [char_to_idx(c) for c in actual_string]

    warmup_time = None
    total_inference_time = 0
    total_correct = 0

    for i, (x, y) in enumerate(chunks):
        # x and y are 1D vectors of mouse positions at timestep=idx
        x = t.from_numpy(x).to(dtype=t.float32)
        y = t.from_numpy(y).to(dtype=t.float32)
        img = build_img(x, y, invert_colors=False)  # match training preprocessing

        vis_img = to_pil_image(img.unsqueeze(0))
        plt.imshow(vis_img, cmap="grey")
        plt.axis("off")
        plt.title(f"Character {i}:")

        # recognize character using finetuned model
        img = img.unsqueeze(0).unsqueeze(0)
        char_probs, predicted_char, inference_time = forward_pass(
            model, img, char_map, log_time=True
        )
        total_inference_time += inference_time
        if warmup_time is None:  # save first inference separately
            warmup_time = inference_time

        # HARDCODE: since we know the string is all uppercase ascii, we're going to
        # enforce that the predicted character is uppercase ascii (idx 26-51)
        predicted_char_idx = inverse_char_map[predicted_char]
        if predicted_char_idx < 26:
            predicted_char_idx = t.argmax(char_probs[26:]).item()
            predicted_char = char_map[predicted_char_idx + 26]

        actual_char_prob = char_probs[actual_string_idx[i]].item()
        complete_str += predicted_char
        actual_char = actual_string[i]

        if predicted_char == actual_char:
            total_correct += 1

        print(
            f"Predicted: {predicted_char} | Actual '{actual_char}' prob: {actual_char_prob:.4f}"
        )

    print(f"\nDecoded string '{complete_str}' in {total_inference_time:.4f} seconds")
    inference_rate = 1 / (
        (total_inference_time - warmup_time) / (len(complete_str) - 1)
    )
    print(
        f"Warmup time: {warmup_time:.4f} | Inference rate: {int(inference_rate)} char / sec"
    )
    print(f"Accuracy: {total_correct*100 / len(chunks):.1f}%")


if __name__ == "__main__":
    main()
