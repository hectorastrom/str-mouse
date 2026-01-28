# @Time    : 2026-01-14 12:24
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : rt_decode.py

# Goal: decode strokes in real time using finetuned model in loop
# will let us test our "vibe check" our live accuracy against the test set
# accuracy

# 1. Continuously record like in data_collect.py, where we chunk & save a new
#    file after e.g. 500ms of mouse inactivity
# 2. Preprocess this movement into an image
# 3. Pass this image to a finetuned model
# 4. Print the character that is decoded from the stroke, and its associated
#    confidence

from src.data.collect_data import record_one_stroke
from src.data.utils import (
    build_img,
    build_char_map,
    build_inverse_char_map,
    char_to_folder_name,
    generate_trial_filename,
    csv_to_numpy,
    velocities_to_positions,
    GlobalInputManager,
)
from src.ml.architectures.cnn import StrokeNet
from src.ml.utils import forward_pass, find_best_checkpoint, get_checkpoint_num_classes, FINETUNE_DIR

from datetime import datetime
import time
import torch as t
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os
import shutil
import argparse

SESSIONS_ROOT = "rt_sessions"
ROOT_DIR = None

def main():
    # 0: setup
    # Find best checkpoint as default
    default_ckpt, default_num_classes = find_best_checkpoint("finetune")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=default_ckpt, 
                        help=f"Finetune checkpoint to evaluate model with from {FINETUNE_DIR} directory (default: {default_ckpt})")
    parser.add_argument("--save-delay", type=int, default=500,
                        help="Time in ms to wait before marking a stroke as a char (default 500ms)")
    args = parser.parse_args()

    # Determine num_classes from checkpoint name, fall back to default
    num_classes = get_checkpoint_num_classes(args.ckpt)
    if num_classes is None:
        num_classes = default_num_classes
        print(f"Warning: Could not determine num_classes from '{args.ckpt}', using {num_classes}")
    
    global ROOT_DIR
    input_manager = GlobalInputManager()
    ckpt_path = f"{FINETUNE_DIR}/{args.ckpt}/best.ckpt"
    print(f"Loading checkpoint: {ckpt_path} ({num_classes} classes)")
    model = StrokeNet.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
    )
    model.eval()
    char_map = build_char_map()
    inv_char_map = build_inverse_char_map()

    session_name = datetime.now().strftime("run_%d-%m-%s")

    ROOT_DIR = f"{SESSIONS_ROOT}/{session_name}"
    os.makedirs(ROOT_DIR, exist_ok=True)
    iter = 0
    all_chars = []  # (char, save_filepath)

    print("Beginning stroke detection!")
    try:
        while True:
            # 1: record using global input manager
            # we'll write to a file, then delete it at the end of the session (can use
            # feedback as data labels!)
            save_filepath = f"{ROOT_DIR}/stroke_{iter}.csv"
            velocities = record_one_stroke(save_filepath, input_manager, args.save_delay)
            print("Stroke detected!")
            iter += 1

            # 2: preprocess velocities into image
            x, y = velocities_to_positions(velocities)
            # VERY VERY IMPORTANT TO DO INFERENCE HOW IT WAS TRAINED
            img = build_img(x, y) # default settings 

            # optional visualization
            # vis_img = to_pil_image(img.unsqueeze(0))
            # vis_img.show(title=f"Character just drawn")

            # 3: pass to finetuned model
            img = img.unsqueeze(0).unsqueeze(0)
            char_probs, pred_char = forward_pass(model, img, char_map)
            pred_char_idx = inv_char_map[pred_char]
            # 4: print decoded character
            print(
                f"I'm {char_probs[pred_char_idx]*100:.2f}% confident you wrote '{pred_char}'"
            )

            all_chars.append((pred_char, save_filepath))

            time.sleep(0.4) # small delay

    except KeyboardInterrupt:
        print("\n\nTerminating stroke detection")
        print("Would you like to label some data?")
        y_n = 'x'
        while y_n not in ('y', 'n'):
            y_n = input("Y/N: ").lower()

        # DON'T LABEL DATA
        if y_n == 'n':
            print("\nProgram complete.")
            clean_up_files(all_chars)
            exit(0)

        # LABEL DATA
        print(f"Great! We have {len(all_chars)} strokes to label together.\n")
        chars_to_keep = {} # idx : correct_char

        # compare all chars against their review
        for i, (pred_char, path) in enumerate(all_chars):
            print(f"For stroke #{i}, I predicted '{pred_char}'")
            x, y = csv_to_numpy(path)
            img = build_img(x, y, downsample_size=128) # bigger for visualization
            vis_img = to_pil_image(img.unsqueeze(0))
            plt.imshow(vis_img, cmap="grey")
            plt.axis('off')
            plt.title(f"Character {i}: {pred_char} predicted")
            plt.pause(0.5) # show for half a second
            plt.clf() # clear figure for reuse

            y_n = "x"
            while y_n not in ("y", "n"):
                y_n = input("Is this correct? (Y/N): ").lower()
                
            # change: we'll also save good samples just to maximize data
            if y_n == "y":
                print("Awesome! I'll remember that.")
                chars_to_keep[i] = pred_char  # i : char
            else:
                print("What was the correct char?")
                correct = confirm_correct_input(inv_char_map)
                print("Thanks! I'll remember that.")
                chars_to_keep[i] = correct  # i : char
            print("----")

        clean_up_files(all_chars, chars_to_keep, training_samples_folder="raw_mouse_data")


def confirm_correct_input(inv_char_map):
    correct = None
    while correct is None:
        user_input = input("(a-z, A-Z, 0-9, or 'space'): ").strip()
        # handle 'space' as special input
        if user_input.lower() == "space":
            correct = " "
        elif len(user_input) == 1 and user_input in inv_char_map:
            correct = user_input
        else:
            print(f"Invalid input '{user_input}'. Enter a single character (a-z, A-Z, 0-9) or 'space'.")
    return correct


def clean_up_files(all_chars : list, chars_to_keep : dict = dict(), training_samples_folder="raw_mouse_data"):
    for i, (_, path) in enumerate(all_chars):
        if i in chars_to_keep.keys():  # idx
            # move file as a new trial to raw_mouse_data
            correct_char = chars_to_keep[i]
            folder_name = char_to_folder_name(correct_char)
            trial_dir = os.path.join(training_samples_folder, folder_name)
            assert os.path.isdir(trial_dir), f"missing {trial_dir} where correct sample should go"
            trial_filename = os.path.join(trial_dir, generate_trial_filename())
            shutil.move(path, trial_filename)
            print(f"Saved {chars_to_keep[i]} as {trial_filename}.")
        else:
            os.remove(path)

    # remove last stroke & folder
    shutil.rmtree(ROOT_DIR)

    print(f"Cleaned up {len(all_chars) - len(chars_to_keep)} files from {ROOT_DIR}!")


if __name__ == "__main__":
    main()
