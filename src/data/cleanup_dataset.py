# @Time    : 2026-01-15 11:50
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : cleanup_dataset.py

# script to clear bad samples from dataset by using the show_my_data.py script
# to see data, and then using delete_sample to remove samples you say look bad

import os
from src.data.utils import delete_sample, RAW_MOUSE_DATA_DIR, build_char_map, build_inverse_char_map, char_to_folder_name, load_chars
from src.visualize.show_my_data import plot_positions, create_figure
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-letter", type=str, default="a", help="Start letter")
    parser.add_argument("--end-letter", type=str, default=" ", help="End letter")
    args = parser.parse_args()
    
    inv_map = build_inverse_char_map()
    char_map = build_char_map()
    start_idx = inv_map[args.start_letter]
    end_idx = inv_map[args.end_letter]
    
    # go through all characters in raw_mouse_data, give a id to each sample, and
    # let users input the ids of the samples to delete

    # initialize a single figure for reuse across all characters
    fig, axes = create_figure(nrows=4, ncols=8)

    # show all the characters at once just like show_my_data.py
    total_deleted_samples = 0
    for char_idx in range(start_idx, end_idx + 1):
        char = char_map[char_idx]
        folder_name = char_to_folder_name(char)
        if not os.path.isdir(os.path.join(RAW_MOUSE_DATA_DIR, folder_name)):
            continue
        filenames, arrays = load_chars(os.path.join(RAW_MOUSE_DATA_DIR, folder_name), return_tensor=False)
        
        # automatically remove empty arrays
        valid_filenames = []
        valid_arrays = []
        for filename, arr in zip(filenames, arrays):
            if arr.size == 0:
                delete_sample(char, filename)
                total_deleted_samples += 1
                print(f"[AUTO] Deleted empty sample: {filename}")
            else:
                valid_filenames.append(filename)
                valid_arrays.append(arr)
        filenames, arrays = valid_filenames, valid_arrays
        
        if len(arrays) == 0:
            print(f"No valid samples for '{char}', skipping...")
            continue
        
        plot_positions(filenames, arrays, persistent_plot=True, use_id_title=True,
                       fig=fig, axes=axes, title_char=char)
        # filenames are ordered by timestamp, so id=0 should be the earliest
        # timestamp
        id_to_filename = {i : filename for i, filename in enumerate(filenames)}

        print("Which samples do you want to delete? (enter ids separated by commas)")
        ids = input("IDs: ")
        if ids == "":
            continue

        filenames_to_delete = [id_to_filename[int(id)] for id in ids.split(",")]
        for filename in filenames_to_delete:
            delete_sample(char, filename)
            print(f"Deleted {filename}")
            total_deleted_samples += 1
    
    plt.close(fig)
    print(f"Complete - deleted {total_deleted_samples} bad samples")

if __name__ == "__main__":
    main()
