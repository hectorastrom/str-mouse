# @Time    : 2026-01-13 15:48
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : show_my_data.py

# Script to visualize my collected mouse stroke data

from src.data.utils import char_to_folder_name, RAW_MOUSE_DATA_DIR, load_chars
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

CHARACTER = ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("character", type=str,
                        help="Character trials to visualize (a-z, A-Z, or 'space')")
    parser.add_argument("--folder", type=str, default=RAW_MOUSE_DATA_DIR, help="Folder with data")
    args = parser.parse_args()
    global CHARACTER
    CHARACTER = args.character
    # handle 'space' string input -> ' ' char for folder lookup
    char_input = ' ' if CHARACTER.lower() == 'space' else CHARACTER
    folder_name = char_to_folder_name(char_input)
    filenames, arrays = load_chars(f"{args.folder}/{folder_name}", return_tensor=False)
    plot_positions(filenames, arrays)

def create_figure(nrows=4, ncols=8):
    """
    Create a reusable figure and axes for plotting character trials.
    
    Returns:
        tuple: (fig, axes) matplotlib figure and axes array
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
    plt.ion()  # enable interactive mode
    return fig, axes


def plot_positions(filenames, character_arrays, persistent_plot=False, use_id_title=False, 
                   nrows=4, fig=None, axes=None, title_char=None):
    """
    Plot character trials. Expects separate lists of filenames and arrays.
    
    Args:
        filenames: List of filenames
        character_arrays: List of (2, T) arrays
        persistent_plot: If True, use pause instead of show
        use_id_title: If True, use ID numbers as titles instead of filenames
        nrows: Number of rows in subplot grid
        fig: Existing figure to reuse (if None, creates new figure)
        axes: Existing axes to reuse (if None, creates new axes)
        title_char: Character to display in suptitle (if None, uses global CHARACTER)
    """
    if len(character_arrays) < 4:
        nrows = len(character_arrays)

    # rows, cols
    ncols = len(character_arrays) // nrows + 1

    # create new figure or reuse existing one
    if fig is None or axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=True, sharey=True)
    else:
        # clear all axes for redrawing
        for ax in axes.flat:
            ax.clear()

    char_display = title_char if title_char is not None else CHARACTER
    fig.suptitle(f"All {len(character_arrays)} trials for character: {char_display}")

    # find boundary ranges
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for chunk in character_arrays:
        x, y = chunk
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

    print(f"Ranges: x[{int(min_x)} to {int(max_x)}], y[{int(min_y)} to {int(max_y)}]")

    # set shared limits on first visible axes
    for ax in axes.flat:
        ax.set_xlim(int(min_x)-50, int(max_x)+50)
        ax.set_ylim(-int(max_y)-50, -int(min_y)+50)  # inverted y

    for i, ax in enumerate(axes.flat):  # easy way to idx row col properly
        if i >= len(character_arrays):
            ax.set_visible(False)
            continue

        ax.set_visible(True)
        ax.axis("off") # hide axis
        x, y = character_arrays[i]
        # high y values mean lower on screen, which is opposite of how its
        # plotted: so we plot -y
        ax.plot(x, -y)
        if use_id_title:
            ax.set_title(f"ID: {i}", fontsize=6)
        else:
            ax.set_title(filenames[i], fontsize=6)

    plt.tight_layout()

    if persistent_plot:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)
    else:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    main()
