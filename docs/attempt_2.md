## Second attempt: Using image representations

In this second attempt, I no longer had any out-of-box models to try. Rather, I
treated this as a distinct problem of character classification and needed to
train my own model from scratch for the task.

Since I didn't want to collect 1000000 strokes myself, I knew I needed to
pretrain. So although we're throwing away timeseries data when converting mouse
velocity recordings into images (this **does** really bother me!), it enables us to 
profit from massive online datasets of handwritten characters, like we do here.
That seemed like a good trade in the short-term.

### Datasets
There are two primary datasets involved in this attempt:

**EMNIST**

Firstly, I benefitted from [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) - an online dataset of handwritten characters. 

This was a rich dataset and easy to get setup, so we could use it for
pretraining of our CNN. 

Originally in attempt 2, we used the `letters` split of the EMNIST dataset, which combined
uppercase + lowercase character drawings into 26 characters. In [attempt
2.5](attempt_2-5.md) I fix this by using the 62-class `byclass` split of the
dataset. I missed this split of the data when I was first building this project,
though the feature detections learned on these case-agnostic labels were
evidently sufficient for a decent result.

**My own mouse data**

The second dataset I colleceted from my own mouse movements. I set up a data
collection script `src/data/collect_data.py` to prompt me to write all characters we want
to classify, then to automatically collect the stroke from stroke start to long
pause. 

I tried to vary my movements as much as possible in this.

Data size: `365 bytes / 1 min of collection`

### CNN Architecture
The CNN is nothing special - most of it is cookie-cutter. It's called `StrokeNet`
and accepts standard "MNIST-style" single channel input `(1x28x28)`.
It consists of three convolutions blocks, each seperated by a `2x2` pooling
layer to reduce input size by half each time.

1. Block 1 expands from 1 -> 32 channels
1. Block 2 expands from 32 -> 64 channels
1. Block 2 expands from 64 -> 128 channels

The model ends with a global pooling + FCC. It features a little dropout, which
I didn't extensively test if it was necessary.

The model size I used was 302K parameters - pretty small.

Notably, this is a `PyTorch Lightning` model. I opted to use this library (and
many other abstraction libraries, like `datasets` and `PIL`) throughout this
project. Afterall, the focus was not on rewriting boiler plate code. I like these
abstraction libraries a lot. 

### Data Collection 
I want to add a specific section on the data collection. My goal was to work
under the constraints of the original challenge, so I wanted to build
`src/data/collect_data.py` on top of the original `src/data/mouse_recorder.py` script. This
saves mouse velocities, which (in this attempt) we integrate to positions, and
pass as images to our CNN.

*my stats: 😉*
```bash
Time elapsed:  00:04:19
Your labelling rate: 0.42 chars/sec
```

### Training & Finetuning
Pretraining occurs in the `src/ml/emnist_pretrain.py` script. It's facilitated through
`Lightning`. I tracked results through `wandb`, which is one of my favorite tools.

Couple of issues I had to whack-a-mole
1. Preprocessing in `main.py` `build_image()` originally made a black stroke on
   a white background - this is completely inverted from EMNIST, and destroyed
   finetune performance
1. Train for longer! Originally I thought 15 batches of my small dataset would
   be enough, but it turned out 150 (1 OOM more) was best
1. When I added uppercase characters, pretraining on `num_classes = 53` and
   finetuning that head worked **MUCH** worse than pretraining on a `num_classes = 27` 
   head, which we later discard in finetuning for the appropriate `53` class head.


### Pretraining Results
The model used is 302K parameters.

*Note that the pretraining checkpoints work for both 27 and 53 class finetuning,
since we just discard the final head in either case.*

| Checkpoint | Epochs | Batch Size | Training Samples | Test Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| twilight-pond-18 | 20 | 128 | 124800 | 94.9% | temp 27 class head; 28x28 |
| iconic-dawn-40 | 20 | 128 | 124800 | 94.9% | reuse 53 class head; 28x28 |
| dulcet-energy-56 | 12 | 128 | 124800 | 94.6% | temp 27 class head; 64x64 |

### Finetuning Results

LOWERCASE LETTERS ONLY (27):
| Checkpoint Name | Epochs | Batch Size | Training Samples | Downsample Size | Seed | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| twilight-pond-18 | **50** | 64 | 321 | `28x28` | 42 | 81.7% |
| twilight-pond-18 | **100** | 64 | 321 | `28x28` | 42 | 91.4% |
| twilight-pond-18 | **200** | 64 | 321 | `28x28` | 42 | 90.3% |
| twilight-pond-18 | 150 | 32 | 321 | **`64x64`** | 42 | 93.5% |
| twilight-pond-18 | 150 | 64 | **415** | `28x28` | 42 | 98.3% |

UPPERCASE + LOWERCASE (53): 
| Checkpoint Name | Pretrain Checkpoint | Epochs | Batch Size | Training Samples | Downsample Size | Seed | Test Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| serene-microwave-46 | twilight-pond-18 | 125 | 64 | 296 | `28x28` | 42 | 82.6% | |
| sweet-frog-47 | twilight-pond-18 | 150 | 64 | **556** | `28x28` | 42 | 91.9% | |
| legendary-sky-48 | twilight-pond-18 | 150 | 64 | 556 | `28x28` | 42 | 87.5% |prefer recent trials |
| pretty-night-49 | twilight-pond-18 | 150 | 64 | 742 | `28x28` | 42 | 83.0% | |
| fresh-wood-50 | twilight-pond-18 | 200 | 64 | 742 | `28x28` | 42 | 89.2% | |
| faithful-eon-51 | twilight-pond-18 | **150** | 64 | **704** | `28x28` | 42 | 92.6% | after cleaning dataset |
| frosty-sea-53 | twilight-pond-18 | 150 | 64 | **927** | `28x28` | 42 | 92.9% | more data |

### Online testing
Models can be online tested using the `src.decode.real_time` script. Just point
the `--ckpt` to the right place and see how the model does.

The script allows *ex posto facto* labeling and model performance analysis when
you hit `Ctrl + C`.

# TODO
- [X] Fix preprocessing
  - [X] Render at final downsampled resolution directly (like 64x64)
  - [X] Preserve aspect ratio w/ stroke centered in frame
  - [X] Add linear interpolation between positions (due to 15ms sampling — may have gaps)
  - [X] Invert colors so it's black stroke on white background
- [X] Train a tiny classifier 
  - [X] Pretrain CNN on EMNIST
  - [X] Finetune CNN on our data
- [X] Build data labelling script
- [X] Build online testing/decoding script
    - [X] Add end-screen data labelling
- [X] BUG: best models always predict one char (`j` or `i`)
    - Just try `polar-water-38` in `src/decode/real_time.py` to see!
    - FIX: use 28x28 (train img size) for inference in `src/decode/real_time.py`
- [X] Add support for uppercase letters 
  - In both data collection & throughout labelling script (stuff is pretty
      hardcoded for 27 characters lol)
- [X] Add data augmentations