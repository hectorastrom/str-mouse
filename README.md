# str(mouse)

## About

**str(mouse)** can decode any cursor movement into one of 53 characters ('a-z',
'A-Z' and space). It does this at a rate exceeding `400 chars/s` running
on an M2 Pro MacBook, by transforming the mouse velocity data into an image of
the character that is then classified by a lightweight (0.3M) CNN.

If you want to read about my progress in building this, challenges I ran into,
and thoughts for the future of the project, read [progress.md](docs/progress.md).

## Usage 

### Try it
This repository contains many, many files. If you just want to get started and
play around with it, run:

```bash
uv run -m src.decode.real_time --ckpt best_finetune
```

### Train it
If you want to collect some data and train the model, first run:
```bash
uv run -m src.data.collect_data
```

Then run:
```bash
uv run -m src.ml.finetune --ckpt best_pretrain
```

### Test it
To test your new model on sample velocity data, run:
```bash
uv run main.py --ckpt <ckpt_name>
```

## Results

My best finetuned model achieved `92.9%` accuracy on its offline test set, and
feels pretty good online, too.

All detailed results can be found in [attempt 2 results](docs/attempt_2.md/#finetuning-results)


## Data
My data and checkpoints can be downloaded from a public `s3` bucket.

To download my raw_mouse_data (as a base for finetuning):
`aws s3 sync s3://hectorastrom-str-mouse/raw_mouse_data ./raw_mouse_data`

I intend to upload checkpoints later, when there is more interesting diversity available.
---