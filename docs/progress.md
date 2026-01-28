# Progress / Internal Logs

This file documents my journey in building an effective, fast, and adaptable
mouse->character decoding script. Scroll through to see the different attempts I
tried, some context on design choices, results, and more!

## Attempts
### Attempt 1
In my first attempt to decode mouse velocity data, I thought the best thing to
do was to use out-of-box OCR models to identify the characters in the image
representation of each character. 

Read about this attempt at [attempt 1](attempt_1.md).

### Attempt 2
In my second attempt, I stuck with an image representation but thought to train
a model from scratch on these handwritten characters. Attempt 1 demonstrated
that OCR engines were ill-suited for rough or continuous strokes, instead
expecting clean, typed characters.

This attempt was pretty successful, and encouraged me to make these efforts
public.

Read about this attempt at [attempt 2](attempt_2.md).

### Attempt 2.5
Attempt 2.5 is not so much an attempt as it is an *iteration* of attempt 2. 

Attempt 2.5 fixes a glaring oversight by pretraining on a different, more
domain-accurate (and larger) EMNIST dataset. It also adds support for digits, in
addition to the uppercase and lowercase letters.

Attempt 3 builds directly on this new, revised attempt 2.

### Attempt 3 (ongoing)
Attempt 3 is the lowest information-loss attempt at mouse velocity decoding.
Rather than converting the timeseries representation to a 2D plot, we'll use the
timeseries information directly with an `LSTM`. 

Note: I waited to try this because I didn't know of any large-scale pretraining
dataset on mouse velocity -> character strokes. As such, all data collected for
this attempt had to be my own (and friends').

Read about this attempt at [attempt 3](attempt_3.md).

# What's next for **scribble**?
**scribble**, at is current state, is a cool demo; but it doesn't provide value to
anyone. I could see **scribble** becoming a useful medium to interact with
computers for those with physical disabilities or those with fine motor
impairments.

**scribble** could also become a cool tool for everyone else. Effectively,
this software transforms the mouse into an additional input medium (imagine
being able to quickly execute a macro with a swipe of your mouse!)

Here are what I think the biggest limitations of **scribble** are, and how I
intend to solve them:
1. Images are a poor medium for rich, timeseries mouse velocity data
    - See this [third attempt](attempt_3.md)!
    
1. Character recognition is best paired with natural language parser
    - Pair the outputs & confidence levels from scribble with a
      language model, which votes on the plausability of certain strings based
      on its knowledge of complete words. 
   - Making this fast & on device is the real challenge. I actually think this
     is a hard problem.

1. Data collection needs to scale, a lot
    - Make this software public & directly useful to automate large-scale data collection, so that people will want to
      use it give the model feedback (training data)


# TODO
- [-] Try an alternative timeseries representation (rather than 2D images -
  we're currently throwing away information)
    - [ ] Add support for different sampling rates (15ms vs. 10ms)
- [ ] (bonus) build a cute scribble UI 
- [ ] Think of a way to have people collect data for me...
- [ ] Add "not-a-char" stroke support so model can express uncertainty
- [ ] Add meta learning 
  - Or use model only as an embedder, and add new characters by embedding it.
    Sampling is done with NN on all the embeddings (see periodic episodic learning)

## Other free wins
1. Collect data for longer (biggest)
2. Increase model capacity
3. Improve model architecture
4. Move to timeseries stroke classification
5. Include information about stroke sizes & relative velocities
  - Important for upper vs. lowercase
