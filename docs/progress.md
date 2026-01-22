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
this software transforms the mouse into an additional input medium (e.g. for 
executing a macro or a bunch of on-device gestures!)

Here are what I think the biggest limitations of **scribble** are, and how I
intend to solve them:
1. Character recognition is best paired with natural language parser
    - We could pair the outputs & confidence levels from mouse to char with a
      language model, which votes on the plausability of certain strings based
      on its knowledge of complete words. To make this fast & on device is the
      challenge. 

1. Images are a poor medium for rich, timeseries mouse velocity data
    - We'd be better off implementing an RNN or LSTM for this same task: super
      lightweight, and using the stroke velocity data in entirety
    - See [third attempt](attempt_3.md)!

1. Data collection needs to scale, a lot
    - I want to make this software public, so people can use it, give the model 
    feedback (training data), and massively scale data collection from just
    myself.
    - Could also add an online platform and send to my friends for a quick boost

As a personal interest, I'd love to add a UI to decode characters in real time, similar to Apple's
Scribble. 

<img src="apple_scribble.png" width=350 alt="Apple Watch Scribble">


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
