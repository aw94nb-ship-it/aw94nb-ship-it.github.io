---
title: 'Reading notes: lexicons for sentiment, affect and connotation'
description: 'What chapter 23 of Jurafsky and Martin’s Speech and Language Processing teaches about representing feelings in words, building lexicons, and finding the words that separate two kinds of text.'
date: 2026-11-26
portfolios: ['applied-ml']
---

These are my notes on chapter 23 of Dan Jurafsky and James H. Martin’s
*Speech and Language Processing* (3rd edition draft, August 2026):
[“Lexicons for Sentiment, Affect, and Connotation”](https://web.stanford.edu/~jurafsky/slp3/23.pdf).
It is a short chapter with light math. Its value is in the questions it makes
you ask before you build anything that claims to read feelings: *which* kind of
feeling, measured *how*, and labeled by *whom*.

## What you need before reading

The chapter leans on four earlier ideas. If they are fresh, it reads in an
afternoon.

- **Word counts as probabilities:** P(w | c), the share of class c’s words that
  are w; odds and log odds; smoothing with pseudo-counts (Appendix B, chapter 3).
- **Embeddings and cosine similarity:** words as vectors, cos(a, b) = a·b / (|a||b|),
  and the centroid (the mean of several vectors) (chapter 5).
- **Pointwise mutual information:** PMI(x, y) = log P(x, y) / (P(x) P(y)) (chapter 5).
- **Classifiers over features:** logistic regression and what a feature weight
  means (chapter 4); contextual embeddings from a BERT-style encoder (chapter 9).

## 1. Affect is several different things

The chapter opens with Scherer’s typology of affective states, ordered roughly
by how long they last:

| State | What it is | Examples |
|---|---|---|
| Emotion | A brief response to an event that matters | angry, sad, proud |
| Mood | Diffuse, low-intensity, long-lasting, often without a clear cause | cheerful, gloomy, irritable |
| Interpersonal stance | The stance one person takes toward another *in this exchange* | warm, cold, contemptuous, supportive |
| Attitude | Enduring likes and dislikes toward things or people | liking, hating, valuing |
| Personality trait | A stable disposition | nervous, hostile, reckless |

Classic sentiment analysis (“is this review positive?”) is really attitude
detection. A system that reads a conversation is mostly reading emotion and
interpersonal stance, which are different targets with different time scales.
Naming which one you are after is the first design decision.

## 2. Two ways to represent an emotion

- **Basic emotions:** a small fixed set from which others are built. Ekman’s six
  (surprise, happiness, anger, fear, disgust, sadness) and Plutchik’s wheel of
  eight in four opposing pairs (joy–sadness, anger–fear, trust–disgust,
  anticipation–surprise).
- **Dimensions:** a point in a space. Most models use **valence** (how pleasant),
  **arousal** (how activated) and often **dominance** (how much control).
  Sentiment is then just the valence axis.

The two views complement each other. Categories are easy to name and show;
dimensions give you distances, averages and a way to say “a little more
upset than before.” The chapter also flags **appraisal theory**, where an
emotion is a judgement about an event: was it my fault, was it certain, can I
control it. That is what tells guilt from anger when valence and arousal look
the same, and the authors expect it to matter more in future work.

One limitation runs through the whole chapter: a lexicon gives a word **one
fixed affect**, whatever the context, the speaker, or the tone of voice.

## 3. The lexicons that exist

- **Polarity lists:** General Inquirer (1966), MPQA, Hu and Liu. Positive and
  negative word lists, a few thousand words each.
- **NRC VAD:** valence, arousal and dominance scores for 20,000 English words.
- **NRC EmoLex:** about 14,000 words marked for Plutchik’s eight emotions plus
  positive and negative.
- **NRC Emotion Intensity:** real-valued anger, fear, joy and sadness scores.
- **LIWC:** 73 hand-built categories (negative emotion, insight, inhibition,
  negation, family…) used widely in social psychology.
- **Concreteness ratings:** 40,000 words rated from abstract (1) to concrete (5).

Check the licence before using any of these in a product. Several of the NRC
resources are free for research only.

## 4. Three ways to build a lexicon

### By hand, with crowd workers (23.3)

Two details are worth stealing for any labeling project:

- **Prime the right sense first.** EmoLex asked annotators a quick synonym
  question (“which word is closest to *startle*?”) before asking about emotion,
  so they judged the intended meaning of the word.
- **Best-worst scaling.** Show four items and ask which is highest and which is
  lowest on a scale. A word’s score is the share of times it was picked best
  minus the share it was picked worst. It is more reliable than asking people
  for a number on a 1–9 scale.

Reliability is checked with **split-half reliability**: split the annotations
in half and correlate the two halves’ scores.

### From a few seed words (23.4)

Start from seed words for two poles (good, lovely, excellent… versus bad,
horrible, poor…) and score every other word by how close it is to each pole.

- **Semantic axis:** average the seed embeddings into two centroids, V+ and V−.
  The axis is V+ − V−, and a word’s score is the cosine between its embedding
  and that axis. It is three lines of numpy and works for any pair of poles,
  not just sentiment.
- **Label propagation (SentProp):** connect each word to its nearest neighbours,
  run random walks from each seed set, and score a word by how often walks from
  the positive seeds land on it compared with walks from the negative seeds.
  Bootstrapping over random subsets of the seeds gives a confidence for each
  score.
- **Other cues:** adjectives joined by *and* tend to share polarity (“fair and
  legitimate”), adjectives joined by *but* tend to differ, negating prefixes
  flip it (*adequate / inadequate*), and WordNet synonyms and antonyms spread
  labels (SentiWordNet).

Seed choice matters. The same method with general seeds, Twitter seeds or
finance seeds produces different lexicons, which is a feature when you want a
domain-specific one.

### From a signal the world already provides (23.5)

Reviews come with star ratings, so the ratings label the words for free.

- **Potts diagrams** plot P(w | rating), normalised across ratings. Strong
  positive words make a J shape, strong negative words a reverse J, mild words a
  hump. Emphatics (*absolutely*, *totally*) form a U: they are common at both
  extremes, so they signal intensity, not direction.
- **Log odds ratio with an informative Dirichlet prior** (Monroe et al., 2008)
  is the method in the chapter I expect to use most. It answers: which words are
  over-represented in corpus i compared with corpus j?

  Raw frequency differences exaggerate very rare and very common words. The fix
  adds counts from a large background corpus as a prior, then turns the result
  into a z-score:

  ```
  δ_w = log[(f_w^i + α_w) / (n^i + α_0 − f_w^i − α_w)]
      − log[(f_w^j + α_w) / (n^j + α_0 − f_w^j − α_w)]

  σ²(δ_w) ≈ 1 / (f_w^i + α_w) + 1 / (f_w^j + α_w)

  score_w = δ_w / σ(δ_w)
  ```

  Here f is a word’s count in each corpus, n each corpus’s size, α_w the word’s
  scaled count in the background corpus and α_0 that corpus’s scaled size. On
  Yelp, comparing 1-star with 5-star reviews, it surfaces more than sentiment
  words: 1-star reviews use negation, *we*, past tense and talk about staff;
  5-star reviews use emphatics, *you* and talk about dessert.

  It works for any two sets of text, which makes it a good tool for auditing a
  dataset: what words actually separate the examples labeled X from the rest?
  If the answer is a handful of stock phrases, a model trained on that data
  will learn the phrases.

## 5. Using lexicons

- **With little or no training data (23.6):** count positive and negative words,
  optionally weighted, and call the text positive if the ratio passes a
  threshold λ. Crude, but a useful baseline.
- **With training data (23.7):** a classifier over all words or bigrams is hard
  to beat when the training and test text are alike. Lexicon counts help as extra
  features when data is sparse or the test text is different, and the classifier’s
  weights show which lexicons matter for which class. At very large scale, keep
  only phrases with high PMI.
- **Affect toward a specific person (23.8):** average a frozen encoder’s
  contextual embeddings for each word, fit regressions from those to the word’s
  valence, arousal and dominance (relabelled sentiment, agency and power), then
  score each mention of a person in context and average over the mentions.
- **Connotation frames (23.9):** verbs carry implications about their arguments.
  In “A *violated* B”, the writer sympathises with B; in “X *survived* Y”, X is
  sympathetic and Y is a hardship; *implore* puts the agent below the theme in
  power, and *wait* gives the agent low agency.

## What I am taking away

1. **Decide which affective state you are measuring.** Emotion, mood and
   interpersonal stance need different time scales and different labels.
   Smoothing a reading over thirty seconds turns an emotion detector into a mood
   detector, so that should be a deliberate choice.
2. **Use categories and dimensions together.** Give every category word a
   coordinate in valence–arousal–dominance space. Then changes, averages and
   “how far apart are these two readings” are measurable, not ad hoc.
3. **Signals carry different dimensions.** Arousal is easier to hear in a voice
   than to read in a transcript; valence is often clearer in the words. Combining
   per dimension can beat combining per category.
4. **A words-only model behaves like a soft lexicon.** The same sentence gets the
   same reading in every tone of voice. Context and delivery have to be inputs if
   the reading is supposed to depend on them.
5. **Label carefully and measure agreement.** Prime the right sense, use
   best-worst scaling, and report split-half reliability before trusting any
   threshold tuned against the labels.
6. **Audit training data with the informative-Dirichlet log odds.** It is cheap
   and shows what a model trained on that data will latch onto.
7. **Check licences early.** The best-known affect lexicons are research-only.
