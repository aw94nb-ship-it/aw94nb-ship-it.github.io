---
title: 'Affective computing from scratch'
description: 'A beginner’s map of the field: what an emotion is, how machines try to read one, the datasets and benchmarks everyone uses, what changed between 2023 and 2026, and where I think the open opportunities are in data and models.'
date: 2026-11-27
portfolios: ['affective-computing']
---

I wrote this for someone who knows machine learning but has never worked on
emotion. It goes in four steps:

1. what an emotion is, and why that question decides everything downstream;
2. the signals, tasks and models;
3. the datasets and benchmarks the field measures itself on, and how to read
   their numbers;
4. what changed recently, and where I see room to do better, in data
   collection and in model development.

The theory section builds on my
[notes on chapter 23 of Jurafsky and Martin](/essays/slp3-ch23-affect-lexicons/),
which covers how feelings are represented in words. The rest draws on the
papers listed at the end and on a 2026 literature survey a collaborator is
writing.

## Part 1. What are we even measuring?

### Affective computing in one paragraph

Rosalind Picard named the field in 1997: systems that **recognise**,
**interpret**, **simulate** or **respond to** human affect. In practice that
means four jobs:

- **Reading** a person: from their words, voice, face or body, estimate what
  they feel.
- **Understanding** why: link the feeling to the situation (“she is upset
  *because* the deadline moved”).
- **Responding** well: reply in a way that fits the person’s state.
- **Expressing**: generate speech or a face that carries an emotion.

Most of the research money and almost all benchmarks go to the first job.

### “Affect” is several different things

Psychologists separate affective states by how long they last and what they
are about. Scherer’s typology, which chapter 23 opens with:

| State | What it is | Lasts | Examples |
|---|---|---|---|
| Emotion | A brief response to an event that matters to you | seconds to minutes | angry, afraid, proud |
| Mood | Diffuse, low-intensity, often without a clear cause | hours to days | cheerful, gloomy, irritable |
| Interpersonal stance | How you position yourself toward someone *in this exchange* | one conversation | warm, cold, supportive, contemptuous |
| Attitude | Lasting likes and dislikes toward things or people | months to years | liking, hating, valuing |
| Personality trait | A stable disposition | years | anxious, hostile, outgoing |

Classic “sentiment analysis” (is this review positive?) is attitude detection.
A tool that listens to a meeting is mostly reading emotion and stance. These
are different targets. If you average a model’s output over thirty seconds, you
have quietly turned an emotion detector into a mood detector.

### Four theories of what an emotion is

Every dataset encodes one of these theories, usually without saying so.

**1. Basic emotions.** Paul Ekman argued for a small set of universal emotions,
each with its own facial signal and physiology: happiness, sadness, anger,
fear, disgust, surprise (sometimes contempt). Plutchik’s wheel has eight,
arranged in opposing pairs (joy–sadness, anger–fear, trust–disgust,
anticipation–surprise). Almost every classic dataset uses a six- or
seven-class version of this list plus “neutral”.

**2. Dimensions.** James Russell placed feelings on a plane with two axes:
**valence** (pleasant to unpleasant) and **arousal** (calm to activated).
Mehrabian added **dominance** (in control to controlled). “Excited” and “calm”
are both pleasant but differ in arousal; “angry” and “afraid” are both
unpleasant and activated but differ in dominance. Sentiment is just the
valence axis.

**3. Appraisal.** An emotion is the result of judging an event against your
goals: does it matter, who caused it, can I cope, is it fair? Two situations
that look identical in valence and arousal can produce guilt (I caused it) or
anger (you caused it). Appraisal explains *why* a feeling arises, which is why
context matters so much in conversation.

**4. Constructed emotion.** Lisa Feldman Barrett argues that emotion
categories are not natural kinds with fixed fingerprints. The brain builds an
emotion from body signals and context using learned concepts. Related work by
Alan Cowen and Dacher Keltner finds that people report far more than six
categories, with smooth gradients between them: 27 distinct categories for
reactions to short videos.

You don’t have to pick a winner. The practical lesson is that each theory
gives you a different **label space**:

| Theory | Labels look like | Typical datasets |
|---|---|---|
| Basic | 6 to 8 mutually exclusive classes | IEMOCAP, MELD, CREMA-D, AffectNet |
| Dimensional | numbers for valence, arousal, dominance | MSP-Podcast, Aff-Wild2 |
| Appraisal | causes, evaluations, explanations | rare as labels; emotion-cause and reasoning tasks |
| Constructed | many overlapping words, distributions, free text | GoEmotions (27), EmoNet (40), open-vocabulary sets |

### The uncomfortable fact: labels are perceptions

In 2019 Barrett and colleagues reviewed the evidence for reading emotion from
faces. People do smile when happy and scowl when angry more often than chance,
but expression varies a lot across people, cultures and situations, and the
same face can mean several things. A later meta-analysis found that the
“expected” facial configuration appears in only a minority of real emotional
episodes.

So almost every label in the field means “what a stranger thought this person
seemed to feel”, not “what the person felt”. These are called **perceived**
and **felt** (or self-reported) emotion, and they can differ a lot. A system
should describe its output honestly: *this sounded frustrated*, not *this
person is frustrated*.

## Part 2. Signals, tasks and models

### What each signal carries

| Signal | Good at | Weak at |
|---|---|---|
| Words (transcript, chat) | valence, specific emotions, causes, stance | arousal; sarcasm when written flat |
| Voice (pitch, loudness, rate, timbre) | arousal (how activated someone is) | valence on its own |
| Face and body | smiles, frowns, attention, some arousal | the inner state (see the critique above) |
| Physiology (heart, skin, EEG) | arousal and stress, with self-report labels | needs wearables; rarely available |

The most repeated finding of the last few years: **arousal is carried by the
voice, valence largely by the words.** When researchers fine-tuned a speech
model to predict valence on podcasts, they found much of its gain came from
the model implicitly learning the words being said.

### The main tasks

- **Speech emotion recognition (SER):** one audio clip in, an emotion class or
  valence/arousal numbers out.
- **Emotion recognition in conversation (ERC):** label every turn of a dialogue,
  using the earlier turns and who is speaking.
- **Facial expression recognition (FER):** label a face image or video clip.
  **Action unit (AU) detection** instead says which facial muscles moved (cheek
  raiser, brow lowerer), which is more objective.
- **Multimodal sentiment:** words + voice + face of someone giving an opinion,
  predict how positive they are.
- **Open-vocabulary recognition:** answer with any emotion words, plus the cues
  that support them (“tense: clipped answers, voice rising at the end”).
- **Response tasks:** empathetic replies, emotional-support conversations,
  expressive text-to-speech.
- **Emotional-intelligence tests for LLMs:** described scenarios, ask the
  model what a character feels or what would help.

### How the models evolved

**Era 1 (roughly to 2018): hand-built features.** Experts computed acoustic
features (pitch statistics, energy, spectral shape) or facial landmarks, then
trained a small classifier on one corpus. Models rarely transferred to new
data.

**Era 2 (2019 to 2023): general encoders.** Large models pre-trained on huge
amounts of unlabelled audio or images became the default starting point. A
few terms you will see everywhere:

- **Encoder:** a model that turns raw input into a vector of numbers (an
  embedding). Speech: wav2vec 2.0, HuBERT, WavLM, and the encoder half of
  Whisper. Text: BERT and RoBERTa. Vision: masked autoencoders and CLIP.
- **Self-supervised:** pre-trained without labels, by predicting hidden parts
  of the input.
- **Frozen encoder + linear probe:** keep the encoder fixed and train only a
  tiny layer on top. If that works, the emotion information was already in the
  embedding.
- **Fine-tuning:** also update the encoder’s weights on the labelled task.

Two results from this era stand out. **emotion2vec** (2024) pre-trained a
speech encoder on emotional speech, and a linear layer on top beat much larger
general models on the standard four-class test. And on text, fine-tuned
RoBERTa models with speaker and context tricks set the bar for conversation
benchmarks.

**Era 3 (2023 to now): LLMs.** Three things happened at once.

1. **The LLM became the classifier.** Papers like InstructERC (2023) feed a 7B
   LLM the dialogue history and an instruction (“what does speaker B feel in
   the last turn?”) and fine-tune it cheaply with **LoRA** (a small set of
   extra weights instead of retraining the whole model). These now lead the
   conversation benchmarks, a few points above the encoder systems.
2. **Audio and “omni” LLMs listen directly.** SALMONN, Qwen-Audio and
   Qwen2.5-Omni connect a speech encoder to an LLM. GPT-4o and Gemini do the
   same commercially. You can ask them for an emotion with no training at all
   (**zero-shot**). Their results are so far modest: zero-shot GPT-4o-audio
   scores well below small fine-tuned audio models on IEMOCAP from a single
   utterance, and giving it twelve turns of context gains it about ten points.
3. **Open vocabulary and explanation.** Instead of picking one of six classes,
   models like Emotion-LLaMA and AffectGPT answer with free emotion words and a
   description of the audible and visible cues behind them. The MER challenge
   series added an open-vocabulary track in 2024 and is now built around LLMs.

A useful middle path is to **describe the voice to a text LLM**: measure
pitch, loudness and speaking rate, write them out as words (“speaking fast and
loud, pitch rising”), and add that to the transcript. On IEMOCAP that gained
about 2.5 points with no audio model inside the LLM at all.

## Part 3. Datasets and benchmarks

### The datasets everyone uses

| Dataset | Signals | Size | Source | Labels | Commercial use |
|---|---|---|---|---|---|
| IEMOCAP (2008) | audio, video, text | ~12 h, 10 actors | acted dyadic scenes | 9 classes + valence/arousal/dominance | no (research only) |
| MSP-Podcast (2019–25) | audio, text | 409 h, >264k turns | real podcasts | 8 classes + valence/arousal/dominance | paid licence |
| MELD (2019) | audio, video, text | 13.7k utterances | the TV sitcom *Friends* | 7 classes + sentiment | media is copyrighted |
| CMU-MOSEI (2018) | audio, video, text | 23k sentences, 1,000 speakers | YouTube opinion videos | sentiment −3..+3, 6 emotions | features only |
| CREMA-D (2014) | audio, video | 7.4k clips, 91 actors | 12 fixed sentences, acted | 6 classes | yes (ODbL) |
| AffectNet (2017) | images | ~450k labelled | web photos | 8 classes + valence/arousal | no (research only) |
| GoEmotions (2020) | text | 58k comments | Reddit | 27 emotions + neutral | yes (Apache-2.0) |
| CANDOR (2023) | audio, video, text | 1,656 conversations, >850 h | real video calls | self-reported feelings | gated, commercial allowed |
| EmoNet-Voice (2025) | audio | ~5,000 h synthetic | generated voices | 40 categories with intensity | yes (CC BY 4.0) |

Three things jump out.

- **Most data is acted or scripted.** Actors performing “angry” on a fixed
  sentence are clean and balanced, and they overstate what a model can do on a
  real conversation.
- **Almost none of it allows commercial training.** IEMOCAP, AffectNet and the
  open-vocabulary MER sets are research-only; MELD and other TV or film sets
  sit on copyrighted media.
- **Self-report is rare.** CANDOR and Meta’s Seamless Interaction (2025) are
  among the few natural-conversation corpora where people said what *they*
  felt.

### How humans agree, and why it matters

Before looking at model scores, look at how often people agree with each
other. Agreement is measured with **Cohen’s or Fleiss’ κ (kappa)**: 0 means no
better than chance, 1 means perfect agreement.

| Dataset | Human agreement |
|---|---|
| IEMOCAP | κ 0.27 over all labels, 0.40 on turns with a majority |
| MSP-Podcast | κ 0.41 on the main emotion |
| MELD | κ 0.43 on emotion, 0.91 on positive/negative/neutral |
| GoEmotions | mean κ 0.29 across 27 emotions |
| AffectNet | two annotators picked the same class 60.7% of the time |

On natural material, people agree only moderately on *which* emotion, and
much better on *whether it is positive or negative*. That sets a ceiling: if
three raters with κ ≈ 0.4 vote on each label, many test items have a
“correct” answer that a competent person would dispute.

### Reading benchmark numbers

The metrics differ by subfield, and they are not interchangeable:

- **Weighted accuracy (WA) / weighted F1:** dominated by the common classes.
  On MELD, where “neutral” dominates, an audio model can score 52% WA and only
  28% **unweighted accuracy (UA)**, the average over classes.
- **Macro-F1:** every class counts equally. Challenges use it.
- **CCC (concordance correlation):** for continuous valence/arousal; 1 is
  perfect.
- **κ** for human agreement, which corrects for chance. Accuracy does not, so
  a model’s accuracy and the raters’ κ are not on the same scale.

Here is roughly where the best verified results sit:

| Benchmark | Best result | How to read it |
|---|---|---|
| IEMOCAP, 4 classes, audio | ~72–73% WA (frozen Whisper or emotion2vec + light head) | only turns where raters agreed; mostly clear-cut |
| MSP-Podcast, 8 classes (Interspeech 2025 challenge) | macro-F1 0.43, baseline 0.33 | real speech is hard; the winner used a text encoder too |
| MELD, text + context (ERC) | weighted F1 ~70–72 (LoRA-tuned 7B LLMs) | audio alone gets ~52% WA here |
| AffectNet, 8 classes | ~64% accuracy | stuck for years, close to the 60.7% two-rater agreement |
| RAF-DB faces | >92% accuracy | ~40 raters per image, so the labels are cleaner |
| CREMA-D, audio | ~76% WA | humans recovered the actor’s intended emotion 41% of the time |
| CMU-MOSEI, positive vs negative | 85–88% | flat since 2022 |

The CREMA-D row is the most instructive one. The model beats humans by 35
points because it was trained on the actors’ *intended* labels from twelve
repeated sentences, and it can learn cues people don’t use. That is not
superhuman empathy. It is the best illustration of why acted data overstates
ability.

### Five traps when comparing papers

1. **Speaker leakage.** IEMOCAP has no official split. About half of older
   papers split randomly, so the same actor appears in training and test, and
   their numbers can’t be compared with speaker-independent ones. EmoBox and
   EMO-SUPERB now publish fixed splits.
2. **Different class sets.** “IEMOCAP” can mean four classes, six, or other
   subsets, each dropping the hard turns.
3. **Different context.** A conversation model sees the previous turns; a
   speech model sees one clip. “Text beats audio” often means “context beats no
   context”.
4. **Benchmarks that don’t test listening.** When a pipeline of speech-to-text
   plus a text LLM matches an audio LLM on an “emotion from speech” benchmark,
   the benchmark is mostly testing the words. That happened on MELD in AHELM
   and on AIR-Bench.
5. **LLM judges.** Open-vocabulary answers are often scored by another LLM, so
   the score depends on the judge model and its version.

### The industry yardsticks

- **Challenges.** The Interspeech 2025 challenge on MSP-Podcast (93 teams) is
  the main one for natural speech. MER (2023 to 2026) covers multimodal and
  open-vocabulary recognition on Chinese film and TV. ABAW covers faces in
  unconstrained video on Aff-Wild2.
- **Benchmark suites.** EmoBox and EMO-SUPERB for speech, MER-UniBench for
  multimodal LLMs, AIR-Bench, SD-Eval and AHELM for how well audio LLMs attend
  to *how* something is said.
- **Emotional intelligence for LLMs.** EQ-Bench, EmoBench, SECEU and standard
  psychology EI tests. On text scenarios, frontier LLMs now score at or above
  the human average: six LLMs averaged 81% on five established EI tests whose
  human samples averaged 56%. But those are *described* situations. From real
  video, GPT-4V scored 44% on a two-choice belief task where people scored 93%.
  Reasoning about feelings in text is largely solved; perceiving them from
  people is not.
- **Regulation.** The EU AI Act bans emotion recognition at work and in
  schools (except for medical or safety reasons) since February 2025, and
  treats other uses as high-risk with a duty to tell people they are being
  analysed. Where and for whom a product runs is now a design input.

## Part 4. What changed recently, and where the opportunity is

### The state of the field in 2026

1. **The models got general; the data didn’t keep up.** One encoder serves
   many corpora and one LLM can classify, explain and respond. The reference
   datasets are still small, mostly acted or scripted, labelled by a few
   strangers, and closed to commercial use.
2. **The big new datasets are synthetic or model-labelled.** EmoNet-Voice is
   generated speech rated by experts; MER-Caption’s 115k clips were described
   by models and checked in part by people. Studies even find human evaluators
   prefer GPT-4’s emotion labels to the original human ones. The risk is
   circularity: models trained on model-made labels and scored against
   model-made references can improve without getting better at reading people.
3. **Words and context did most of the work.** On conversational benchmarks,
   a lot of what looked like progress in “multimodal” reading came from the
   transcript and the dialogue history.
4. **Measured against real speech, progress is slow.** The best eight-class
   score on natural podcasts went from 0.36 (Odyssey 2024) to 0.43 (Interspeech
   2025) macro-F1, and the attribute baseline was hard to beat both years.
5. **Disagreement is becoming data.** Benchmarks are starting to keep every
   rater’s vote and score against the distribution, not a single majority
   label.

### Opportunity 1: data collection

The gap is clear: **natural conversation, with consent, labelled by the person
who felt it, and licensed for commercial use.** Almost no public dataset has
all four. A product that people use in their real conversations can collect
exactly that, if it is designed for it from the start.

What I would collect, roughly from most to least reliable:

- **Passive baselines per person.** How this person’s pitch, pace and
  loudness usually sound. No labels needed, and it turns an absolute reading
  (“loud”) into a relative one (“louder than she usually is”).
- **In-the-moment taps.** A one-tap “was this right?” on a reading the person
  actually saw. Cheap, sparse and very valuable, because it is the person’s own
  label at the moment it mattered.
- **Implicit feedback.** Did they use the suggestion, dismiss it, or ignore it?
  Noisy, but plentiful.
- **End-of-conversation self-report.** A short “how did that feel?” with a few
  dimensions, collected while the memory is fresh.
- **Model-drafted labels, clearly marked.** Useful for volume. Kept separate
  from human labels and never used alone as the test set.

Rules that make this data usable later:

- **Consent from everyone in the conversation**, not only the user, and clear
  retention limits. Respect the regulation map above.
- **Several raters on a test subset**, keeping every vote, so agreement can be
  reported and the ceiling is known.
- **Record perceived and felt separately.** A stranger’s rating and the
  person’s own report are two labels, not one.
- **Store context**: who was speaking, the preceding turns, the setting. Most
  public data throws this away.

### Opportunity 2: model development

- **Read words and voice for what each does best.** Let the transcript and
  context drive *which* feeling; let the voice drive *how strong* (arousal) and
  catch the moments where tone contradicts the words. Mismatch between channels
  is itself a signal: sarcasm, polite irritation, a brave face.
- **Personalise with a small layer on a shared model.** Keep a strong shared
  encoder frozen and learn a light per-person calibration from that person’s
  feedback. Guard it: only switch to the personal version when it beats the
  shared one on that person’s own held-out labels, and fall back otherwise.
  Naive online learning on noisy feedback can easily make a model worse.
- **Output calibrated distributions, not one confident word.** With labels
  this ambiguous, “60% frustrated, 30% anxious” is more honest than
  “frustrated”, and a product can stay quiet when the model is unsure. Few
  affect papers even report calibration.
- **Work in real time, on the device.** The strongest open-vocabulary models
  are 7B+ LLMs; a live conversation needs a decision within a turn. Small
  encoders run in real time but give coarse labels. Streaming evaluation
  (latency, stability over time, half-finished sentences) is nearly absent from
  benchmarks.
- **Use LLMs for reasoning, small models for perception.** LLMs already reason
  about feelings in described situations at human level. The weak link is
  turning live audio into a good description. A small fast perception model
  that writes a precise description, handed to an LLM, plays to both
  strengths.
- **Evaluate honestly.** Natural conversations, speaker-independent splits,
  stated context, a words-only control to show what the voice adds, several
  raters per item, self-report where possible, and human spot checks wherever
  an LLM is the judge.

## Glossary

- **Valence / arousal / dominance:** pleasant–unpleasant, calm–activated,
  in control–controlled.
- **Perceived vs felt emotion:** what an observer thinks you feel vs what you
  report feeling.
- **κ (kappa):** agreement between raters, corrected for chance.
- **WA / UA:** accuracy over all items / averaged over classes.
- **Macro-F1:** F1 averaged over classes, each counted equally.
- **CCC:** agreement between predicted and true continuous values.
- **Encoder:** turns input into an embedding vector.
- **Linear probe:** a single trained layer on a frozen encoder.
- **LoRA:** cheap fine-tuning through small added weight matrices.
- **Zero-shot:** used with no task-specific training.
- **ERC / SER / FER:** emotion recognition in conversation / from speech / from
  faces.

## Sources

- Jurafsky and Martin, *Speech and Language Processing*, 3rd ed. draft,
  [chapter 23](https://web.stanford.edu/~jurafsky/slp3/23.pdf); my
  [notes](/essays/slp3-ch23-affect-lexicons/).
- Picard, *Affective Computing* (1997). Ekman, “An argument for basic
  emotions” (1992). Russell, “A circumplex model of affect” (1980). Scherer,
  “What are emotions? And how can they be measured?” (2005). Barrett, *How
  Emotions Are Made* (2017). Cowen and Keltner (2017).
- Barrett et al., “Emotional expressions reconsidered” (2019); Durán and
  Fernández-Dols (2021).
- Datasets: Busso et al. (IEMOCAP 2008; MSP-Podcast 2019, 2025), Poria et al.
  (MELD 2019), Zadeh et al. (CMU-MOSEI 2018), Cao et al. (CREMA-D 2014),
  Mollahosseini et al. (AffectNet 2017), Demszky et al. (GoEmotions 2020), Reece
  et al. (CANDOR 2023), Schuhmann et al. (EmoNet-Voice 2025), Agrawal et al.
  (Seamless Interaction 2025).
- Models: Ma et al., emotion2vec and EmoBox (2024); Wagner et al. (2023); Lei
  et al., InstructERC (2023); Wu et al., SpeechCueLLM (2024); Cheng et al.,
  Emotion-LLaMA (2024); Lian et al., AffectGPT (2025); Murzaku et al., OmniVox
  (2025).
- Benchmarks and challenges: Naini et al., Interspeech 2025 MSP-Podcast
  challenge; Goncalves et al., Odyssey 2024 challenge; Wu et al., EMO-SUPERB
  (2024); Lee et al., AHELM (2025); Yang et al., AIR-Bench (2024); Schlegel et
  al., LLMs on EI tests (2025); Jin et al., MMToM-QA (2024).
- Regulation: EU Artificial Intelligence Act, Article 5(1)(f) and Annex III;
  Commission guidelines on prohibited practices (February 2025).
