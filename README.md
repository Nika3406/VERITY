# VERITY

## Overview

**VERITY** is a research project for detecting propaganda techniques in politically charged online discourse, with a special focus on Reddit. Unlike traditional propaganda detection systems that are trained mainly on formal news articles, this project targets informal, conversational, and community-driven language such as Reddit posts, replies, memes, slogans, sarcasm, and emotionally loaded political arguments.

The project uses a transformer-based multi-label classifier built on **DeBERTa-v3-large** with **RoRA adapters**. It combines established propaganda benchmarks such as **SemEval-2020 Task 11** with a newly collected Reddit-domain dataset. The goal is not only to classify propaganda, but also to support fragment/span-level detection and visualization of propaganda techniques inside real online discussions.

---

## Project Goals

- Collect politically charged Reddit posts and comments from multiple communities.
- Use SemEval-2020 propaganda annotations as a baseline label space.
- Build a Reddit-domain propaganda dataset through active learning and LLM-assisted annotation.
- Fine-tune a DeBERTa-based multi-label classifier for propaganda detection.
- Calibrate per-label thresholds instead of relying on one fixed threshold.
- Evaluate predictions using overlap-aware fragment/span-level scoring.
- Provide prediction tools that can highlight propaganda techniques in text.

---

## Why Reddit?

Reddit is a useful domain for propaganda detection because political language on the platform is often different from traditional news writing.

Reddit discussions frequently contain:

- Informal and emotional language
- Sarcasm, memes, slogans, and inside jokes
- Echo chamber behavior
- Partisan framing
- Short comments with high rhetorical intensity
- Community-specific slang and coded political references

Because of this, models trained only on formal news articles may struggle when applied to Reddit-style discourse.

---

## Model Architecture

The main classifier uses:

- **Backbone:** `microsoft/deberta-v3-large`
- **Task:** Multi-label propaganda technique classification
- **Output:** 14 propaganda technique labels
- **Loss:** `BCEWithLogitsLoss`
- **Adapter Method:** RoRA-style low-rank adapters
- **Thresholding:** Per-label calibrated thresholds

The model freezes most of the DeBERTa backbone and injects RoRA adapters into later transformer layers. This makes training more efficient while preserving the general language understanding ability of the pretrained model.

RoRA adapters are applied to transformer projection layers such as:

- Query projection
- Key projection
- Value projection
- Dense projection

This allows the model to adapt to propaganda-specific patterns without fully retraining all 442M parameters.

---

## Propaganda Labels

The project uses the SemEval-style 14-label propaganda taxonomy:

1. `appeal_to_authority`
2. `appeal_to_fear_prejudice`
3. `bandwagon_reductio_ad_hitlerum`
4. `black_and_white_fallacy`
5. `causal_oversimplification`
6. `doubt`
7. `exaggeration_minimisation`
8. `flag_waving`
9. `loaded_language`
10. `name_calling_labeling`
11. `repetition`
12. `slogans`
13. `thought_terminating_cliches`
14. `whataboutism_straw_men_red_herring`

These labels support multi-label classification, meaning a single Reddit post or sentence can contain multiple propaganda techniques at the same time.

---

## Dataset Pipeline

### 1. SemEval Processing

The project begins by processing SemEval-2020 Task 11 propaganda data into a clean 14-label format.

Main scripts:

```bash
python semeval_extract_with_gold.py
python semeval_data_processor.py
```

These scripts produce files such as:

```text
semeval_examples.csv
semeval_processed.csv
gold.csv
semeval_input_texts.csv
```

The SemEval data is used as the initial supervised training foundation.

---

### 2. Reddit Data Collection

Reddit data is collected using PRAW and subreddit discovery tools.

Main scripts:

```bash
python subreddit_discovery.py
python reddit_scraper_sentences.py
python batch_scraper.py
```

The Reddit scraper collects posts and replies from politically active or politically relevant communities. Example target communities include:

```text
r/politics
r/conservative
r/worldnews
r/PoliticalDiscussion
r/NeutralPolitics
r/PoliticalCompassMemes
r/AskPolitics
r/geopolitics
```

The scraper removes unnecessary metadata and focuses on text content suitable for annotation and model training.

---

### 3. Active Learning Selection

Instead of randomly labeling Reddit posts, the project uses active learning to select the most informative examples.

Main script:

```bash
python active_learning_selector.py
```

Supported uncertainty strategies include:

- Entropy sampling
- Margin uncertainty
- Least confidence
- Mixed strategy

Example command:

```bash
python active_learning_selector.py \
  --reddit_csv datasets/ \
  --model_dir ./deberta-propaganda-multilabel \
  --output_csv active_learning_candidates.csv \
  --n_samples 1500 \
  --strategy mixed
```

This selects Reddit samples where the model is most uncertain, making annotation more efficient.

---

### 4. LLM Ensemble Labeling

Selected Reddit samples are labeled using a local LLM ensemble through Ollama.

Main script:

```bash
python llm_ensemble_labeler.py
```

Example models:

```text
llama3.1:8b
mistral:7b
gemma2:9b
```

Example command:

```bash
python llm_ensemble_labeler.py \
  --input active_learning_candidates.csv \
  --output al_labeled_consensus.csv
```

Only samples where multiple LLMs agree are kept as high-confidence annotations. This improves label quality and reduces noisy training data.

---

## Training Pipeline

The main training script is:

```bash
python debertaL_v2.py
```

The training pipeline supports multiple phases:

### Phase 1: Full Fine-Tuning

```bash
python debertaL_v2.py --phase finetune
```

This trains the model on the processed SemEval dataset.

### Phase 2: RoRA Adapter Training

```bash
python debertaL_v2.py --phase rora --epochs 4
```

This freezes the backbone and trains RoRA adapters.

### Phase 3: Active Learning Retraining

```bash
python debertaL_v2.py --phase al_retrain \
  --al_labeled_csv al_labeled_consensus.csv \
  --output_dir deberta-propaganda-multilabel_rora
```

This adds the LLM-labeled Reddit examples into the training set.

### Phase 4: Threshold Calibration

```bash
python debertaL_v2.py --phase calibrate \
  --output_dir deberta-propaganda-multilabel_rora
```

This recalibrates per-label decision thresholds for better precision and recall.

---

## Prediction

The main prediction script is:

```bash
python predict_v2.py
```

This script:

- Loads the trained DeBERTa model
- Injects RoRA adapter weights
- Loads the label mapping
- Loads calibrated per-label thresholds
- Predicts propaganda techniques for new text

The model uses per-label thresholds from:

```text
per_label_thresholds.json
```

If the threshold file is missing, the script falls back to a default threshold of `0.5`.

---

## Fragment-Level Evaluation

Since propaganda detection often requires identifying specific text fragments, this project includes span-level evaluation tools.

Main scripts:

```bash
python export_pred_sentence_fragments.py
python span_shrinker.py
python propaganda_fragment_eval.py
python threshold_sweep_semeval.py
python threshold_optimizer.py
```

The evaluation pipeline supports:

- Sentence-level prediction export
- Character-offset fragment scoring
- Span shrinking for tighter predictions
- Overlap-aware precision, recall, and F1
- Per-technique threshold optimization

This is important because predicting an entire sentence may be too broad when the actual propaganda phrase is only a small part of the sentence.

---

## Example Workflow

A typical full workflow looks like this:

```bash
# 1. Extract SemEval labels and gold spans
python semeval_extract_with_gold.py

# 2. Process SemEval into model-ready format
python semeval_data_processor.py

# 3. Train initial model
python debertaL_v2.py --phase finetune

# 4. Collect Reddit data
python subreddit_discovery.py
python batch_scraper.py

# 5. Select uncertain Reddit samples
python active_learning_selector.py \
  --reddit_csv datasets/ \
  --model_dir ./deberta-propaganda-multilabel \
  --output_csv active_learning_candidates.csv \
  --n_samples 1500 \
  --strategy mixed

# 6. Label selected samples using LLM ensemble
python llm_ensemble_labeler.py \
  --input active_learning_candidates.csv \
  --output al_labeled_consensus.csv

# 7. Retrain with Reddit data
python debertaL_v2.py --phase al_retrain \
  --al_labeled_csv al_labeled_consensus.csv \
  --output_dir deberta-propaganda-multilabel_rora

# 8. Calibrate thresholds
python debertaL_v2.py --phase calibrate \
  --output_dir deberta-propaganda-multilabel_rora

# 9. Run prediction
python predict_v2.py
```

---

## Repository Structure

```text
BERTPropagandaDetection/
│
├── debertaL_v2.py
├── predict_v2.py
├── active_learning_selector.py
├── llm_ensemble_labeler.py
├── factoid_loader.py
│
├── semeval_data_processor.py
├── semeval_extract_with_gold.py
├── export_pred_sentence_fragments.py
├── propaganda_fragment_eval.py
├── span_shrinker.py
├── threshold_optimizer.py
├── threshold_sweep_semeval.py
│
├── batch_scraper.py
├── reddit_scraper_sentences.py
├── subreddit_discovery.py
│
├── datasets/
├── semeval_datasets/
├── deberta-propaganda-multilabel/
├── deberta-propaganda-multilabel_rora/
│
└── README.md
```

---

## Key Features

- Reddit-focused propaganda detection
- Multi-label classification
- DeBERTa-v3-large backbone
- RoRA adapter fine-tuning
- Active learning sample selection
- LLM ensemble annotation pipeline
- Per-label threshold calibration
- Span-level evaluation
- SemEval-compatible 14-label taxonomy
- Reddit scraping and subreddit discovery tools

---

## Ethical Considerations

This project is designed for research and educational use.

Important safeguards:

- Reddit usernames and personal identifiers should not be released.
- Only public Reddit text should be collected.
- Metadata should be minimized or removed before dataset release.
- The model should not be used to automatically punish, ban, or profile users.
- Predictions should be treated as probabilistic research outputs, not absolute truth.
- Political bias should be monitored carefully during dataset construction and evaluation.

The goal is to study rhetorical manipulation patterns, not to target individuals.

---

## Limitations

This project has several known limitations:

- LLM-generated annotations may still contain errors.
- Reddit language is highly context-dependent.
- Sarcasm and memes remain difficult for transformer classifiers.
- Some propaganda categories overlap heavily.
- Low-frequency labels may produce weaker F1 scores.
- Models trained on political Reddit may not generalize to all online communities.
- Propaganda detection is partly interpretive and should include human review.

---

## Future Work

Possible improvements include:

- Building a larger manually verified Reddit annotation set
- Adding explainability visualizations for detected spans
- Creating a web interface for propaganda highlighting
- Comparing BERT, RoBERTa, DeBERTa, and Longformer models
- Improving sarcasm and meme-aware detection
- Adding subreddit-level bias analysis
- Testing cross-domain generalization on news, forums, and social media
- Creating a public anonymized benchmark dataset

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended packages include:

```text
torch
transformers
datasets
pandas
numpy
scikit-learn
tqdm
praw
python-dotenv
sentence-transformers
matplotlib
nltk
```

For LLM ensemble labeling, install and run Ollama:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama serve
```

---

## Example Use Case

Given a Reddit comment such as:

```text
Only an idiot would believe this. Everyone knows they are destroying the country.
```

The model may detect:

```text
name_calling_labeling
loaded_language
bandwagon_reductio_ad_hitlerum
```

The final tool can be extended to highlight the exact phrase or sentence fragment responsible for each prediction.

---

## Project Status

This project currently includes:

- Reddit scraping tools
- SemEval processing tools
- Active learning candidate selection
- LLM ensemble labeling
- DeBERTa-v3-large training
- RoRA adapter fine-tuning
- Prediction script
- Fragment-level evaluation tools
- Threshold optimization scripts

The project is suitable for a research portfolio, academic presentation, or continued development into a usable propaganda analysis tool.

---

## Author

**Nicholas Shvelidze**  
Computer Science Student  
Penn State University

---

## Acknowledgments

This project is inspired by prior work on fine-grained propaganda detection, especially SemEval-2020 Task 11 and the Propaganda Techniques Corpus. It extends those ideas into Reddit-style informal political discourse using active learning, LLM-assisted labeling, and transformer-based classification.
