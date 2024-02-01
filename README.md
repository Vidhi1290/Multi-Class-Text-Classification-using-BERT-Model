# Multi-Class Text Classification with BERT 🚀

## Overview

- Predict consumer financial product categories using BERT, a powerful pre-trained NLP model.
- Dataset: Over two million customer complaints.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, torch, nltk, sklearn, transformers

## Prerequisites

1. Install torch framework.
2. Familiarity with text classification using Naive Bayes, Skip Gram, RNN, and LSTM.
3. Understand attention mechanisms in NLP.

## Approach

1. **Installation**
   - Use pip to install required packages.

2. **Data Processing**
   - Read CSV, handle null values, encode labels, preprocess text.

3. **Model Building**
   - Create BERT model, define dataset, train and test functions.

4. **Training**
   - Load data, split, create datasets and loaders.
   - Train BERT model on GPU/CPU.

5. **Predictions**
   - Make predictions on new text data.

## Project Structure

- **Input:** complaints.csv
- **Output:** bert_pre_trained.pth, label_encoder.pkl, labels.pkl, tokens.pkl
- **Source:** model.py, data.py, utils.py
- **Files:** Engine.py, bert.ipynb, processing.py, predict.py, README.md, requirements.txt

## Takeaways

1. Solving business problems using pre-trained models.
2. Leveraging BERT for text classification.
3. Data preparation and model training.
4. Making predictions on new data.

Explore and contribute! 🌐✨
