# Multi-Class Text Classification with BERT 🚀

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/release)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.8%2B-orange)](https://pytorch.org/get-started/locally/)

## Project Overview

### 🏢 Business Overview
In this NLP project, we aim to perform multiclass text classification using a pre-trained BERT model. The dataset consists of more than two million customer complaints about consumer financial products, with columns for complaint text and product labels.

### 🎯 Aim
The goal is to leverage the power of the BERT (Bidirectional Encoder Representations) model, an open-source ML framework for Natural Language Processing, to achieve state-of-the-art results in multiclass text classification.

## Data Description

The dataset includes customer complaints about financial products, with columns for complaint text and product labels. The task is to predict the product category based on the complaint text.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, torch, nltk, numpy, pickle, re, tqdm, sklearn, transformers

## Prerequisite

1. Install the torch framework
2. Understanding of Multiclass Text Classification using Naive Bayes
3. Familiarity with Skip Gram Model for Word Embeddings
4. Knowledge of building Multi-Class Text Classification Models with RNN and LSTM
5. Understanding Text Classification Model with Attention Mechanism in NLP

## Approach

1. **Data Processing**
   - Read CSV, handle null values, encode labels, preprocess text.

2. **Model Building**
   - Create BERT model, define dataset, train and test functions.

3. **Training**
   - Load data, split, create datasets and loaders.
   - Train BERT model on GPU/CPU.

4. **Predictions**
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

## Open for Project Collaboration

🤝 **Kindly connect on [LinkedIn](https://www.linkedin.com/in/vidhi-waghela-434663198/) and follow on [Kaggle](https://www.kaggle.com/vidhikishorwaghela). Let's collaborate and innovate together! 🌐✨
