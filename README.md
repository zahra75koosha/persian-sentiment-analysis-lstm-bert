# Persian Sentiment Analysis with LSTM and BERT

A deep learning project for binary sentiment classification of Persian text using a CNN-BiLSTM model and a pretrained BERT model.

## Overview

This project implements two deep learning approaches for Persian sentiment analysis:

1. A hybrid CNN-BiLSTM model trained on the provided dataset
2. A pretrained Persian BERT model fine-tuned for sentiment classification

The task is to classify Persian comments into two sentiment categories: **HAPPY** and **SAD**.

## Dataset

The project uses separate training and test datasets:

- `train.csv`
- `test.csv`

Each sample contains a Persian comment and its corresponding sentiment label.

## Models

### 1. CNN-BiLSTM

The first approach combines convolutional and recurrent neural networks.

The architecture includes:

- Tokenization
- Sequence padding
- Conv1D
- MaxPooling1D
- Bidirectional LSTM
- Dense layers
- Dropout
- Softmax output layer

The model is trained using the Adam optimizer and categorical cross-entropy loss.

Early stopping is used to prevent overfitting.

### 2. Pretrained BERT

The second approach uses the pretrained Persian BERT model:

`HooshvareLab/bert-fa-base-uncased-sentiment-snappfood`

The model is fine-tuned on the provided training dataset for binary sentiment classification.

Training configuration includes:

- Learning rate: `2e-5`
- Batch size: `16`
- Epochs: `3`
- Weight decay: `0.01`

The best model is selected based on F1-score.

## Evaluation

The models are evaluated on the test dataset using:

- Accuracy
- Precision
- Recall
- F1-score

## Prediction

The project also provides an interactive prediction function that accepts a new Persian text and returns its predicted sentiment.

Example:

```text
Input: این محصول خیلی خوب بود
Output: HAPPY
