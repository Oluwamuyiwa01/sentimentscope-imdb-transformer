# sentimentscope-imdb-transformer
Transformer trained from scratch in PyTorch for IMDB movie review sentiment analysis
# SentimentScope: Transformer Sentiment Analysis on IMDB

A transformer-based model, trained from scratch in PyTorch, that classifies IMDB movie reviews as positive or negative. Built as the final project of the Future AWS AI Programmer course, framed as helping a fictional entertainment company (CineScope) understand user sentiment for its recommendation system.

## Results

| Metric | Value |
|---|---|
| Test accuracy | 77.22% |
| Final validation accuracy | 79.36% |
| Model size | about 4.7M parameters |

## Approach

- **Data:** the 25,000 labeled training reviews were shuffled and split into 22,500 for training and 2,500 for validation. The 25,000-review test set was kept untouched until the end. The classes are perfectly balanced.
- **Tokenization:** `bert-base-uncased` subword tokenizer, with reviews truncated or padded to 128 tokens.
- **Model:** `DemoGPT`, a transformer with 4 layers, 4 attention heads, and 128-dimensional embeddings. I adapted it for classification by adding mean pooling over the token dimension and a linear head with 2 outputs.
- **Training:** 3 epochs, AdamW optimizer, learning rate 3e-4, batch size 32, cross-entropy loss.
- **Inference:** a saved checkpoint and a `SentimentPredictor` class that loads it and predicts sentiment for new reviews.

## Repository contents

- `sentimentscope.ipynb`: the full notebook (data loading, exploration, model, training, testing, inference)
- `sentimentscope_checkpoint.pt`: the trained model weights

## Limitations and next steps

- Reviews are truncated to 128 tokens, so the model never sees the end of longer reviews.
- The attention heads use a causal mask, so each token sees only earlier tokens. Removing it would give bidirectional attention, which often helps classification.
- The mean pooling also averages over padding tokens. A masked mean would be a cleaner choice.
- Fine-tuning a pretrained model such as DistilBERT would likely give a large accuracy gain.

## Dataset

The Large Movie Review Dataset (IMDB, `aclImdb_v1`). The dataset is not included in this repository.
