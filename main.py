import numpy as np
import os
import re
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from CNN import *
from LSTM import *
import cnn_util
import rnn_util


# Load pre-trained word embeddings as a dict
def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        _, dim = map(int, f.readline().split())  # get dim from the first line
        embeddings = {}
        for line in f:   # start from the 2nd line
            parts = line.split()
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
    return embeddings

# Load datasets
def load_reviews(folder_path):
    reviews = []
    for sentiment in ['positive', 'negative']:
        path = os.path.join(folder_path, sentiment)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r') as f:
                # tuple: (str review, str p/n label)
                reviews.append((f.read(), sentiment))
    return reviews

# Tokenize and preprocess
def tokenize(text, keep_punctuation=False):
    text = text.lower()
    if not keep_punctuation:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()

def preprocess_data_cnn(reviews, word_to_idx, lmax):
    tokenized_reviews = [tokenize(review) for review, _ in reviews]
    X = [words_to_indices(tokens, word_to_idx) for tokens in tokenized_reviews]
    X = [review[:lmax] + [0] * (lmax - len(review)) for review in X]  # zero padding
    y = [1 if sentiment == 'positive' else 0 for _, sentiment in reviews]
    return torch.tensor(X), torch.tensor(y)

def preprocess_data_rnn(reviews, word_to_idx):
    tokenized_reviews = [tokenize(review) for review, _ in reviews]
    X = [words_to_indices(tokens, word_to_idx) for tokens in tokenized_reviews]
    y = [1 if sentiment == 'positive' else 0 for _, sentiment in reviews]
    return X, y

def words_to_indices(tokens, word_to_idx):
    return [word_to_idx.get(token, 0) for token in tokens]

# Plotting loop (after training and testing are complete)
def plot_stats(model_name, train_losses, train_accs, test_losses, test_accs):
  f, axarr = plt.subplots(2,2, figsize=(12, 8))
  axarr[0, 0].plot(train_losses)
  axarr[0, 0].set_title("Train objective vs Epoch")
  axarr[0, 1].plot(train_accs)
  axarr[0, 1].set_title("Train accuracy vs Epoch")
  axarr[1, 0].plot(test_losses)
  axarr[1, 0].set_title("Test objective vs Epoch")
  axarr[1, 1].plot(test_accs)
  axarr[1, 1].set_title("Test accuracy vs Epoch")
  plt.suptitle(model_name)
  plt.show()

def run_cnn(X_train, y_train, X_test, y_test):
    # models w/ and w/o pre-trained embeddings
    model_w_embed = TextCNN(vocab_size=10000,
                            embed_dim=100,
                            num_filters=100,
                            filter_size=3,
                            fc_dim=100,
                            output_dim=1,
                            embeddings=embedding_matrix)
    model_wo_embed = TextCNN(vocab_size=10000,
                            embed_dim=100,
                            num_filters=100,
                            filter_size=3,
                            fc_dim=100,
                            output_dim=1)

    num_epochs = 10
    # Create DataLoader for batching
    batch_size = 30  # modify later, common numbers 32 or 64
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model_names = ["CNN w/ pretrained embedding", "CNN w/o pretrained embedding"]
    # Binary Cross-Entropy with Logits Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='sum')   # loss function, "objective"
    optimizer_w = torch.optim.Adam(model_w_embed.parameters(), lr=0.0002)
    optimizer_wo = torch.optim.Adam(model_wo_embed.parameters(), lr=0.0005)

    name = model_names[0]
    print(f"Training {name}...")
    cnn_w_train_losses, cnn_w_train_accs, cnn_w_test_losses, cnn_w_test_accs = cnn_util.train_model(model_w_embed, train_loader, test_loader, criterion, optimizer_w, num_epochs)
    name = model_names[1]
    print(f"Training {name}...")
    cnn_wo_train_losses, cnn_wo_train_accs, cnn_wo_test_losses, cnn_wo_test_accs = cnn_util.train_model(model_wo_embed, train_loader, test_loader, criterion, optimizer_wo, num_epochs)

    plot_stats("CNN w/ pretrained embedding", cnn_w_train_losses, cnn_w_train_accs, cnn_w_test_losses, cnn_w_test_accs)
    plot_stats("CNN w/o pretrained embedding", cnn_wo_train_losses, cnn_wo_train_accs, cnn_wo_test_losses, cnn_wo_test_accs)

def run_rnn(X_train, y_train, X_test, y_test):
    # models w/ and w/o pre-trained embeddings
    lstm_w_embed = TextLSTM(vocab_size=10000,
                            embed_dim=100,
                            hidden_dim=100,
                            fc_dim=100,
                            output_dim=1,
                            embeddings=embedding_matrix)
    lstm_wo_embed = TextLSTM(vocab_size=10000,
                            embed_dim=100,
                            hidden_dim=100,
                            fc_dim=100,
                            output_dim=1)

    num_epochs = 10

    model_names = ["LSTM w/ pretrained embedding", "LSTM w/o pretrained embedding"]
    # Binary Cross-Entropy with Logits Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(reduction='sum')   # loss function, "objective"
    optimizer_w = torch.optim.Adam(lstm_w_embed.parameters(), lr=0.003)
    optimizer_wo = torch.optim.Adam(lstm_wo_embed.parameters(), lr=0.003)

    model_names = ["RNN w/ pretrained embedding", "RNN w/o pretrained embedding"]
    name = model_names[0]
    print(f"Training {name}...")
    lstm_w_train_losses, lstm_w_train_accs, lstm_w_test_losses, lstm_w_test_accs = rnn_util.train_model(lstm_w_embed, X_train, y_train, X_test, y_test, criterion, optimizer_w, num_epochs)
    name = model_names[1]
    print(f"Training {name}...")
    lstm_wo_train_losses, lstm_wo_train_accs, lstm_wo_test_losses, lstm_wo_test_accs = rnn_util.train_model(lstm_wo_embed, X_train, y_train, X_test, y_test, criterion, optimizer_wo, num_epochs)

    plot_stats("RNN w/ pretrained embedding", lstm_w_train_losses, lstm_w_train_accs, lstm_w_test_losses, lstm_w_test_accs)
    plot_stats("RNN w/o pretrained embedding", lstm_wo_train_losses, lstm_wo_train_accs, lstm_wo_test_losses, lstm_wo_test_accs)



if __name__ == "__main__":
    # data_path = '/Users/MinyiRen/Desktop/11441/hw3/hw3-handout/src/data'
    data_path = "../data/"   # assuming data folder is also under src

    embeddings = load_embeddings(data_path+'all.review.vec.txt')  # dict, word:vec, size=56050
    train_reviews = load_reviews(data_path+'train')  # list of tuples, len=#reviews=2000
    test_reviews = load_reviews(data_path+'test')    # len=2000

    tokenized_train = [tokenize(review) for review, _ in train_reviews]  # list of lists of tokenized reviews
    word_freq = Counter(word for review in tokenized_train for word in review)
    top_10k_words = {word for word, _ in word_freq.most_common(10000)}
    filtered_embeddings = {word: vec for word, vec in embeddings.items() if word in top_10k_words}

    # Convert words to indices
    word_to_idx = {word: idx for idx, (word, _) in enumerate(word_freq.most_common(10000))}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}


    # Calculate statistics (Q1)
    unique_words = set(word for review, _ in train_reviews for word in tokenize(review, keep_punctuation=True))
    num_positive = sum(1 for _, sentiment in train_reviews if sentiment == 'positive')
    num_negative = len(train_reviews) - num_positive
    avg_length = sum(len(tokenize(review, keep_punctuation=True)) for review, _ in train_reviews) / len(train_reviews)
    max_length = max(len(tokenize(review, keep_punctuation=True)) for review, _ in train_reviews)

    print("Total unique words:", len(unique_words))
    print("Total training examples:", len(train_reviews))
    print("Positive to negative ratio:", num_positive / num_negative)
    print("Average document length:", avg_length)
    print("Max document length:", max_length)

    embedding_matrix = [filtered_embeddings.get(idx_to_word[idx], [0]*100)
                        for idx in range(10000)]  # dim = lmax*dim = 10k*100

    # ----------------------------------- CNN ----------------------------------- #
    # Preprocess data for CNN
    # lmax = max(len(tokenize(review)) for review, _ in train_reviews) # =3816 (w/ punctuation), =3357 (w/o)
    # X_train, y_train = preprocess_data_cnn(train_reviews, word_to_idx, lmax)
    # X_test, y_test = preprocess_data_cnn(test_reviews, word_to_idx, lmax)

    # run_cnn(X_train, y_train, X_test, y_test)


    # ----------------------------------- RNN ----------------------------------- #
    # Preprocess data for RNN
    X_train, y_train = preprocess_data_rnn(train_reviews, word_to_idx)  # list of lists, list of ints
    X_test, y_test = preprocess_data_rnn(test_reviews, word_to_idx)

    run_rnn(X_train, y_train, X_test, y_test)

