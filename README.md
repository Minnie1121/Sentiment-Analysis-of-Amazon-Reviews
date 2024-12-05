# Sentiment-Analysis-of-Amazon-Reviews

## Overview
This project implements deep learning models to classify text data effectively, focusing on both Recurrent Neural Networks (RNN) and Convolutional Neural Networks (CNN). It explores the impact of pre-trained word embeddings and compares their performance with embeddings learned from scratch. The dataset is a subset of Multi-Domain Sentiment Dataset (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/), which was provided by the 11-441 Text and Graph-based Mining course at Carnegie Mellon University and is not included in this repository. It includes Amazon Reviews on different products, including books and musical instruments.

## Project Goals
* Develop deep learning-based text classifiers using RNN (LSTM) and CNN architectures.
* Compare the effectiveness of pre-trained word embeddings against learned embeddings.
* Evaluate and visualize model performance on a sentiment classification dataset.

## Key Features
### Preprocessing
* Tokenized text data and converted it to word indices.
* Used padding to handle variable-length sequences.
* Limited vocabulary to the top 10,000 frequent words for efficiency.

### Models
1. **RNN with LSTM**:
   - Network: Word embedding lookup layer -> LSTM layer -> fully connected layer(on the hidden state of the last LSTM cell) -> output prediction
   - Hidden dimension for LSTM cell: 100
   - Activation for LSTM cell: tanh
   - Fully connected layer dimension 100, activation: None (i.e. this layer is linear)

2. **CNN**:
   - Network: Word embedding lookup layer -> 1D CNN layer -> fully connected layer -> output prediction
   - b. Number of filters: 100
   - c. Filter length: 3
   - d. CNN Activation: Relu

### Evaluation
- Measured accuracy and loss on training and testing datasets.
- Visualized training/testing performance metrics over time for all model variations (with and without pre-trained embeddings).

## Results
- **CNN with pre-trained embeddings** achieved the best accuracy and was faster to train than RNN models.
- Pre-trained embeddings improved performance and reduced training time for both RNN and CNN.
- CNN's use of data batching and efficient design contributed to faster training times than RNN.

## Implementation Details
- **Programming Language**: Python.
- **Libraries**: PyTorch, NumPy, Matplotlib.
- **Code Structure**:
  - Separate implementations for RNN and CNN models.
  - Includes modules for data preprocessing and visualization.

## Challenges and Observations
- Zero-padding affected LSTM performance negatively but worked well with CNN due to convolutional operations.
- Pre-trained embeddings significantly helped models converge faster and perform better.
- Initial learning rates and optimization methods (Adam) played a critical role in achieving convergence.

## Conclusion
This project demonstrates the effectiveness of deep learning architectures in text classification tasks and highlights the advantages of using pre-trained embeddings. CNN emerged as the faster and more efficient model, while RNN provided valuable insights into sequential text processing. Future work may involve experimenting with advanced architectures and larger datasets.
