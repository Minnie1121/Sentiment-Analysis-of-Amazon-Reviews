import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN Model
"""
Define the CNN Model.
input arguments:
    vocab_size = # of unique words in data set (= lmax = 3816)
    embed_dim = dim of word embeddings (= 100), input channels
    num_filters = # of filters for each convolution (= 100), output channels
    filter_size = filter length = kernel size (= 3), scalar b/c 1D
    fc_dim = fully connected layer dimension (= 100)
    output_dim = 1
    embeddings = the embedding matrix
"""
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_size, fc_dim,
                 output_dim, embeddings=None):
        super(TextCNN, self).__init__()
        # Word embedding lookup layer: a seq of N words, D-dim vector => NxD)
        # converts each word in the input sequence into a dense vector representation
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(torch.FloatTensor(embeddings))
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 1D CNN layer: (N-filter_size +1)*num_filters = (N-3+1)*100
        # with activation function ReLu = max(0, x)
        self.conv = nn.Conv1d(embed_dim, num_filters, filter_size)

        # Fully connected layer: given fc_dim = 100, with no activation (linear)
        # processes the features extracted by the CNN layer and reduces them to a specific dim
        self.fc = nn.Linear(num_filters, fc_dim)

        # Output prediction: 1D, scalar
        # binary classification, so it's the logit for positive
        # To get probabilities: output_prob = torch.sigmoid(model(input_data))
        self.output = nn.Linear(fc_dim, output_dim)


    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        # Embedding Layer
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        # Transpose to fit Conv1d expected input
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_length)
        # 1D Convolution followed by ReLU activation
        x = torch.relu(self.conv(x))  # (batch_size, num_filters, seq_length-filter_size+1)
        # Max Pooling Layer; to take the max value over the entire seq for each filter
        x = torch.max_pool1d(x, x.size(2)).squeeze(2)  # (batch_size, num_filters)
        # Fully Connected Layer
        x = self.fc(x)   # (batch_size, fc_dim)
        # Output Layer
        x = self.output(x)  # (batch_size, 1), raw logits
        return x
