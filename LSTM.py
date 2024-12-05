import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the RNN Model: LSTM
"""
word embedding loopup layer -> LSTM layer -> fully connected layer (on the hidden state of the last LSTM cell) -> output prediction
hidden_dim = hidden dimension for LSTM cell = 100
activation for LSTM cell: tanh
fc layer dimension = 100, activation: None (linear)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, fc_dim, output_dim, embeddings=None):
        super(TextLSTM, self).__init__()
        # self.n_layers = n_layers
        self.n_layers = 1   # default
        self.hidden_dim = hidden_dim

        # Word embedding lookup layer
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim).from_pretrained(torch.FloatTensor(embeddings))
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM layer with tanh activation (built-in)
        # batch_first=True then the input&output tensors are (batch, seq, features) instead of (seq, batch, features)
        # The LSTM takes word embeddings as inputs, and outputs hidden states.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Fully connected layer (linear)
        self.fc = nn.Linear(hidden_dim, fc_dim)

        # Output prediction
        self.output = nn.Linear(fc_dim, output_dim)

    def forward(self, x):
        # Since we're not using batching, the input x will be of shape (seq_length,)
        # After embedding, x will be of shape (1, seq_length, embed_dim) because of unsqueeze(0)
        # x = torch.tensor(x)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).to(device)  
        x = self.embedding(x).unsqueeze(0)
        # initialzie hidden state and cell state for first input
        h0, c0 = self.init_hidden()   # (n_layers=1, batch_size, hidden_dim)
        # pass in input and hidden state into LSTM and get the outputs
        # lstm has builtin tanh and initialize hidden states as 0's by default
        x, (hn, cn) = self.lstm(x, (h0, c0))   # x: (batch_size, seq_length, hidden_dim)
        # use the hidden state of the last LSTM cell for the fc layer
        last_hidden = hn.squeeze(0)
        out = self.fc(last_hidden)
        out = self.output(out)
        return out

    def init_hidden(self):
        h0 = torch.zeros(self.n_layers, 1, self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, 1, self.hidden_dim).to(device)
        return h0, c0
