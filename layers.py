"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        emb = self.embed(word_idxs)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class WordAndCharEmbedding(nn.Module):
    """Embedding layer used by BiDAF, with the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, use_hwy_encoder=False, use_2_conv_filters = True):
        super(WordAndCharEmbedding, self).__init__()
        n_filters = hidden_size
        self.drop_prob = drop_prob
        self.use_hwy_encoder = use_hwy_encoder
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = CharEmbedding(char_vectors, n_filters=n_filters, kernel_size=3, drop_prob=drop_prob)
        self.proj = nn.Linear(word_vectors.size(1)+2*n_filters, hidden_size, bias=False)
        if use_hwy_encoder:
            self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)   # (batch_size, seq_len, embed_size)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        # word_emb = self.proj(word_emb)  # (batch_size, seq_len, hidden_size//2)

        char_emb = self.char_embed(char_idxs)   # (batch_size, seq_len, hidden_size//2)
        # char_emb = F.dropout(char_emb, self.drop_prob, self.training) #Dropout is done in CNN layer

        emb = torch.cat((word_emb, char_emb), dim=2)  # (batch_size, seq_len, 350)
        assert(emb.size()[2] == word_emb.size()[2] + char_emb.size()[2])

        if self.use_hwy_encoder:
            emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)

        # emb = F.dropout(emb, self.drop_prob, self.training) # ToDo: Should we apply dropout collectively here?

        # emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class CharEmbedding(nn.Module):
    """Embedding layer used by BiDAF, for character-level embedding using 1d-CNN.

    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
    """
    def __init__(self, char_vectors, n_filters, kernel_size, drop_prob, use_2_conv_filters):
        super(CharEmbedding, self).__init__()

        self.n_filters = n_filters
        self.drop_prob = drop_prob
        self.use_2_conv_filters = use_2_conv_filters

        self.char_embed = nn.Embedding.from_pretrained(char_vectors) # do we want to freeze char embeddings as well?

        # we want hidden_size//2 = 50 filters.
        # width of each filter is kernel_size, so CNN will iterate over 3 char long substrings, 1 by 1 extracting some features.
        # height of the filter is: Length of char embedding (char_embed_size)
        # Filter size: (kernel_size, char_embed_size)
        # Each filter will be applied for a word, and will produce a vector
        # Then Max pool over time.
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=char_vectors.size(1),
                out_channels=n_filters,
                kernel_size=kernel_size,
            ),
            nn.ReLU(),
            # nn.Dropout(drop_prob),
            nn.BatchNorm1d(num_features=n_filters), # some people claimed it helped them.
            nn.AdaptiveMaxPool1d(1), #We want to max pool on last dimension (i.e. over ther char sequence, along the width of the word) and we want to pick 1 max value.
        )

        if use_2_conv_filters:
            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=char_vectors.size(1),
                    out_channels=n_filters,
                    kernel_size=5,
                ),
                nn.ReLU(),
                # nn.Dropout(drop_prob),
                nn.BatchNorm1d(num_features=n_filters), # some people claimed it helped them.
                nn.AdaptiveMaxPool1d(1), #We want to max pool on last dimension (i.e. over ther char sequence, along the width of the word) and we want to pick 1 max value.
            )

    def forward(self, x):
        #ToDo: Are Reshapes costly operations? Looksmlike they copy objects, keeping old ones around. maybe we should use Views?

        (batch_size, seq_len, word_len) = x.shape # (batch_size, seq_len, word_len)

        y = x.reshape(x.shape[0]*x.shape[1], -1) # (batch_size * seq_len, word_len)

        emb = self.char_embed(y)   # (batch_size * seq_len, word_len, char_embed_size)
        assert (emb.shape == (batch_size*seq_len, word_len, 64))

        emb = F.dropout(emb, self.drop_prob, self.training)

        emb = torch.transpose(emb, 1, 2)  # (batch_size * seq_len, char_embed_size, word_len)

        emb1: torch.Tensor = self.conv1(emb)   # (batch_size * seq_len, out_channels=50, 1)
        # Step 1: Conv1d filters: shape: (batch_size * seq_len, out_channels=50, word_len-4) as filter width is 5, with 1 stride, so that dimesion will be word_len-4
        # Step 2: MaxPool1D across last dimension. so shape will be: (batch_size * seq_len, out_channels=50, 1)


        emb1.squeeze(-1) # (batch_size * seq_len, out_channels=50)
        emb1 = emb1.reshape(x.shape[0], x.shape[1], -1) # (batch_size, seq_len, out_channels=50)
        assert(emb1.shape == ((batch_size, seq_len, self.n_filters)))

        if self.use_2_conv_filters:
            emb2: torch.Tensor = self.conv2(emb)
            emb2.squeeze(-1) # (batch_size * seq_len, out_channels=50)
            emb2 = emb2.reshape(x.shape[0], x.shape[1], -1) # (batch_size, seq_len, out_channels=50)
            assert(emb2.shape == ((batch_size, seq_len, self.n_filters)))

            emb = torch.cat((emb1, emb2), dim = 2)
        else:
            emb = emb1

        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class CoAttention(nn.Module):
    """Dynamic co=attention.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.linear = nn.Linear(hidden_size, hidden_size)
        # self.csentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.qsentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        # for weight in (self.csentinel, self.qsentinel):
        #     nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        # BiDAF stuff
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # Coattention stuff
        qprime = torch.tanh(self.linear(q))
        scoat = torch.matmul(c , qprime.transpose(1, 2))
        scoat1 = masked_softmax(scoat, q_mask, dim=2)       # (batch_size, c_len, q_len)
        scoat2 = masked_softmax(scoat, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # BiDAF stuff
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        # Co-attention stuff
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        acoat = torch.bmm(scoat1, qprime)
        # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
        bcoat = torch.bmm(scoat2.transpose(1, 2), c)
        scoat = torch.bmm(scoat1, bcoat)

        # Merge BiDAF and Coattention
        x = torch.cat([c, a, c * a, c * b, scoat, acoat], dim=2)  # (bs, c_len, 6 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class CoAttentionOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(12 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(12 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class CoAttentionV2(nn.Module):
    """Dynamic co=attention.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.csentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        # self.qsentinel = nn.Parameter(torch.zeros(hidden_size, 1))
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        # for weight in (self.csentinel, self.qsentinel):
        #     nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        # BiDAF stuff
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # Coattention stuff
        qprime1 = F.ReLU(self.linear1(q))
        qprime = F.ReLU(self.linear2(qprime1))
        scoat = torch.matmul(c , qprime.transpose(1, 2))
        scoat1 = masked_softmax(scoat, q_mask, dim=2)       # (batch_size, c_len, q_len)
        scoat2 = masked_softmax(scoat, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # BiDAF stuff
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        # Co-attention stuff
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        acoat = torch.bmm(scoat1, qprime)
        # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
        bcoat = torch.bmm(scoat2.transpose(1, 2), c)
        scoat = torch.bmm(scoat1, bcoat)

        # Merge BiDAF and Coattention
        x = torch.cat([c, a, c * a, c * b, scoat, acoat, scoat*acoat], dim=2)  # (bs, c_len, 6 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class CoAttentionOutputV2(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(16 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(16 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class SelfAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.p_weight1 = nn.Parameter(torch.zeros(4*hidden_size, int(np.sqrt(hidden_size))))
        self.p_weight2 = nn.Parameter(torch.zeros(4*hidden_size, int(np.sqrt(hidden_size))))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        for weight in (self.p_weight1, self.p_weight2):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        ss = self.get_self_similarity_matrix(x) # (bs, c_len, c_len)
        ss1 = masked_softmax(ss, c_mask, dim=1)
        patt = torch.bmm(ss1, b)

        return patt

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

    def get_self_similarity_matrix(self, b):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        b_len= b.size(1)
        b = F.dropout(b, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        #q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(b, self.p_weight1) # (bs, c_len, sqrt(hidden_size))
        s1 = torch.matmul(b, self.p_weight2) # (bs, c_len, sqrt(hidden_size))
        s = torch.matmul(s0, s1.transpose(1, 2)) # (bs, c_len, c_len)

        return s


class SelfAttentionOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

class Multihead_Attention(nn.Module):
    """
    """

    def __init__(self, hidden_dim, input_dim=None, num_heads=1, drop_prob=0.2, cross = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_Q = nn.Linear(input_dim, hidden_dim)
        self.linear_K = nn.Linear(input_dim, hidden_dim)
        self.linear_V = nn.Linear(input_dim, input_dim)
        self.num_heads = num_heads
        self.head_size = self.hidden_dim//self.num_heads
        self.head_size_v = input_dim//self.num_heads
        self.drop_prob = drop_prob
        self.cross = cross

    def forward(self, Context, Question=None, context_mask=None, question_mask=None):
        """
        Q: N, T_q, C_q
        K: N, T_k, C_k
        V: N, T_v, C_v
        :return:
        """
        N = Context.size()[0]  # batch size
        if self.cross: # cross-attention
            # print("Context.shape = ", Context.shape, ", Question.shape = ", Question.shape)
            # Linear layer, project to hidden_dim, hidden_dim can be smaller than embedding size
            # Q stands for query, K stands for key, V stands for value
            # Context becomes Q, Question becomes key and value
            Qr = nn.ReLU()(self.linear_Q(Context))
            Kr = nn.ReLU()(self.linear_K(Question))
            Vr = nn.ReLU()(self.linear_V(Question))

            # Split into heads along "embedding"-dimension
            Qpieces = Qr.split(split_size=self.head_size, dim=2) # split the embedding dimension into heads
            Kpieces = Kr.split(split_size=self.head_size, dim=2)
            Vpieces = Vr.split(split_size=self.head_size_v, dim=2)

            # concatenate the pieces/heads along "batch"-dimension
            Qbatched = torch.cat(Qpieces, dim=0)  # (heads*N, T_q, head_size)
            Kbatched = torch.cat(Kpieces, dim=0)  # (heads*N, T_k, head_size)
            Vbatched = torch.cat(Vpieces, dim=0)  # (heads*N, T_v, head_size)
            # print("Qbatched.shape, Kbatched.shape, Vbatched.shape = ", Qbatched.shape, Kbatched.shape, Vbatched.shape)

            # Multiplicative similarity
            outputs = torch.bmm(Qbatched, Kbatched.transpose(2, 1))

            # Scale-normalization
            outputs = outputs / (Kbatched.size()[-1] ** 0.5)

            # Key, Query Masking
            # print("context_mask, question_mask = ", context_mask.shape, question_mask.shape)
            # q_mask = torch.unsqueeze(context_mask, -1)
            # k_mask = torch.unsqueeze(question_mask, -2)
            q_mask, k_mask = context_mask.type(torch.float32), question_mask.type(torch.float32)
            # print("q_mask, k_mask = ", q_mask.shape, k_mask.shape)
            masks = torch.bmm(q_mask, k_mask)  # (N, T_q, T_k), torch.bmm doesn't work for bool
            masks = masks.repeat(self.num_heads, 1, 1)  # (heads*N, T_q, T_k)
            # print("masks.shape = ", masks.shape)
            # print("outputs.shape = ", outputs.shape)

            similarity = masks * outputs + (1 - masks) * -1e30
            # print("similarity.shape = ", similarity.shape)

            # Activation
            context_weights = nn.Softmax(dim=2)(similarity)  # (h*N, T_q, T_k)
            question_weights = nn.Softmax(dim=1)(similarity)
            # print("context_weights.shape, question_weights.shape = ", context_weights.shape, question_weights.shape)

            # Dropouts
            context_weights = F.dropout(context_weights, self.drop_prob, self.training) #F.dropout(c, self.drop_prob, self.training)
            question_weights = F.dropout(question_weights, self.drop_prob, self.training)

            # Weighted sum
            context_att = torch.bmm(context_weights, Vbatched)  # ( h*N, T_q, C/h)
            question_att = torch.bmm(question_weights.transpose(-2, -1), Qbatched)
            context_coatt = torch.bmm(context_weights, question_att)
            # print("context_att.shape, question_att.shape = ", context_att.shape, question_att.shape)

            # Restore shape
            context_att = context_att.split(N, dim=0)  # (N, T_q, C)
            context_att = torch.cat(context_att, dim=2)
            question_att = question_att.split(N, dim=0)  # (N, T_q, C)
            question_att = torch.cat(question_att, dim=2)
            context_coatt = context_coatt.split(N, dim=0)  # (N, T_q, C)
            context_coatt = torch.cat(context_coatt, dim=2)
            # print("context_att.shape, question_att.shape, context_coatt.shape = ", context_att.shape, question_att.shape, context_coatt.shape)

            return context_att, question_att, context_coatt

        else: # self-attention

            # Linear layer, project to hidden_dim, hidden_dim can be smaller than embedding size
            # Q stands for query, K stands for key, V stands for value
            # Context becomes Q, Question becomes key and value
            Qr = nn.ReLU()(self.linear_Q(Context))
            Kr = nn.ReLU()(self.linear_K(Context))
            Vr = nn.ReLU()(self.linear_V(Context))

            # Split into heads along "embedding"-dimension
            Qpieces = Qr.split(split_size=self.head_size, dim=2) # split the embedding dimension into heads
            Kpieces = Kr.split(split_size=self.head_size, dim=2)
            Vpieces = Vr.split(split_size=self.head_size_v, dim=2)

            # concatenate the pieces/heads along "batch"-dimension
            Qbatched = torch.cat(Qpieces, dim=0)  # (heads*N, T_c, head_size)
            Kbatched = torch.cat(Kpieces, dim=0)  # (heads*N, T_c, head_size)
            Vbatched = torch.cat(Vpieces, dim=0)  # (heads*N, T_c, head_size)
            # print("Qbatched.shape, Kbatched.shape, Vbatched.shape = ", Qbatched.shape, Kbatched.shape, Vbatched.shape)

            # Multiplicative similarity
            outputs = torch.bmm(Qbatched, Kbatched.transpose(2, 1)) # (heads*N, T_c, T_c)
            # print("outputs.shape = ", outputs.shape)

            # Scale-normalization
            outputs = outputs / (Kbatched.size()[-1] ** 0.5)

            # Key, Query Masking
            T_c = context_mask.shape[1]
            #context_mask = torch.unsqueeze(context_mask, -1)
            context_mask = context_mask.type(torch.float32)
            context_mask = context_mask.repeat(self.num_heads, 1, T_c)
            # print("context_mask.shape, outputs.shape = ", context_mask.shape, outputs.shape)
            similarity = context_mask * outputs + (1 - context_mask) * -1e30
            # print("similarity.shape = ", similarity.shape)

            # Activation
            context_weights = nn.Softmax(dim=2)(similarity)  # (h*N, T_q, T_k)

            # Dropouts
            context_weights = F.dropout(context_weights, self.drop_prob, self.training) #F.dropout(c, self.drop_prob, self.training)

            # Weighted sum
            context_att = torch.bmm(context_weights, Vbatched)  # ( h*N, T_q, C/h)

            # Restore shape
            context_att = context_att.split(N, dim=0)  # (N, T_q, C)
            context_att = torch.cat(context_att, dim=2)

            return context_att


class Attention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1, use_self_attention=False, use_multihead=False, multihead_count = 4):
        super().__init__()
        self.drop_prob = drop_prob
        self.use_self_attention = use_self_attention
        self.use_multihead = use_multihead
        # self.linear1 = nn.Linear(hidden_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.p_weight1 = nn.Parameter(torch.zeros(6*hidden_size, int(np.sqrt(hidden_size))))
        self.p_weight2 = nn.Parameter(torch.zeros(6*hidden_size, int(np.sqrt(hidden_size))))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.bias = nn.Parameter(torch.zeros(1))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)

        if use_multihead:
            self.multihead_attn = Multihead_Attention(hidden_size, hidden_size, multihead_count, drop_prob=drop_prob, cross=1) # hidden_dim, input_dim=None, num_heads=1, drop_prob=0.2, cross = 1
            if use_self_attention:
                self.multihead_self = Multihead_Attention(hidden_size, 6*hidden_size, multihead_count, drop_prob=drop_prob, cross=0)
        else:
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)

        # self.pbias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)

        s = self.get_similarity_matrix(c, q)       # (batch_size, c_len, q_len)

        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        # Co-attention stuff
        #qprime1 = F.relu(self.linear1(q))
        #qprime = F.relu(self.linear2(qprime1))
        #scoat = torch.matmul(c , qprime.transpose(1, 2))
        if self.use_multihead:
            acoat, bcoat, scoat3 = self.multihead_attn(c, q, c_mask, q_mask) # Context, Question=None, context_mask=None, question_mask=None -> similarity, context_weights, question_weights, context_att, question_att
        else:
            qprime1 = F.ReLU(self.linear1(q))
            qprime = F.ReLU(self.linear2(qprime1))
            scoat = torch.matmul(c , qprime.transpose(1, 2))
            scoat1 = masked_softmax(scoat, q_mask, dim=2)       # (batch_size, c_len, q_len)
            scoat2 = masked_softmax(scoat, c_mask, dim=1)
            acoat = torch.bmm(scoat1, qprime)
        # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
            bcoat = torch.bmm(scoat2.transpose(1, 2), c)
            scoat3 = torch.bmm(scoat1, bcoat)
        # scoat1 = masked_softmax(scoat, q_mask, dim=2)       # (batch_size, c_len, q_len)
        # scoat2 = masked_softmax(scoat, c_mask, dim=1)       # (batch_size, c_len, q_len)
        # # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        # acoat = torch.bmm(scoat1, q)
        # # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
        # bcoat = torch.bmm(scoat2.transpose(1, 2), c)
        # scoat3 = torch.bmm(scoat1, bcoat)
        # Co-attention stuff
        #qprime1 = F.relu(self.linear1(q))
        #qprime = F.relu(self.linear2(qprime1))
        #scoat = torch.matmul(c , qprime.transpose(1, 2))
        #scoat1 = masked_softmax(scoat, q_mask, dim=2)       # (batch_size, c_len, q_len)
        #scoat2 = masked_softmax(scoat, c_mask, dim=1)       # (batch_size, c_len, q_len)
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        acoat = torch.bmm(scoat1, qprime)
        # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
        bcoat = torch.bmm(scoat2.transpose(1, 2), c)
        scoat = torch.bmm(scoat1, bcoat)

        # BiDAF
        # print("c.shape, a.shape, scoat3.shape, acoat.shape = ", c.shape, a.shape, scoat3.shape, acoat.shape)
        x = torch.cat([c, a, c * a, c * b, scoat3, acoat], dim=2)  # (bs, c_len, 4 * hid_size) torch.cat([c, a, c * a, c * b, scoat3, acoat], dim=2)  # (bs, c_len, 6 * hid_size)

        # self attention
        # ss = self.get_self_similarity_matrix(x) # (bs, c_len, c_len)
        # ss1 = masked_softmax(ss, c_mask, dim=1)
        # patt = torch.bmm(ss1, x)
        if self.use_self_attention:
            patt = self.multihead_self(x, context_mask = c_mask) + x
            return patt
        else:
            return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

    # def get_self_similarity_matrix(self, b):
    #     """Get the "similarity matrix" between context and query (using the
    #     terminology of the BiDAF paper).
    #
    #     A naive implementation as described in BiDAF would concatenate the
    #     three vectors then project the result with a single weight matrix. This
    #     method is a more memory-efficient implementation of the same operation.
    #
    #     See Also:
    #         Equation 1 in https://arxiv.org/abs/1611.01603
    #     """
    #     b_len= b.size(1)
    #     b = F.dropout(b, self.drop_prob, self.training)  # (bs, c_len, hid_size)
    #     #q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)
    #
    #     # Shapes: (batch_size, c_len, q_len)
    #     s0a = torch.matmul(b, self.p_weight1) # (bs, c_len, sqrt(hidden_size))
    #     s1a = torch.matmul(b, self.p_weight2) # (bs, c_len, sqrt(hidden_size))
    #     sa = torch.matmul(s0a, s1a.transpose(1, 2)) # (bs, c_len, c_len)
    #
    #     s0 = torch.matmul(b, self.p_weight1).expand([-1, -1, b_len])
    #     s1 = torch.matmul(b, self.p_weight2).transpose(1, 2)\
    #                                        .expand([-1, b_len, -1])
    #     s2 = torch.matmul(b * self.p2_weight, b.transpose(1, 2))
    #     s = s0 + s1 + s2 + self.pbias + sa
    #     return s


class AttentionOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super().__init__()
        self.att_linear_1 = nn.Linear(12 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(12 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class HighwayMaxoutNetwork(nn.Module):
    """HMN network for dynamic decoder.

    Based on the Co-attention paper:

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, mod_out_size, hidden_size, max_out_pool_size):
        super(HighwayMaxoutNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.maxout_pool_size = max_out_pool_size

        print(mod_out_size, hidden_size, max_out_pool_size)

        self.r = nn.Linear(2 * mod_out_size + hidden_size, hidden_size, bias=False)

        self.W1 = nn.Linear(mod_out_size + hidden_size, max_out_pool_size * hidden_size)

        self.W2 = nn.Linear(hidden_size, max_out_pool_size * hidden_size)

        #self.dropout_m_t_2 = nn.Dropout(p=dropout_ratio)

        self.W3 = nn.Linear(2 * hidden_size, max_out_pool_size)


    def forward(self, mod, h_i, u_s_prev, u_e_prev, mask):
        # mod       (batchSize, seqlen, attention_encoding_size)
        # u_s_prev  (batch_size, mod_out_size)
        # u_e_prev  (batch_size, mod_out_size)
        # h_i       (batch_size, self.hidden_size)

        (batch_size, seq_len, mod_out_size) = mod.shape # (batchSize, seqlen, mod_out_size)

        r = F.tanh(self.r(torch.cat((h_i, u_s_prev, u_e_prev), 1)))  # (batch_size, hidden_size)
        r_expanded = r.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size).contiguous()  # (batch_size, seq_len, hidden_size)

        W1_inp = torch.cat((mod, r_expanded), 2)  # (batch_size, seq_len, hidden_size + mod_out_size)

        # Max Pooling activation. (mix of ReLU and Leaku ReLU)
        # we are doing MLP maxpooling. We could have also done, CONV max pooling.
        # https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb

        m_t_1 = self.W1(W1_inp) # (batch_size, seq_len, hidden_size * pool_size)
        m_t_1 = m_t_1.view(batch_size, seq_len, self.maxout_pool_size, self.hidden_size) # (batch_size, seq_len, pool_size, hidden_size)
        m_t_1, _ = m_t_1.max(2) # (batch_size, seq_len, hidden_size)

        assert(m_t_1.shape == (batch_size, seq_len, self.hidden_size))

        m_t_2 = self.W2(m_t_1)  # (batch_size, seq_len, pool_size * hidden_size)
        m_t_2 = m_t_2.view(batch_size, seq_len, self.maxout_pool_size, self.hidden_size) # (batch_size, seq_len, pool_size, hidden_size)
        m_t_2, _ = m_t_2.max(2)  # (batch_size, seq_len, hidden_size)

        alpha_in = torch.cat((m_t_1, m_t_2), 2)  # (batch_size, seq_len, 2* hidden_size)
        alpha = self.W3(alpha_in)  #  (batch_size, seq_len, pool_size)
        logits, _ = alpha.max(2)  # (batch_size, seq_len)

        log_p = masked_softmax(logits, mask, log_softmax=True)

        return log_p

class IterativeDecoderOutput(nn.Module):
    """Output layer used by Coattention paper.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, att_out_dim, mod_out_dim, max_decode_steps, maxout_pool_size, older_out_layer, drop_prob):
        super().__init__()
        self.max_decode_steps = max_decode_steps
        self.att_out_dim = att_out_dim
        self.hidden_size = hidden_size
        self.mod_out_dim = mod_out_dim

        ## Either enable or disable this.
        self.fuse_att_mod = False

        if self.fuse_att_mod:
            self.att_mod_proj = nn.Linear(self.mod_out_dim + self.att_out_dim, self.mod_out_dim)
            # self.mod_out_dim = self.mod_out_dim + self.att_out_dim

        # input to RNN will be: [u_s_i-1 ; u_e_i-1] 
        # self.decoder = RNNEncoder(2 * mod_out_dim, hidden_size, 1, drop_prob=drop_prob, bidirectional=False)
        self.decoder = nn.LSTMCell(2 * mod_out_dim, hidden_size, bias=True)

        # some people forget the biases, and initialize lstm with that.
        # see if its required.

        self.HMN_start = HighwayMaxoutNetwork(self.mod_out_dim, hidden_size, maxout_pool_size)
        self.HMN_end = HighwayMaxoutNetwork(self.mod_out_dim, hidden_size, maxout_pool_size)

    def forward(self, att, mod, mask):

        # att:  (batch_size, seq_len, att_enc_size)
        # mod:  (batch_size, seq_len, mod_out_size)
        # mask: (batch_size, seq_len)
        
        # if self.fuse_att_mod:
        #     mod = torch.cat((mod, att), dim=2)

        (batch_size, seq_len, _) = mod.shape

        log_p1, log_p2 = self.olderDecoder(att, mod, mask)

        if self.fuse_att_mod:
            mod = self.att_mod_proj(torch.cat((mod, att), dim=2))

        _, s_prev = torch.max(log_p1, dim = 1)
        _, e_prev = torch.max(log_p2, dim = 1)

        # how to initialize s_prev, e_prev? Its not mentioned in paper.
        # s_prev = torch.zeros(batch_size, ).long() # (batch_size, )
        # e_prev = torch.sum(mask, 1) - 1           # (batch_size, )
        dec_state_i = None

        assert(s_prev.shape == (batch_size, ))
        assert(e_prev.shape == (batch_size, ))

        batch_idxs = torch.arange(0, batch_size, out=torch.LongTensor(batch_size))

        log_p1s = None # (batch_size, n_iterations, seq_len, )
        log_p2s = None # (batch_size, n_iterations, seq_len, )

        for _ in range(self.max_decode_steps):
            u_s_prev = mod[batch_idxs, s_prev, :]  #  (batch_size, mod_out_size)
            u_e_prev = mod[batch_idxs, e_prev, :]  #  (batch_size, mod_out_size)

            u_cat = torch.cat((u_s_prev, u_e_prev), 1)  # (batch_size, 2 * mod_out_size)

            h_i, c_i = self.decoder(u_cat, dec_state_i) # Each h_i and c_i (batch_size, self.hidden_size)
            dec_state_i = (h_i, c_i)

            assert(h_i.shape == (batch_size, self.hidden_size))

            s_prev_probs = self.HMN_start(mod, h_i, u_s_prev, u_e_prev, mask) # (batch_size, seq_len)
            _, s_prev = torch.max(s_prev_probs, dim=1)
            s_prev_probs = s_prev_probs.unsqueeze(1) # (batch_size, 1, seq_len)

            # print(s_prev)
            u_s_prev = mod[batch_idxs, s_prev, :]  #  (batch_size, mod_out_size)

            e_prev_probs = self.HMN_end(mod, h_i, u_s_prev, u_e_prev, mask)
            _, e_prev = torch.max(e_prev_probs, dim=1)
            e_prev_probs = e_prev_probs.unsqueeze(1) # (batch_size, 1, seq_len)

            if log_p1s == None and log_p2s == None:
                log_p1s = s_prev_probs
                log_p2s = e_prev_probs
            else:
                log_p1s = torch.cat((log_p1s, s_prev_probs), dim=1)
                log_p2s = torch.cat((log_p2s, e_prev_probs), dim=1)

        assert(log_p1s.shape == (batch_size, self.max_decode_steps, seq_len))
        assert(log_p2s.shape == (batch_size, self.max_decode_steps, seq_len))

        # We can just return the final probabilities.
        # But the paper is doing commulative losses for all probs.
        # Also, we are not stopping if next prediction is same as current prediction, so we may penalize extra for such cases.
        # we can do something smart about this ^^ , when calculating commulative loss in train.py, maybe??
        return log_p1s, log_p2s
