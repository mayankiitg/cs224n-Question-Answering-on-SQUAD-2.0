"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

from distutils.command.config import config
import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, use_char_emb, use_dynamic_coattention, use_self_attention, use_attention, use_dynamic_decoder, use_hwy_encoder, use_multihead, multihead_count, drop_prob=0.,use_2_conv_filters = True):
        super(BiDAF, self).__init__()
        print("initializing Bidaf!")
        self.use_dynamic_coattention = use_dynamic_coattention
        self.use_self_attention = use_self_attention
        self.use_attention = use_attention
        self.use_dynamic_decoder = use_dynamic_decoder
        self.use_hwy_encoder = use_hwy_encoder
        self.use_multihead = use_multihead
        self.multihead_count = multihead_count

        att_out_dim = 0
        mod_out_dim = 0

        if use_char_emb:
            print("Using character embeddings")
            self.emb = layers.WordAndCharEmbedding(word_vectors=word_vectors,
                                        char_vectors=char_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob,
                                        use_hwy_encoder = use_hwy_encoder,
                                        use_2_conv_filters = use_2_conv_filters)
        else:
            self.emb = layers.Embedding(word_vectors=word_vectors,
                                        hidden_size=hidden_size,
                                        drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
        if self.use_dynamic_coattention:
            print("Using dynamic coattention!")
            self.att = layers.CoAttention(hidden_size=2 * hidden_size,
                                             drop_prob=drop_prob)

            self.mod = layers.RNNEncoder(input_size=12 * hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

            self.out = layers.CoAttentionOutput(hidden_size=hidden_size,
                                          drop_prob=drop_prob)

            att_out_dim = 12 * hidden_size
            mod_out_dim = 2 * hidden_size
        elif self.use_attention:
            print("Using coattent plus passage self-attention!")
            self.att = layers.Attention(hidden_size=2 * hidden_size,
                                             drop_prob=drop_prob,
                                             use_self_attention = use_self_attention,
                                             use_multihead = use_multihead,
                                             multihead_count = multihead_count)

            self.mod = layers.RNNEncoder(input_size=12 * hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

            self.out = layers.AttentionOutput(hidden_size=hidden_size,
                                          drop_prob=drop_prob)

            att_out_dim = 12 * hidden_size
            mod_out_dim = 2 * hidden_size
        else:
            self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                             drop_prob=drop_prob)

            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

            self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                          drop_prob=drop_prob)

            att_out_dim = 8 * hidden_size
            mod_out_dim = 2 * hidden_size

        # Override out layer, if we want to use, DynamicDecoder.
        if self.use_dynamic_decoder:
            print("Using Dynamic Decoder")
            self.out = layers.IterativeDecoderOutput(hidden_size=hidden_size,
                                                    att_out_dim=att_out_dim,
                                                    mod_out_dim=mod_out_dim,
                                                    max_decode_steps=4,   # Hyper Param
                                                    maxout_pool_size=4,   # Hyper Param
                                                    older_out_layer=self.out,
                                                    drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        # cw_idxs: (max__context_len, )
        # context char size cc_idxs: (max_context_len, max_word_len)
        # question char size qc_idxs: (max_question_len, max_word_len)

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
