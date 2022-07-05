import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_att = nn.Linear(config.hidden_size, config.hidden_size)
        self.keys = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_att = nn.Linear(config.hidden_size, config.hidden_size)
        self.values = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Like MLP mixer model (since we have a point-wise multiply)
        self.query_att.weight.data.fill_(0)
        self.query_att.bias.data.fill_(1)
        self.key_att.weight.data.fill_(0)
        self.key_att.bias.data.fill_(1)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attn_norm(hidden_states)
        
        batch_size, seq_len, d_model = hidden_states.shape
        query = self.query(hidden_states)
        keys = self.keys(hidden_states)
        values = self.values(hidden_states)
        
        # equation (3) manual (causal) softmax
        query_weight = torch.exp(self.query_att(query) / d_model**.5)
        query_weight = query_weight / torch.cumsum(query_weight, dim = 1)
        
        # equation (4)
        pooled_query = torch.cumsum(query_weight * query, dim = 1)
        
        # corresponds to "p_i = q * k_i" in paper
        mixed_keys = pooled_query * keys
        
        # equation (5) manual (causal) softmax
        keys_weight = torch.exp(self.key_att(mixed_keys) / d_model**.5)
        keys_weight = keys_weight / torch.cumsum(keys_weight, dim = 1)
        
        # equation (6)
        pooled_keys = torch.cumsum(keys_weight * mixed_keys, dim = 1)
        
        # corresponds to "u_i = k * v_i" in paper
        weighted_values = pooled_keys * values
        
        # dropout last like megatron
        weighted_values = self.attn_dropout(weighted_values)
      
        return weighted_values
    
    
class CausalConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, groups):
        super().__init__()
        self.kernel_size = kernel_size
        self.convolutional_layer = nn.Conv1d(hidden_size, hidden_size, groups = groups,
                                             kernel_size = kernel_size, padding = 0)
        
    def forward(self, hidden_states):
        # batch len, seq len, embedding -> batch len, embedding, seq len
        mod = hidden_states.permute(0, 2, 1)
        
        # padding for casual convolution
        mod = F.pad(mod, pad=(self.kernel_size-1, 0), mode='constant', value=0)
        mod = self.convolutional_layer(mod)
        
        # unpermute
        mod = mod.permute(0, 2, 1)
        
        return mod
    
    
class ReducedFastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        if config.convolve is True:
            self.keys = CausalConvolution(config.hidden_size, config.kernel_size, config.groups)
            self.key_att = CausalConvolution(config.hidden_size, config.kernel_size, config.groups)
            self.values = CausalConvolution(config.hidden_size, config.kernel_size, config.groups)
        else:
            self.keys = nn.Linear(config.hidden_size, config.hidden_size)
            self.key_att = nn.Linear(config.hidden_size, config.hidden_size)
            self.values = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attn_norm(hidden_states)
        
        batch_size, seq_len, d_model = hidden_states.shape
        keys = self.keys(hidden_states)
        values = self.values(hidden_states)
        
        # equation (5) manual (causal) softmax
        keys_weight = torch.exp(self.key_att(keys) / d_model**.5)
        keys_weight = keys_weight / torch.cumsum(keys_weight, dim = 1)
        
        # equation (6)
        pooled_keys = torch.cumsum(keys_weight * keys, dim = 1)
        
        # corresponds to "u_i = k * v_i" in paper
        weighted_values = pooled_keys * values
        
        # dropout last like megatron
        weighted_values = self.attn_dropout(weighted_values)
      
        return weighted_values


class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        
        self.convolve = config.convolve
        
        if config.reduced_attention is True:
            self.attention = ReducedFastSelfAttention(config)
        else:
            self.attention = FastSelfAttention(config)

        self.boom = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.gelu = nn.GELU()
        self.unboom = nn.Linear(4*config.hidden_size, config.hidden_size)
        self.boom_norm = nn.LayerNorm(config.hidden_size)
        self.boom_drop = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask):
        mod = hidden_states

        mod = mod + self.attention(mod, attention_mask)
        
        mod = mod + self.__boom(mod)
        
        return mod

    def __boom(self, hidden_states):
        mod = self.boom_norm(hidden_states)
        mod = self.boom(mod)
        mod = self.gelu(mod)
        mod = self.unboom(mod)
        
        # possible parameter saving like SHA-RNN
        # mod = torch.stack(mod.chunk(4, dim = -1), dim = -1).sum(dim = -1)

        if self.convolve is True:
            mod = self.boom_drop(mod)

        return mod


# adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class FastformerDecoder(nn.Module):
    def __init__(self, config):
        super(FastformerDecoder, self).__init__()
        self.config = config
        self.decoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = PositionalEncoding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.proj_logits = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        position_embeddings = self.position_embeddings(input_embs)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        
        if self.config.convolve:
            embeddings = self.dropout(embeddings)
        
        layer_outputs = embeddings
        for i, layer_module in enumerate(self.decoders):
            layer_outputs = layer_module(layer_outputs, extended_attention_mask)

        return self.proj_logits(layer_outputs)
    

class Accumulator(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size,config.hidden_size,padding_idx=0)
        self.fastformer_model = FastformerDecoder(config)
        self.criterion = nn.CrossEntropyLoss(label_smoothing = .1)
    
    def forward(self, input_ids, labels, attention_mask):
        embds=self.word_embedding(input_ids)
        logits = self.fastformer_model(embds, attention_mask)
        loss = self.criterion(logits.view(-1, self.config.vocab_size), labels.view(-1)) 
        return loss, logits
