import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class OldFastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.query_att = nn.Linear(self.input_dim, self.input_dim)
        self.keys = nn.Linear(self.input_dim, self.input_dim)
        self.key_att = nn.Linear(self.input_dim, self.input_dim)
        self.values = nn.Linear(self.input_dim, self.input_dim)
        
        self.attn_norm = nn.LayerNorm(self.input_dim)
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
        scaling_vec = torch.arange(1, seq_len+1, device=hidden_states.device, requires_grad=False).reshape(1, -1, 1)
        
        # equation (3)
        query_weight = F.softmax(self.query_att(query) / d_model**.5, dim = -1)
        
        # equation (4), scaling because of cumulative sum
        pooled_query = torch.cumsum(query_weight * query, dim = 1) / scaling_vec
        
        # corresponds to "p_i = q * k_i" in paper
        mixed_keys = pooled_query * keys
        
        # equation (5)
        keys_weight = F.softmax(self.key_att(mixed_keys) / d_model**.5, dim = -1)
        
        # equation (6)
        pooled_keys = torch.cumsum(keys_weight * mixed_keys, dim = 1) / scaling_vec
        
        # corresponds to "u_i = k * v_i" in paper
        weighted_values = pooled_keys * values
      
        return weighted_values


# removes unnecessary parameters/steps (ie. using softmax attention vectors which make little sense for global attention)
class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.hidden_size
        self.query = nn.Linear(self.input_dim, self.input_dim)
        self.keys = nn.Linear(self.input_dim, self.input_dim)
        self.values = nn.Linear(self.input_dim, self.input_dim)
        
        self.attn_norm = nn.LayerNorm(self.input_dim)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attn_norm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        query = self.query(hidden_states)
        keys = self.keys(hidden_states)
        values = self.values(hidden_states)
        scaling_vec = torch.arange(1, seq_len+1, device=hidden_states.device, requires_grad=False).reshape(1, -1, 1)
        
        pooled_query = torch.cumsum(query, dim = 1) / scaling_vec

        mixed_keys = pooled_query * keys
        pooled_keys = torch.cumsum(mixed_keys, dim = 1) / scaling_vec
        
        weighted_value = pooled_keys * values
      
        return weighted_value


class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.convolve = False
        if config.convolve is True:
            self.convolve = True
            self.kernel_size = config.kernel_size
            self.conv = nn.Conv1d(config.hidden_size, config.hidden_size,
                                  groups = config.groups, kernel_size = self.kernel_size, padding = 0)
            self.conv_norm = nn.LayerNorm(config.hidden_size)
        
        if config.old_attention is True:
            self.attention = OldFastSelfAttention(config)
        else:
            self.attention = FastSelfAttention(config)

        self.boom = nn.Linear(config.hidden_size, config.hidden_size*4)
        self.gelu = nn.GELU()
        self.unboom = nn.Linear(4*config.hidden_size, config.hidden_size)
        self.boom_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states, attention_mask):
        mod = hidden_states
        
        if self.convolve:
            # convolutional attention
            mod = mod + self.__convolve(mod)
        
        # cumsum attention
        mod = mod + self.attention(mod, attention_mask)
        
        mod = mod + self.__boom(mod)
        
        return mod
    
    def __convolve(self, hidden_states):
        hidden_states = self.conv_norm(hidden_states)
        
        # batch len, seq len, embedding -> batch len, embedding, seq len
        batch_len, _, embedding_len = hidden_states.shape
        conv_attention = hidden_states.permute(0, 2, 1)
        
        # padding
        conv_attention = F.pad(conv_attention, pad=(self.kernel_size-1, 0), mode='constant', value=0)
        conv_attention = self.conv(conv_attention)
        
        # unpermute
        conv_attention = conv_attention.permute(0, 2, 1)
        
        conv_attention = self.dropout(conv_attention)
        
        return conv_attention
    
    def __boom(self, hidden_states):
        mod = self.boom_norm(hidden_states)
        mod = self.boom(mod)
        mod = self.gelu(mod)
        mod = self.unboom(mod)
        
        mod = self.dropout(mod)
        
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
        embeddings = self.dropout(embeddings)
        #all_hidden_states = [embeddings]
        
        layer_outputs = embeddings
        for i, layer_module in enumerate(self.decoders):
            layer_outputs = layer_module(layer_outputs, extended_attention_mask)
            #all_hidden_states.append(layer_outputs)
        
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
