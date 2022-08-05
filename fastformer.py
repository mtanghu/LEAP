import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

import math



class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        assert config.hidden_size % self.n_heads == 0, "hidden_size is not divisible by n_heads"
        self.head_size = config.hidden_size // self.n_heads
        
        # only considering single head for now
        self.Wquery = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.query_attn = nn.Parameter(torch.zeros(config.hidden_size))
        self.Wkeys = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.key_attn = nn.Parameter(torch.zeros(config.hidden_size))
        self.Wvalues = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        
        self.gelu = nn.GELU()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.mid_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attn_norm(hidden_states)

        query = self.Wquery(hidden_states)
        keys = self.Wkeys(hidden_states)
        values = self.Wvalues(hidden_states)
        
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.unsqueeze(2)
        
        # equations (3-4)
        pooled_query = self.__additive_attn(query, self.query_attn, attention_mask)
        
        # projection after multihead attention like original transformer
        pooled_query = self.attn_dropout(pooled_query)
        pooled_query = self.attn_proj(pooled_query)
        
        # corresponds to "p_i = q * k_i" in paper
        mixed_keys = pooled_query * keys
        
        # residual + norm for extra stability
        residual = hidden_states + mixed_keys
        mixed_keys = self.mid_norm(residual)
        
        # equations (5-6)
        pooled_keys = self.__additive_attn(mixed_keys, self.key_attn, attention_mask)
        
        # no projection here to save parameters
        pooled_keys = self.attn_dropout(pooled_keys)
        
        # corresponds to "u_i = k * v_i" in paper
        weighted_values = pooled_keys * values

        # residual connection here since there are 2 rounds of additive attention
        return residual + weighted_values
    
    
    def __additive_attn(self, x, learned_vector, attention_mask):
        '''This function implements equations 3-4 which are the same as equations 5-6'''
        
        batch_size, seq_len, hidden_size = x.shape
                
        # manual dot product (for multihead attention)
        attn = x * learned_vector
        attn = attn.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        attn = attn.sum(dim = -1) / self.head_size**.5 # scaling
        
        # masking out pad tokens
        attn += attention_mask
        
        # manual softmax and cumulative sum
        attn = torch.exp(attn)
        attn = attn.unsqueeze(3)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        x = torch.cumsum(attn * x, dim = 1) / torch.cumsum(attn, dim = 1)
        
        # concat everything back together
        x = x.reshape(batch_size, seq_len, hidden_size)
        
        return x



class CausalConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, groups, dropout = .1):
        super().__init__()
        self.kernel_size = kernel_size
        self.convolutional_layer = nn.Conv1d(hidden_size, hidden_size, groups = groups,
                                             kernel_size = kernel_size, padding = 0, bias = False)
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.conv_drop = nn.Dropout(dropout)

        
    def forward(self, hidden_states):
        # layer norms still makes sense like "A ConvNet for the 2020s"
        mod = self.conv_norm(hidden_states)
        
        # batch len, seq len, embedding -> batch len, embedding, seq len (conv1d input format)
        mod = mod.permute(0, 2, 1)
        
        # padding to ensure causality
        mod = F.pad(mod, pad=(self.kernel_size-1, 0), mode='constant', value=0)
        mod = self.convolutional_layer(mod)
        
        # unpermute
        mod = mod.permute(0, 2, 1)
        
        mod = self.conv_drop(mod)
        
        return mod



class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        
        self.convolve = config.convolve
        if self.convolve is True:
            self.convolutional_layer = CausalConvolution(
                config.hidden_size, config.kernel_size,
                config.groups, dropout = config.hidden_dropout_prob)
        
        self.attention = FastSelfAttention(config)

        self.boom = nn.Linear(config.hidden_size, config.hidden_size*4, bias = False)
        self.gelu = nn.GELU()
        self.unboom = nn.Linear(4*config.hidden_size, config.hidden_size, bias = False)
        self.boom_norm = nn.LayerNorm(config.hidden_size)
        self.boom_drop = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, attention_mask):
        mod = hidden_states
        
        if self.convolve is True:
            mod = mod + self.convolutional_layer(mod)
        
        # residual connection is handled by attention layer
        mod = self.attention(mod, attention_mask)
        
        mod = mod + self.__boom(mod)
        
        return mod


    def __boom(self, hidden_states):
        mod = self.boom_norm(hidden_states)
        mod = self.boom(mod)
        mod = self.gelu(mod)
        mod = self.unboom(mod)
        
        # possible parameter saving like SHA-RNN (seems to slow down training significantly)
        # mod = torch.stack(mod.chunk(4, dim = -1), dim = -1).sum(dim = -1)
        
        mod = self.boom_drop(mod)

        return mod



class FastformerDecoder(nn.Module):
    def __init__(self, config):
        super(FastformerDecoder, self).__init__()
        self.config = config
        self.decoders = nn.ModuleList([FastformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, 
                input_embs, 
                attention_mask, 
                pooler_index=0):

        batch_size, seq_length, _ = input_embs.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings)
        
        layer_outputs = embeddings
        for i, layer_module in enumerate(self.decoders):
            layer_outputs = layer_module(layer_outputs, attention_mask)

        return layer_outputs
    


# Create configuation compatible with HuggingFace
class FastformerLMConfig(PretrainedConfig):
    model_type = "FastformerForCausalLM"
    def __init__(self, hidden_size = 256, vocab_size = 32100, n_heads = 4,
                 max_position_embeddings = 1024, groups = 1, kernel_size = 4,
                 convolve = False, num_hidden_layers = 4, hidden_dropout_prob = .1,
                 initializer_range = .02, label_smoothing = 0):
        super().__init__(
            hidden_size = hidden_size, vocab_size = vocab_size, n_heads = n_heads,
            max_position_embeddings = max_position_embeddings, groups = groups, kernel_size = kernel_size,
            convolve = convolve, num_hidden_layers = num_hidden_layers, hidden_dropout_prob = hidden_dropout_prob,
            initializer_range = initializer_range, label_smoothing = label_smoothing
        )



class FastformerForCausalLM(PreTrainedModel):
    config_class = FastformerLMConfig
    
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size,config.hidden_size, padding_idx=0)
        self.proj_logits = nn.Linear(config.hidden_size, config.vocab_size)
        self.fastformer_model = FastformerDecoder(config)
        self.criterion = nn.CrossEntropyLoss(label_smoothing = config.label_smoothing)
        self.eval_criterion = nn.CrossEntropyLoss(label_smoothing = 0)
        
        # weight tying
        self.proj_logits.weight = self.word_embedding.weight
        
        self.apply(self._init_weights)


    def forward(self, input_ids, attention_mask=None, labels = None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)
        
        embds=self.word_embedding(input_ids)
        layer_outputs = self.fastformer_model(embds, attention_mask)
        logits = self.proj_logits(layer_outputs) / self.config.hidden_size**.5
        
        loss = None
        if labels is not None:
            # shift logits the same gpt2 does at https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_gpt2.html
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if self.training:
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = self.eval_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss = loss, logits = logits)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



# register config with huggingface
AutoConfig.register("FastformerForCausalLM", FastformerLMConfig)
AutoModelForCausalLM.register(FastformerLMConfig, FastformerForCausalLM)
