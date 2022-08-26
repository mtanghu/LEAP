import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput



class FastSelfAttention(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        assert config.hidden_size % self.n_heads == 0, "hidden_size is not divisible by n_heads"
        self.head_size = config.hidden_size // self.n_heads
        self.window_size = window_size
                
        self.attn_vec = nn.Parameter(torch.zeros(config.hidden_size))
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.Wv = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.mid_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attn_norm(hidden_states)
        
        # additive attention
        pre_attn = self.attn_proj(hidden_states)
        attended = self.__additive_attn(hidden_states, self.attn_vec, attention_mask)
        attended = self.attn_dropout(attended)
        
        queries = self.Wq(hidden_states)
        keys = self.Wk(attended)
        values = self.Wv(attended)

        weighted_values = self.__self_attn(queries, keys, values)
        weighted_values = self.attn_dropout(weighted_values)
        
        return weighted_values


    def __additive_attn(self, x, learned_vector, attention_mask):
        '''This function implements equations 3-4 which are the same as equations 5-6'''
        
        batch_size, seq_len, hidden_size = x.shape
                
        # manual dot product/down projection (for multihead attention)
        attn = x * learned_vector
        attn = attn.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        attn = attn.sum(dim = -1) / self.head_size**.5 # scaling
        
        # masking out pad tokens
        attn += attention_mask
        
        # manual softmax and cumulative sum
        attn = torch.exp(attn)
        attn = attn.unsqueeze(3)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        
        ## windowed cumsum
        s = torch.cumsum(attn * x, dim = 1)
        s = s - self.__window_align(s)

        z = torch.cumsum(attn, dim = 1)
        z = z - self.__window_align(z)
        
        g = s / z
        
        # concat heads
        g = g.reshape(batch_size, seq_len, hidden_size)
        
        return g


    def __window_align(self, x):
        seq_len = x.shape[1]
        
        # zero out the last window_size vectors, and roll these vectors to the front
        # thus, at every sequence index will contain the "past" cumuluative sum to subtract away
        clone_x = torch.clone(x)
        clone_x[:,-self.window_size:] = 0
        clone_x = torch.roll(clone_x, self.window_size, 1)
        
        return clone_x


    def __self_attn(self, q, k, v):
        batch_size, seq_len, hidden_size = q.shape

        # reshape for multihead formulation
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        
        # manual "matrix dot product" for speed (in einsum notation "bshe, bshe->bsh") of queries and keys
        similarities = (q * k).sum(dim = -1)
        
        # scaling
        similarities = similarities / self.head_size**.5
        
        # weight the values by similarity (so tokens can recieve only desired information)
        weighted_values = similarities.unsqueeze(3) * v
        
        # concat heads
        return weighted_values.reshape(batch_size, seq_len, hidden_size)



class FastformerLayer(nn.Module):
    def __init__(self, config, window_size):
        super(FastformerLayer, self).__init__()
        self.attention = FastSelfAttention(config, window_size)

        self.boom = nn.Linear(config.hidden_size, config.hidden_size*4, bias = False)
        self.activation = nn.GELU()
        self.unboom = nn.Linear(4*config.hidden_size, config.hidden_size, bias = False)
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
        mod = self.activation(mod)
        mod = self.unboom(mod)
        
        # possible parameter saving like SHA-RNN (seems to slow down training significantly)
        # mod = torch.stack(mod.chunk(4, dim = -1), dim = -1).sum(dim = -1)
        
        mod = self.boom_drop(mod)

        return mod



class FastformerDecoder(nn.Module):
    def __init__(self, config):
        super(FastformerDecoder, self).__init__()
        self.config = config
        self.decoders = nn.ModuleList([FastformerLayer(config, window_size)
                                       for _, window_size in zip(range(config.n_layer), config.window_sizes)])
        self.position_embeddings = nn.Embedding(config.n_positions, config.hidden_size)
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
                 use_local_att = True, window_sizes = None, n_positions = 1024,
                 n_layer = 4, hidden_dropout_prob = .1, initializer_range = .02):
        
        assert not (use_local_att is False and window_sizes is not None), \
            "window sizes set when not using local attention"

        if use_local_att is True and window_sizes is not None:
            assert len(window_sizes) == n_layer, "len(window_sizes) should match # of hidden layers"

        elif use_local_att is True and window_sizes is None:
            window_sizes = [4 * (2**i) for i in range(n_layer)]
            
            # last layer should still be global attention
            window_sizes[-1] = n_positions
        else:
            # don't use windows, i.e. windows are global size
            window_sizes = [n_positions for _ in range(n_layer)]

        super().__init__(
            hidden_size = hidden_size, vocab_size = vocab_size, n_heads = n_heads,
            use_local_att = use_local_att, window_sizes = window_sizes, n_positions = n_positions,
            n_layer = n_layer, hidden_dropout_prob = hidden_dropout_prob,
            initializer_range = initializer_range
        )



class FastformerForCausalLM(PreTrainedModel):
    config_class = FastformerLMConfig
    
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx = 0)
        self.proj_logits = nn.Linear(config.hidden_size, config.vocab_size)
        self.fastformer_model = FastformerDecoder(config)
        self.criterion = nn.CrossEntropyLoss()
        
        # weight tying
        self.proj_logits.weight = self.word_embedding.weight
        
        self.apply(self._init_weights)


    def forward(self, input_ids, attention_mask = None, labels = None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)

        attention_mask = attention_mask.to(dtype = next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.unsqueeze(2)
        
        embds=self.word_embedding(input_ids)
        layer_outputs = self.fastformer_model(embds, attention_mask)
        logits = self.proj_logits(layer_outputs) / self.config.hidden_size**.5
        
        loss = None
        if labels is not None:
            # shift logits the same gpt2 does at
            # https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_gpt2.html
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss = loss, logits = logits)


    def _init_weights(self, module):
        # same initialization as gpt2
        # https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_gpt2.html
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean = 0.0, std = self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



# register config with huggingface
AutoConfig.register("FastformerForCausalLM", FastformerLMConfig)
AutoModelForCausalLM.register(FastformerLMConfig, FastformerForCausalLM)