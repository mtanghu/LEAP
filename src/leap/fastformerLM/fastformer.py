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

        # only considering single head for now
        self.Wquery = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.query_attn = nn.Parameter(torch.randn(self.n_heads, self.head_size) * config.initializer_range)
        self.Wkeys = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.key_attn = nn.Parameter(torch.randn(self.n_heads, self.head_size) * config.initializer_range)
        self.Wvalues = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, mod, attention_mask):
        mod = self.attn_norm(mod)

        query = self.Wquery(mod)
        keys = self.Wkeys(mod)
        values = self.Wvalues(mod)
        
        # equations (3-4)
        pooled_query = self.__additive_attn(query, self.query_attn, attention_mask)
        pooled_query = self.attn_dropout(pooled_query)
        
        # corresponds to "p_i = q * k_i" in paper
        mixed_keys = pooled_query * keys
        
        # equations (5-6)
        pooled_keys = self.__additive_attn(mixed_keys, self.key_attn, attention_mask)
        pooled_keys = self.attn_dropout(pooled_keys)
        
        # corresponds to "u_i = k * v_i" in paper
        weighted_values = pooled_keys * values

        # residual here because of 2 rounds of Additive Attention
        return weighted_values
    
    
    def __additive_attn(self, x, learned_vector, attention_mask):
        '''This function implements equations 3-4 which are the same as equations 5-6'''
        
        batch_size, seq_len, hidden_size = x.shape
        
        # reshape for multihead attention
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        
        # norm for stability (as a part of strong scaling)
        x = self.__real_norm(x)
        learned_vector = self.__real_norm(learned_vector)
                
        # manual down projection for speed (instead of using einsum)
        attn = (x * learned_vector).sum(dim = -1)
        
         # strong scaling so that the max attention score is 7 (because of previous norming)
        attn = (attn / self.head_size) * 7
        
        # masking out pad tokens
        attn += attention_mask
        
        # manual softmax and cumulative sum steps (equations 5/10 in github)
        attn = torch.exp(attn)
        attn = attn.unsqueeze(3)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        
        ## windowed cumsum (equation 10 in github)
        s = torch.cumsum(attn * x, dim = 1)
        s = s - self.__window_align(s)
        z = torch.cumsum(attn, dim = 1)
        z = z - self.__window_align(z)
        
        # finish softmax by dividing by normalization term z
        g = s / z
        
        # concat everything back together
        g = g.reshape(batch_size, seq_len, hidden_size)
        
        return g


    def __real_norm(self, x):
        # normalize x on the last dimension (the head dimension in this case) with eps term for stability
        return (x - x.mean(dim = -1).unsqueeze(-1)) / (x.std(dim = -1).unsqueeze(-1) + 1e-5)


    def __window_align(self, x):
        seq_len = x.shape[1]
        
        # zero out the last window_size vectors, and roll these vectors to the front
        # thus, at every sequence index will contain the "past" cumuluative sum to subtract away
        clone_x = torch.clone(x)
        clone_x[:,-self.window_size:] = 0
        clone_x = torch.roll(clone_x, self.window_size, 1)
        
        return clone_x



class FastformerLayer(nn.Module):
    def __init__(self, config, window_size):
        super(FastformerLayer, self).__init__()
        self.attention = FastSelfAttention(config, window_size)

        self.boom = nn.Linear(config.hidden_size, config.hidden_size * 4, bias = False)
        self.activation = nn.GELU()
        self.unboom = nn.Linear(4 * config.hidden_size, config.hidden_size, bias = False)
        self.boom_norm = nn.LayerNorm(config.hidden_size)
        self.boom_drop = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, mod, attention_mask):
        # no residual here as it is handled by the the FastSelfAttention
        mod = mod + self.attention(mod, attention_mask)
        
        mod = mod + self.__boom(mod)
        
        return mod


    def __boom(self, mod):
        mod = self.boom_norm(mod)
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
        self.last_norm = nn.LayerNorm(config.hidden_size)
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
        
        embds = self.word_embedding(input_ids)
        layer_outputs = self.fastformer_model(embds, attention_mask)
        layer_outputs = self.last_norm(layer_outputs)
        logits = self.proj_logits(layer_outputs)
        
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