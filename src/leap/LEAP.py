import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput



class MultiheadLeap(nn.Module):
    def __init__(self, hidden_size, n_heads, window_size, dropout = .1):
        super(MultiheadLeap, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // n_heads
        self.window_size = window_size

        self.drop = nn.Dropout(dropout)


    def forward(self, q, f1, f2, v, attention_mask = None):        
        batch_size, seq_len, hidden_size = v.shape
        
        # reshape for multihead formulation
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        f1 = f1.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        f2 = f2.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_size)
        
        # normalize focus vectors so focus scores don't explode
        f1 = self.__real_norm(f1)
        f2 = self.__real_norm(f2)
        
        # apply dropout to regularize focus scores (only dropout on one is fine because of the dot product)
        f1 = self.drop(f1)

        # manual "matrix dot product" for speed (in einsum notation "bshe, bshe->bsh") of focus vectors
        focus_scores = (f1 * f2).sum(dim = -1)
        
        # STRONG scaling so that the max focus score is 5
        focus_scores = (focus_scores / self.head_size) * 5
        
        # masking out pad tokens
        if attention_mask is not None:
            focus_scores += attention_mask
        
        # manual softmax within cumulative sum
        focus_scores = torch.exp(focus_scores)
        focus_scores = focus_scores.unsqueeze(-1)
        
        # windowed cumsum 
        s = torch.cumsum(focus_scores * v, dim = 1)
        s = s - self.__window_align(s)
        z = torch.cumsum(focus_scores, dim = 1)
        z = z - self.__window_align(z)
        
        # finish off manual softmax by dividing by normalization term z
        focused_v = s / z
        
        # regularize to stop queries/alignment from overfitting and overfitting reliance on sequence information
        focused_v = self.drop(focused_v)
        
        # querying by measuring dot product alignment (normal scaling is fine here)
        alignment = torch.sigmoid((q * focused_v).sum(dim = -1) / self.head_size**.5)
        focused_v = alignment.unsqueeze(-1) * focused_v
        
        # concat heads
        focused_v = focused_v.reshape(batch_size, seq_len, hidden_size)
        
        return focused_v


    def __real_norm(self, x):
        # normalize x on the last dimension (the head dimension in this case) with eps term for stability
        return (x - x.mean(dim = -1).unsqueeze(-1)) / (x.std(dim = -1).unsqueeze(-1) + 1e-5)


    def __window_align(self, x):
        seq_len = x.shape[1]
        
        # zero out the last window_size vectors, and roll these vectors to the front
        # thus, at every sequence index will contain the "past" cumuluative sum to subtract away
        clone_x = torch.clone(x)
        clone_x[:,-self.window_size:] *= 0
        clone_x = torch.roll(clone_x, self.window_size, dims = 1)
        
        return clone_x



class LeapBlock(nn.Module):
    def __init__(self, config, window_size):
        super(LeapBlock, self).__init__()

        self.pre_norm = nn.LayerNorm(config.hidden_size)
        
        # modules for leap
        self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.Wf1 = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.Wf2 = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.Wv = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.leap = MultiheadLeap(config.hidden_size, config.n_heads, window_size,
                                  dropout = config.hidden_dropout_prob)

        # modules for feedforward layer (aka boom layer)
        self.boom = nn.Linear(config.hidden_size, config.hidden_size*4, bias = False)
        self.activation = nn.GELU()
        self.unboom = nn.Linear(4*config.hidden_size, config.hidden_size, bias = False)
        self.boom_norm = nn.LayerNorm(config.hidden_size)
        self.boom_drop = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, mod, attention_mask):
        # pre-norming
        mod_normed = self.pre_norm(mod)
        
        # transformations so each matrix has its own purpose
        queries = self.Wq(mod_normed)
        focus1 = self.Wf1(mod_normed)
        focus2 = self.Wf2(mod_normed)
        values = self.Wv(mod_normed)
        
        # unnormed residual connection
        mod = mod + self.leap(queries, focus1, focus2, values, attention_mask)
        
        # feedforward layer with pre-norming
        mod_normed = self.pre_norm(mod)
        mod = mod + self.__boom(self.boom_norm(mod_normed))
        
        return mod


    def __boom(self, mod):
        mod = self.boom(mod)
        mod = self.activation(mod)
        mod = self.unboom(mod)
        
        # possible parameter saving like SHA-RNN (seems to slow down training significantly)
        # mod = torch.stack(mod.chunk(4, dim = -1), dim = -1).sum(dim = -1)
        
        mod = self.boom_drop(mod)

        return mod



class LeapDecoder(nn.Module):
    def __init__(self, config):
        super(LeapDecoder, self).__init__()
        self.config = config
        self.decoders = nn.ModuleList([LeapBlock(config, window_size)
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
class LeapConfig(PretrainedConfig):
    model_type = "LeapForCausalLM"
    def __init__(self, hidden_size = 256, vocab_size = 32100, n_heads = 4,
                 use_local_att = True, window_sizes = None, n_positions = 1024,
                 n_layer = 4, hidden_dropout_prob = .1, initializer_range = .02):
        
        # check head sizes
        assert hidden_size % n_heads == 0, "hidden_size is not divisible by n_heads"
        
        # check window sizes (and set them automatically if not set)
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



class LeapForCausalLM(PreTrainedModel):
    config_class = LeapConfig
    
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx = 0)
        self.proj_logits = nn.Linear(config.hidden_size, config.vocab_size)
        self.leap_model = LeapDecoder(config)
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
        attention_mask = attention_mask.unsqueeze(-1)
        
        embds=self.word_embedding(input_ids)
        layer_outputs = self.leap_model(embds, attention_mask)
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
AutoConfig.register("LeapForCausalLM", LeapConfig)
AutoModelForCausalLM.register(LeapConfig, LeapForCausalLM)