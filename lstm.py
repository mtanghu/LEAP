import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput



class LstmForCausalLM(PreTrainedModel):
    def __init__(self, hidden_size, n_layer, vocab_size, hidden_dropout_prob = .1):
        super().__init__(PretrainedConfig())
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj_logits = nn.Linear(hidden_size, vocab_size, bias = False)
        
        self.rnn = nn.LSTM(
            input_size = hidden_size,
            hidden_size = hidden_size,
            num_layers = n_layer,
            dropout = hidden_dropout_prob,
            batch_first = True
        )
        
        self.last_norm = nn.LayerNorm(hidden_size)
        self.criterion = nn.CrossEntropyLoss()
        
        # weight tying
        self.proj_logits.weight = self.word_embedding.weight
        

    def forward(self, input_ids, attention_mask = None, labels = None, return_dict = False, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape).to(input_ids.device)
        
        embds = self.word_embedding(input_ids)
        layer_outputs = self.rnn(embds)[0]
        layer_outputs = self.last_norm(layer_outputs)
        logits = self.proj_logits(layer_outputs)
        
        loss = None
        if labels is not None:
            # set pad token labels to -100 to be ignored
            labels.masked_fill_(attention_mask == 0, -100)
            
            # shift logits the same gpt2 does at
            # https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_gpt2.html
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if return_dict is True:
            print("returning")
            return {"loss": loss, "logits": logits}

        return CausalLMOutput(loss = loss, logits = logits)