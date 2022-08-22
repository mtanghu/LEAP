My Drive

# Additive Attention Is All You Need?
This curiosity project adapts [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) for causal language modeling. Code loosely adapted from the [original authors' fastformer code](https://github.com/wuch15/Fastformer) though virtually all parts of the code have been rewritten. Note that this project is somewhat unique as most Linear Attention mechanisms cannot be used for parallel decoder language modeling (see Eleuther comments on ["Have you considered more efficient architectures or methods?](https://www.eleuther.ai/faq/)). Also as per the original paper, the models considered in this repo do run faster than a standard Transformer when run with the same # of layers and layer sizes (which is not the case with other forms of sparse linear attention).

This README will summarize Additive Attention and annotate a number of its details, then show an unique connection to [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) in the linearization process/math as well as preliminary results which show that Additive Attention is potentially comparable to full attention (though only on small scales so far), and there is room for development!

## Usage & Development

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/))

```bash
pip install fastformerLM
```

Then to use in python (setting the config how you want):
```python
from fastformer import FastformerForCausalLM, FastformerLMConfig

config = FastformerLMConfig(
    hidden_size = 256, # size of embeddings
    vocab_size = 32100, # number of tokens, if you have a tokenizer use len(tokenizer) instead
    n_positions = 2048, # max number of tokens to process at once
    n_heads = 4, # number of heads to use in multi-head attention
    convolve = True, # whether to employ a convolutional layer (note: will increase parameter count)
    groups = 1, # number of groups in convolution layer (ignored if convolve = False) 
    kernel_size = 4, # kernel size for convolution layer (ignored if convolve = False) 
    num_hidden_layers = 4, # how many stacked decoder layers to use
    label_smoothing = 0, # amount of label smoothing to use
    initializer_range = .02, # standard deviation for weight initialization
    hidden_dropout_prob = .1 # dropout value used for embeddings, attention, and feedforward layers
)

model = FastformerForCausalLM(config)

# this model is compatible with huggingface and its "trainer" interface
from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=<YOUR TOKENIZED DATASET>
)

trainer.train()
```
A more complete toy training example with a dataset, tokenization, and evaluations can be found at ``FastLM.ipynb`` in this repository.

### Development
A number possibilities for development and usage come to mind:

1. Additive Attention may work better on other & more specific NLP tasks (Question Answering, Summarization, etc.) as per the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which especially applies to unique datasets. Other Linear Attention papers seem to use unique datasets.
2. More ablation and feature adding studies could be performed to improve performance.
3. To help with scalability, maybe [local attention](https://github.com/lucidrains/local-attention) could be used instead of convolution. (note a kind of local Additive Attention was tried informally, but it didn't seem to work well)
4. Additive Attention only has global attention (and this project also implements local attention with convolution) and thus may be an interesting model to play around with for exploring attentional mechanisms.
5. Visual Transformers may only need the kind of "global attention" that made Additive Attention SOTA for classification in the original paper. Thus for causal image generation, this project may work.
6. If this can be scaled up with good performance, because of the RNN formulation, it would be possible to have a large language model run on an edge device.


## Brief Explanation of Additive Attention

The general concept of Additive Attention is that is instead of allowing each embedded token to attend to every other embedded token (which is N^2 complexity where N is the sequence length), Additive Attention* relies on “global attention vectors” which condense information about the entire sequence into a single vector through addition/summation (giving the name “Additive Attention”*). A global attention vector then confers information about the entire sequence to individual embeddings through pointwise multiplying the global attention vector with each embedding vector. The specifics of this last step and other structural details are best explained [in the original paper]([https://arxiv.org/pdf/2108.09084.pdf](https://arxiv.org/pdf/2108.09084.pdf)). We will however dive deeper into the Additive Attention mechanism itself as we will need to adapt it for causal language modeling rather than classification (as was the purpose of the original paper)

Paraphrasing to some degree, the Additive Attentional mechanism described in [Wu et al. 2021](https://arxiv.org/pdf/2108.09084.pdf)) is primarily just the following equations:

Consider a sequence of (possibly transformed) embeddings $\boldsymbol{x_i}$ with $i$ from 1 to N…

1.  Get an “attention weight” $\alpha_i$ (which is just a scalar) for each embedding by projecting the embedding to a single dimension that will be scaled and softmax-ed over the sequence dimension, i.e.

$$
\begin{align}
	(1)\ \alpha_i =  {exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{N} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}
\end{align}
$$

2.  Multiply the embeddings by their “attention weight” (so important embeddings are emphasized over unimportant embeddings which are pushed toward 0), and sum over the sequence dimension to get a “global attention vector” $\boldsymbol{g}$ that contains information about the entire sequence, i.e.

$$ 
\begin{align}
	(2)\ \boldsymbol{g} = \sum_{i=1}^{N} \alpha_i \boldsymbol{x_i}
\end{align}
$$

Which is clearly $O(N)$ or linear in time complexity w.r.t the sequence length $N$.
  
\* Not to be confused with [Additive Attention by Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473v7)

## Additive Attention for Causal Language Modeling

Causal Language Modeling or decoder-based language modeling is where a language model is tasked with *generating* the next token given all previous tokens, though training is performed in parallel with all tokens present in training, BUT token embeddings are not allowed to receive information about future tokens. This restriction presents a challenge because, at a high level, a global attention vector that confers information about the entire sequence to each individual token embedding will certainly allow a token embedding to “see into the future” unduly. To remedy this, we need to create an equivalent sequence of global attention vectors, one for each token, that only contains sequence information **up to each token**.

To do this rigorously, let's start by substituting equation (1) into equation (2)

$$
\begin{align}
	(3)\ \boldsymbol{g} = \sum\limits_{\ell=1}^{N}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{N} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}*\boldsymbol{x_\ell}
\end{align}
$$


Now let us instead create $\boldsymbol{g_i}$, which would be the equivalent global attention vector for sequence information up to (and including) token $i$. This gives us:

$$
\begin{align}
	(4)\ \boldsymbol{g_i} = \sum\limits_{\ell=1}^{i}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}*\boldsymbol{x_\ell}
\end{align}
$$


Though we may have a time complexity issue. The original Additive Attention mechanism shown in equation (3) takes $O(N)$ time, so recalculating it for every token $i$ as equation (4) might suggest would yield a $O(N^2)$ time complexity. Furthermore, because of the nested summation in equation (4) it may seem impossible to reuse previous calculations to get a linear time complexity. However, in a style reminiscent** of [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) we can rewrite equation 4 by factoring out the denominator, i.e.

$$
\begin{align}
	(5)\ \boldsymbol{g_i} =  {\sum\limits_{\ell=1}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) *\boldsymbol{x_\ell} \over \sum\limits_{j=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}
\end{align}
$$

Where the summation terms in the numerator and denominator for each $i$ can be calculated simply by cumulative sum. To elaborate on this and the linear complexity, we can write this as a kind of RNN as Katharpoulos et al. (2020) did.

$$
\begin{align}
	(6)\ \boldsymbol{s_i} = \boldsymbol{s_{i-1}} +exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) *\boldsymbol{x_i}
\end{align}
$$

$$
\begin{align}
	(7)\ z_i = z_{i-1} +exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) 
\end{align}
$$

$$
\begin{align}
	(8)\ \boldsymbol{g_i} = {\boldsymbol{s_i} \over z_i}
\end{align}
$$

**Note that Katharpoulos et al. didn't quite have the same process, they instead had to use a kernel to approximate the similarity scores of the keys and queries (and the associativity property) and likewise didn't and couldn't use a softmax. This Additive Attention math still uses a softmax, but the similarity between the two approaches/styles is how they both come from rewriting the equations and using some simplification techniques to achieve linear in complexity attention that can be represented as an RNN/cumulative sum.

## Model Structure
Because this is a causal language model the code is structured like one and implements the following:

- Ordering of residual connections, layer norms and dropouts follows [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) by Shoeybi, Patwary & Puri et al. 2020.
- Learned positional embeddings ([Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) we considered, but decided against because it would unfairly give an advantage to the model when compared against normal transformers/gpt2.
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3))
- Label smoothing of .1 ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762)) are also used. (Though huggingface seems to oddly apply label smoothing during validation as well so later experiments will forgo label smoothing)
- Attention masking of pad tokens ([Viswani et al. 2017](https://arxiv.org/abs/1706.03762))
- Scaling output projection (right before softmax) by $\frac{1}{\sqrt{d_{model}}}$ and no biases on linear layers (except for output projection) like [PALM](https://arxiv.org/pdf/2204.02311.pdf)
- Due to some training instability, a residual connection and layer norm was placed in the middle of the full additive attention process 

### Causal Convolution
An attempt was made to try to model local contexts better (since the Additive Attention mechanism only models global sequence information) using a "causal convolutional layer" somewhat inspired by [CNN Is All You Need](https://arxiv.org/abs/1712.09662) by Qimeng Chen & Ren Wu 2017. This section will be brief though as convolution may be replaced with [local attention](https://github.com/lucidrains/local-attention) at some point.

A 1D convolution was added as a layer into the transformer architecture with a residual connection and layernorm/dropout just as the attentional/feed-forward layers have in the [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) transformer architecture. This layer was placed before the attentional layer. Furthermore, to keep the "causal" property while maintaining parallelism, we pad the sequence dimension with zero vectors at the start of the sequence ($kernel\ size - 1$ of them). The effect is that as the "kernel"/window slides across the sequence each output $i$ will only have information from embedding up $i$-th token embedding.

## Results

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/results.png?raw=True)

Plotted is the validation bits per character of the Additive Attention models (blue and orange) compared to full attention model (green) with the model sizes stated in the legend. The "Convolutional Additive Attention" employs the changes described in the mini-section [Causal Convolution](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need#causal-convolution).

### Training details

All models were trained on a single NVIDIA GeForce RTX 2060 on Wikitext-2 (raw) using a T5 tokenizer with sequence lengths of 1024 and batch size 4.

Model details: all models had a model dimension of 256 (feedforward dimension is 4x the model dimension), 4 attention heads, 4 layers (though the convolutional additive attention only has 3 layers to fit the convolutional layer) and dropout probability of .1 on embedding, attention and feedforward layers. The convolutional layer had a kernel size of 4.

Optimizer: [AdamW](https://arxiv.org/abs/1711.05101) with learning rate of 1e-3 to start and linear annealing, betas=(.9,.999) and weight decay = .01

### Soft Lessons
- I made a few mistakes in adapting Additive Attention to Causal Language Modeling mostly just spurred from ignorance. I can see the value/necessity of going over the math and checking the math (as the Additive Attention for Causal Language Modeling section does) for ensuring accuracy in this kind of Neural Network context.
- Label smoothing can mess up measuring validation loss and the smoothing is applied when measuring the validation loss
- Something that I've experienced already but this project reaffirmed strongly: a lot happens in the experimental process, most of it unexpected! Experiments often didn't go the way one might think even though previous results in other papers would suggest certain results. I think it's important to accept this and quickly adjust to a new plan that hopefully is even better than the previous since now more information is known!

## References
Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020, November). Transformers are rnns: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_ (pp. 5156-5165). PMLR.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_.

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help?. _Advances in neural information processing systems, 32_.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. _Advances in neural information processing systems, 30_.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2022). Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

Press, O., & Wolf, L. (2016). Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_.

Chen, Q., & Wu, R. (2017). CNN is all you need. _arXiv preprint arXiv:1712.09662_.

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_.
