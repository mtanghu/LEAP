# Additive Attention Is All You Need?
This curiosity project adapts [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) for causal language modeling. Code loosely adapted from the [original authors' fastformer code](https://github.com/wuch15/Fastformer) though virtually all parts of the code have been rewritten. 

This README will summarize Additive Attention and annotate a number of its details, then show an unique connection to [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) in the linearization process as well as preliminary results which show that Additive Attention is a little far off from full attention, but there is room for development!

## Usage & Development

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/))

```bash
git clone https://github.com/mtanghu/Additive-Attention-Is-All-You-Need.git
cd Additive-Attention-Is-All-You-Need
pip install .
```

Then to use in python (setting the config how you want):
```python
from fastformer import FastformerForCausalLM, FastformerLMConfig

config = FastformerLMConfig(
    hidden_size = 256, # size of embeddings
    vocab_size = 384, # number of tokens, if you have a tokenizer use len(tokenizer) instead
    max_position_embeddings = 256, # max number of tokens to process at once
    convolve = False, # whether to employ a convolutional layer (note: will increase parameter count)
    groups = 2, # number of groups in convolution layer (ignored if convolve = False) 
    kernel_size = 4, # kernel size for convolution layer (ignored if convolve = False) 
    num_hidden_layers = 6, # how many stacked decoder layers to use
    label_smoothing = .1, # amount of label smoothing to use
    initializer_range = .02, # standard deviation for weight initialization
    hidden_dropout_prob = .1 # dropout value used for embeddings, attention, and feedforward layers
)

model = FastformerForCausalLM(config)

# this model is compatible with huggingface and its "trainer" interface
from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=<YOUR DATASET>
)

trainer.train()
```
A more complete toy training example with a dataset and evaluations can be found at ``FastLM.ipynb`` in this repository.

### Development
While there does seem to be a gap between full attention and Additive Attention,  there are possibilities for development and usage. Four primary possibilities come to mind:

1. Additive Attention may work better on other NLP tasks (Question Answering, Summarization, etc.) as per the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which especially applies to unique datasets. Other Linear Attention papers seem to use unique datasets.
2. Additive Attention only has global attention (and this project also implements local attention with convolution) and thus may be an interesting model to play around with for exploring attentional mechanisms.
3. Visual Transformers may only need the kind of "global attention" that made Additive Attention SOTA for classification in the original paper. Thus for causal image generation, this project may work.
4. Testing on the [Long Range Arena](https://github.com/google-research/long-range-arena), it may be possible for Additive Attention to achieve SOTA on some of the benchmarks


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


Though we may have a time complexity issue. The original Additive Attention mechanism shown in equation (3) takes $O(N)$ time, so recalculating it for every token $i$ as equation (4) might suggest would yield a $O(N^2)$ time complexity. Furthermore, because of the nested summation in equation (4) it may seem impossible to reuse previous calculations to get a linear time complexity. However, in a style reminiscent of [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) we can rewrite equation 4 by factoring out the denominator, i.e.

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

## Model Structure
Because this is a causal language model the code is structured like one and implements the following:

-  Ordering of residual connections, layer norms and dropouts follows [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) by Shoeybi, Patwary & Puri et al. 2020.
- Learned positional embeddings ([Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3))
- Label smoothing of .1 ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762)) are also used. (Though huggingface seems to oddly apply label smoothing during validation as well so later experiments will forgo label smoothing)
- Attention masking of pad tokens ([Viswani et al. 2017](https://arxiv.org/abs/1706.03762))
- Due to some training instability, a residual connection and layer norm was placed in the middle of the full additive attention process 

### Causal Convolution
An attempt was made to try to model local contexts better (since the Additive Attention mechanism only models global sequence information) using a "causal convolutional layer" somewhat inspired by [CNN Is All You Need](https://arxiv.org/abs/1712.09662) by Qimeng Chen & Ren Wu 2017. Though it doesn't seem to improve performance (surprisingly?) and seems to saturate more quickly (i.e. validation loss stops decreasing more quickly) so the description will be brief.

A 1D convolution was added as a layer into the transformer architecture with a residual connection and layernorm/dropout just as the attentional/feed-forward layers have in the [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) transformer architecture. This layer was placed before the attentional layer. Furthermore, to keep the "causal" property while maintaining parallelism, we pad the sequence dimension with zero vectors at the start of the sequence ($kernel\ size - 1$ of them). The effect is that as the "kernel"/window slides across the sequence each output $i$ will only have information from embedding up $i$-th token embedding.

(Note: a number of adaptations were tried like removing/moving a gelu layer, removing dropout as is usual for CNNs, replacing key/query/value transformations with CNNs, etc. none of which particularly improved results)

## Results
While the training scheme used is preliminary (wikitext-2 with a character tokenizer), the tests are still even for all models and it should be relatively clear that there is a clear gap between Additive Attention and full attention (we use [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by RWCLAS 2018). It may be possible that Additive Attention could compare favorably with other linear attention mechanisms on certain tasks, though this is unexplored currently (see [development section](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need#development)).

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/results.png?raw=True)

Plotted is the validation bits per character of the Additive Attention models (blue and orange) compared to full attention model (green). The "Convolutional Additive Attention" employs the changes described in the mini-section "Causal Convolution".

### Soft Lessons
- I made a few mistakes in adapting Additive Attention to Causal Language Modeling mostly just spurred from ignorance. I can see the value/necessity of going over the math and checking the math (as the Additive Attention for Causal Language Modeling section does) for ensuring accuracy in this kind of Neural Network context.
- Something that I've experienced already but this project reaffirmed strongly: a lot happens in the experimental process, most of it unexpected! Experiments often didn't go the way one might think even though previous results in other papers would suggest certain results. I think it's important to accept this and quickly adjust to a new plan that hopefully is even better than the previous since now more information is known!

## References
Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020, November). Transformers are rnns: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_ (pp. 5156-5165). PMLR.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_.

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help?. _Advances in neural information processing systems, 32_.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. _Advances in neural information processing systems, 30_.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

Press, O., & Wolf, L. (2016). Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_.

Chen, Q., & Wu, R. (2017). CNN is all you need. _arXiv preprint arXiv:1712.09662_.

Welcome file
Displaying Welcome file.
