# Additive Attention Is Not All You Need?
This curiosity project adapts Additive Attention described by Wu et al. (2021) for causal language modeling. This repo will show some preliminary experiments which explore linear attention and how maybe additive attention doesn't quite work that well for causal language modeling. Code loosely adapted from the [original authors' fastformer code](https://github.com/wuch15/Fastformer) though virtually all parts of the code have been rewritten. ``fastformer.py`` contains a HuggingFace compatible model and the different layers that go into it. ``FastLM.ipynb`` is the training/testing notebook where integration with HuggingFace is shown.

The purpose of this project was to see whether the state-of-the-art results shown in the original paper would translate to Causal Language Modeling. As you'll see, Additive Attention falls short, though the later sections will summarize Additive Attention and annotate a number of its details, then show an unique connection to [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) in the linearization process.

## Brief Explanation of Additive Attention

The general concept of Additive Attention is that is instead of allowing each embedded token to attend to every other embedded token (which is N^2 complexity where N is the sequence length), Additive Attention* relies on “global attention vectors” which condense information about the entire sequence into a single vector through addition/summation (giving the name “Additive Attention”*). A global attention vector then confers information about the entire sequence to individual embeddings through pointwise multiplying the global attention vector with each embedding vector. The specifics of this last step and other structural details are best explained [in the original paper]([https://arxiv.org/pdf/2108.09084.pdf](https://arxiv.org/pdf/2108.09084.pdf)). We will however dive deeper into the Additive Attention mechanism itself as we will need to adapt it for causal language modeling rather than classification (as was the purpose of the original paper)

Paraphrasing to some degree, the Additive Attentional mechanism described in [Wu et al. 2021](https://arxiv.org/pdf/2108.09084.pdf)) is primarily just the following equations:

Consider a sequence of (possibly transformed) embeddings $\boldsymbol{x_i}$ with $i$ from 1 to N…

1.  Get an “attention weight” $\alpha_i$ (which is just a scalar) for each embedding by projecting the embedding to a single dimension that will be scaled and softmax-ed over the sequence dimension, i.e.

$$
\begin{align}
	(1)\ \alpha_i =  {exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}
\end{align}
$$

2.  Multiply the embeddings by their “attention weight” (so important embeddings are emphasized over unimportant embeddings which are pushed toward 0), and sum over the sequence dimension to get a “global attention vector” $\boldsymbol{g}$ that contains information about the entire sequence, i.e.

$$ 
\begin{align}
	(2)\ \boldsymbol{g} = \sum_{\ell=1}^{N} \alpha_\ell \boldsymbol{x_\ell}
\end{align}
$$

Which is clearly $O(N)$ or linear in time complexity w.r.t the sequence length $N$.
  
\* Not to be confused with [Additive Attention by Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473v7)

## Additive Attention for Causal Language Modeling

Causal Language Modeling or decoder-based language modeling is where a language model is tasked with *generating* the next token given all previous tokens, though training is performed in parallel with all tokens present in training, BUT token embeddings are not allowed to receive information about future tokens. This restriction presents a challenge because, at a high level, a global attention vector that confers information about the entire sequence to each individual token embedding will certainly allow a token embedding to “see into the future” unduly. To remedy this, we need to create an equivalent sequence of global attention vectors, one for each token, that only contains sequence information **up to each token**.

To do this rigorously, let's start by substituting equation (1) into equation (2)

$$
\begin{align}
	(3)\ \boldsymbol{g} = \sum\limits_{\ell=1}^{N}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}*\boldsymbol{x_\ell}
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
Because this is a causal language model, the code follows the structure/ordering of [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) by Shoeybi, Patwary & Puri et al. 2020. Standard practices like learned positional embeddings ([Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) and weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) are also used. 

### Causal Convolution
An attempt was made to try to model local contexts better (since the Additive Attention mechanism only models global sequence information) using a "causal convolutional layer" somewhat inspired by [CNN Is All You Need](https://arxiv.org/abs/1712.09662) by Qimeng Chen & Ren Wu 2017. Though it didn't seem to improve performance (surprisingly?) so the description will be brief.

A 1D convolution was added as a layer into the transformer architecture with a residual connection and layernorm/dropout just as the attentional/feed-forward layers have in the [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) transformer architecture. This layer was placed before the attentional layer. Furthermore, to keep the "causal" property while maintaining parallelism, we pad the sequence dimension with zero vectors at the start of the sequence ($kernel\ size - 1$ of them). The effect is that as the "kernel"/window slides across the sequence each output $i$ will only have information from embedding up $i$-th token embedding.

(Note: a number of adaptations were tried like removing/moving a gelu layer, removing dropout as is usual for CNNs, replacing key/query/value transformations with CNNs, etc. none of which particularly improved results)

## Results
While the training scheme used is preliminary (wikitext-2 with a character tokenizer), the tests are still even for all models and it should be relatively clear that there is a clear gap between Additive Attention and full attention (we use [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by RWCLAS 2018). It may be possible that Additive Attention could compare favorably with other linear attention mechanisms on certain tasks, though this is unexplored currently.

![alt text](https://github.com/mtanghu/Additive-Attention-Is-Not-All-You-Need-Maybe/blob/main/results.png?raw=True)

Plotted is the validation bits per character of the Additive Attention models (blue and orange) compared to full attention model (green). The "Convolutional Additive Attention" employs the changes described in the mini-section "Causal Convolution". To speculate on why Additive Attention is so lackluster, the results in the original paper were for classification with a [BERT](https://arxiv.org/abs/1810.04805) style model where only global summarization of a sequence is needed (like how a normal CLS token/accumulation does) and the Additive Attention mechanism just implements this prior. This may not be suffient for Causal Language Modeling as each token may need very specific information about other tokens and not just a global summary.

### Soft lessons
- It was interesting to see the "gap" in the validation losses both graphically and being the one actually running the models. In papers it might just seem like a benchmark has been "improved" by some arbitrary amount, but this project has helped me appreciate how non-trivial positive results are.
- I made a few mistakes in adapting Additive Attention to Causal Language Modeling mostly just spurred from ignorance. I can see the value/necessity of going over the math and checking the math (as the Additive Attention for Causal Language Modeling section does) for ensuring accuracy in this kind of Neural Network context.
- Something that I've experienced already but this project reaffirmed strongly: a lot happens in the experimental process, most of it unexpected! Experiments often didn't go the way one might think even though previous results in other papers would suggest certain results. I think it's important to accept this and quickly adjust to a new plan that hopefully is even better than the previous since now more information is known!

## References
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020, November). Transformers are rnns: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_ (pp. 5156-5165). PMLR.

Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

Press, O., & Wolf, L. (2016). Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_.

Chen, Q., & Wu, R. (2017). CNN is all you need. _arXiv preprint arXiv:1712.09662_.
