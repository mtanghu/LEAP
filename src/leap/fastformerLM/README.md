# Additive Attention Is All You Need?

In this section, we adapt Additive Attention first introduced in [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) specifically for causal language modeling. This was the early inspiration for LEAP where most of the linearization math/positive aspects of the model come from here (as such the READMEs for both are similar).

This README will show a new *strongly* scaled dot-product which should be of independent interest to Softmax-based Attention as a whole, then move on to show annotated Additive Attention mathematics, a unique linearization process/math (which allows for an RNN formulation), show how this approach can be used for linear local attention, as well as preliminary results which show that Additive Attention (when local attention is used) is potentially comparable to full attention.

The code at `fastformer.py` in this folder is annotated with how the equations from the paper and this repo relate to code and should be easy to read (the relevant class would be `FastSelfAttention` at the top of the code).

## Usage & Development

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/))

```bash
pip install leap-transformer
```

Then to use in python (setting the config how you want):
```python
from leap import FastformerForCausalLM, FastformerLMConfig

config = FastformerLMConfig(
    hidden_size = 128, # size of embeddings
    vocab_size = 32100, # number of tokens, if you have a tokenizer use len(tokenizer) instead
    n_positions = 2048, # max number of tokens to process at once
    n_layer = 6, # how many stacked decoder layers to use
    use_local_att = True, # whether to use windowed/local Additive Attention
    window_sizes = None, # window sizes to use for windowed/local Additive Attention for each layer (set automatically if None)
    n_heads = 4, # number of heads to use in multi-head attention
    initializer_range = .02, # standard deviation for weight initialization
    hidden_dropout_prob = .1, # dropout value used for embeddings, attention, and feedforward layers
    rescale = 10 # what to rescale the attention values with, set lower if you have unstable/NaN loss
)

model = FastformerForCausalLM(config)

# this model is compatible with huggingface and its "trainer" interface
from transformers import Trainer
trainer = Trainer(
    model = model,
    args = <YOUR TRAINING ARGS>,
    train_dataset = <YOUR TOKENIZED DATASET>,
    ...<YOUR OTHER TRAINER ARGS>
)

trainer.train()
```
A more complete training example with a dataset, tokenization, and evaluations can be found at ``FastLM.ipynb`` in this folder which can be run with only 6GB of VRAM (GPU memory).

## Rescaled Dot-Product

This project encounted some training instability, where some preliminary investigation found that the attention/focus scores that were being generated were absurdly large/small. This may be of independent interest to training instability in large language models (say in [Megatron](https://arxiv.org/abs/1909.08053) or [PALM](https://arxiv.org/abs/2204.02311)) as the arguments shown in this section may offer a reasonable explaination for why there may be training instability particularly in large models.

### Normal Scaled Dot-Product Attention 

<div></div>Let's consider the normal context where an "Attention score" $A_{ij}$ of a query and a key is calculated as follows 

$$
A_{ij} = {exp(\boldsymbol{Q_i} \cdot \boldsymbol{K_j} / \sqrt{d_{model}}) \over \sum\limits_{j= 0}^{N-1} exp(\boldsymbol{Q_i} \cdot \boldsymbol{K_j} / \sqrt{d_{model}})}
$$

<div></div>To break this down, we simply measure the "similarity" of Query vector $i$ with Key vector $j$ (measured through dot product), then scale by a factor of $1 \over \sqrt{d_{model}}$ (we will get back to this). To ensure a "limited attentional span" we apply a softmax (i.e. dividing the similarity score of Query $i$ with Key $j$ by that Query's similarities with all the other Keys, an analgous situation would be calculating "the weight of a test on the final grade") strengthening this "limited attentional span" effect with exponentiation where a strong/high similarity between Query $i$ and Key $j$ will get exploded to a very large number and very negative similarity scores will mapped to an exponentially small number.

### Why scale by $1 \over \sqrt{d_{model}}$?

<div></div>According to [Attention is All you Need](https://arxiv.org/abs/1706.03762) (Viswani et al. 2017), the reason is simple. Consider that $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ are random normal vectors (which may be the case with random initialization and when using layernorm), it is easy to show then that the dot product of $\boldsymbol{Q_i}$ with $\boldsymbol{K_j}$ will have mean of 0 and std of $\sqrt{d_{model}}$ which is remedied simply by scaling by $1 \over \sqrt{d_{model}}$ bringing the std back to 1. The authors note that this can "(push) the softmax function into regions where it has extremely small gradients" when $d_{model}$ is large.

### What is the issue?

<div></div>While the reason provided for scaling by the original paper/authors is valid and makes sense, it only considers where $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ are "random and normal". In fact, as training happens and parameters are updated through optimization/gradient descent, we can be more and more assured that $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ will be neither be random nor normal!

- <div></div>On the matter of normality , even if LayerNorm is used to normalize the embedding vector $\boldsymbol{x_i}$ before be transformed into $\boldsymbol{Q_i} = W_q \boldsymbol{x_i}$ and $\boldsymbol{K_j} = W_k \boldsymbol{x_i}$, the weight matrices $W_q$ and $W_k$ will be learned to make $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ not normal (there are good reasons for why this would happen, like a particular $\boldsymbol{K_j}$ vector may be learned to become especially large if the token it represents is particularly "important" for the entire sequence)
-  <div></div>On the matter of randomness, there is an especially problematic scenario when $\boldsymbol{Q_i} == \boldsymbol{K_j}$ which is very likely to happen when there is highly deterministic token interactions where a token would need to strongly pay attention to another token (like verb conjugation or subject-verb agreement where there is only one right answer). Even if we assume normality, it is easy to show that this case will cause the dot-product of $\boldsymbol{Q_i}$ with $\boldsymbol{K_j}$ to have mean of $d_{model}$ (!!) (because $\boldsymbol{Q_i} \cdot \boldsymbol{K_j} = \sum\limits_{z=0}^{d_{model-1}} r_z^2$ where $r$ is a random normal variable which would likewise have mean of 1, also consider the case that $\boldsymbol{Q_i} == -\boldsymbol{K_j}$ when the there should be no alignment).

<div></div>Note that this general argument applies even when $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ are only *slightly* correlated, and at large scales with huge $d_{model}$ values it can be very easy for there to be spurious correlations between parts of $\boldsymbol{Q_i}$ and $\boldsymbol{K_j}$ after only mild training that will blow up the dot-product similarity. While there is no conclusive argument that this will necessarily cause loss spikes, this does create training instability for the simplified verisions of attention explored in this project, and in general having extremely large or extremely small attention values that get more extreme with scale would likely create some kind of training issues that at the very least "(push) the softmax function into regions where it has extremely small gradients" (the original rationale for scaled dot-product attention).

### The fix

<div></div>This project simply enforces normality (to the extent possible) using an unparameterized LayerNorm, dividing by $d_{model}$ and not $\sqrt{d_{model}}$, and multiplying by a set constant $s$.  In rigorous terms, let us define this "rescaled dot-product" using the symbol " $\bar{\cdot}$ " where for two vectors $\boldsymbol{x}, \boldsymbol{y}$ of dimension $d_{model}$:

$$\boldsymbol{x}\ \bar{\cdot}\ \boldsymbol{y} = \left({ \boldsymbol{x} - E[\boldsymbol{x}] \over \sqrt{Var[\boldsymbol{x}]}} \cdot  {\boldsymbol{y} - E[\boldsymbol{y}] \over \sqrt{Var[\boldsymbol{y}]} }\right) *{s \over d_{model}}$$

<div></div>This enforces that even if $\boldsymbol{x} ==\boldsymbol{y}$, after normalization, their dot product will have mean $d_{model}$, which we divide by to bring the mean back to 1. Then to allow for larger dot-product similarity values, we multiply by the set constant $s$ (15 seems to work fine and, after exponentiation in the softmax, e^15 should be larger than what anyone would reasonably need) to rescale the the size of the dot-product. Thus, this "recaled dot-product" will not produce larger values with larger $d_{model}$. It should be recognized that LayerNorm/normalizing a vector does not make the vector a "normal" vector (where each element is drawn from the normal distribution). This doesn't seem to be a problem emperically though. This project will work on concrete experments to show this as well as the value of this technique in the future, however preliminary measurements find that this still keeps attention sparse (in fact the pre-softmax dot product similarities quickly approach 15 within the first few steps of training) and does effectively limit the maximum value for this dot product (only a max value of ~16 was ever found).

</br>***Note: The rest of this README is meant to explain Additive Attention. To not confuse readers, the math will be shown without mentioning rescaling, but do note in the implementation that all dot products are replaced.***

## Brief Explanation of Additive Attention

<div></div>The general concept of Additive Attention is that is instead of having each embedded token to pairwise attend to each other embedded token (which happens in normal full Attention with $O(N^2)$ complexity where $N$ is the sequence length), Additive Attention relies on “global attention vectors” which condense information about the entire sequence into a single vector through a learned softmax weighted *sum* (giving the name “Additive Attention”\*). A global attention vector then confers information about the entire sequence to individual embeddings through pointwise multiplying the global attention vector with each embedding vector. The specifics of this last step and other structural details are best explained [in the original paper]([https://arxiv.org/pdf/2108.09084.pdf](https://arxiv.org/pdf/2108.09084.pdf)). We will however dive deeper into the Additive Attention mechanism itself as we will need to adapt it for causal language modeling rather than classification (as was the purpose of the original paper).

This project considers the Additive Attentional mechanism described in [Wu et al. 2021](https://arxiv.org/pdf/2108.09084.pdf) to only be just the following equations (as all the other equations/steps do not apply to the sequence dimension):

- <div></div>Consider a sequence of embeddings $\boldsymbol{x_{i}}$ with $i$ from 0 to N-1 (for N tokens)

- <div></div>Get an “attention weight” $\alpha_{i}$ (which is just a scalar) for each embedding by projecting the embedding to a single dimension that will be scaled and softmax-ed over the sequence dimension, i.e.

$$
	(1)\ \alpha_i =  {exp(\boldsymbol{w}^T \boldsymbol{x_{i}}) \over \sum\limits_{j=0}^{N-1} exp(\boldsymbol{w}^T \boldsymbol{x_{j}})}
$$

- <div></div>Multiply the embeddings by their “attention weight” (so important embeddings are emphasized over unimportant embeddings which are pushed toward 0, note how this offers "explainability"), and sum over the sequence dimension to get a “global attention vector” ${\boldsymbol{g}}$ that contains information about the entire sequence, i.e.

$$ 
	(2)\ \boldsymbol{g} = \sum_{i=0}^{N-1} \alpha_{i} \boldsymbol{x_i}
$$

<div></div>Which is clearly $O(N)$ or linear in time complexity w.r.t the sequence length $N$. Note that this is also $O(1)$ in path length in the same way full attention is $O(1)$ path length as any token $\boldsymbol{x_{i}}$ can directly confer information to ${\boldsymbol{g}}$ without having to go through other tokens (and just like full attention, this is aided by the softmax which will clear away unimportant tokens by down weighting them). Note that the original paper uses a scaling term, however we address that in the Rescaled Dot Product section where we will not use the normal dot-product scaling.
  
\* Not to be confused with [Additive Attention by Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473v7)

## Additive Attention for Causal Language Modeling

Causal Language Modeling or decoder-based language modeling is where a language model is tasked with *generating* the next token given all previous tokens, though training is performed in parallel where all tokens present in training, BUT token embeddings are not allowed to receive information about future tokens. This restriction presents a challenge because, at a high level, a global attention vector that confers information about the entire sequence to each individual token embedding will certainly allow a token embedding to “see into the future” unduly. To remedy this, we need to create an equivalent sequence of global attention vectors, one for each token, that only contains sequence information **up to each token**.

- To do this rigorously, let's start by substituting equation (1) into equation (2)

$$
	(3)\ \boldsymbol{g} = \sum\limits_{\ell=0}^{N-1}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) \over \sum\limits_{j=0}^{N-1} exp(\boldsymbol{w}^T \boldsymbol{x_j})}*\boldsymbol{x_\ell}
$$


- <div></div>Now let us instead create $\boldsymbol{g_{i}}$, which would be the equivalent global attention vector for sequence information up to (and including) token $i$. This gives us:

$$
	(4)\ \boldsymbol{g_i} = \sum\limits_{\ell=0}^{i}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) \over \sum\limits_{j=0}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j})}*\boldsymbol{x_\ell}
$$


- <div></div>Though we may have a time complexity issue. The original Additive Attention mechanism shown in equation (3) takes $O(N)$ time, so recalculating it for every token $i$ as equation (4) might suggest would yield a $O(N^2)$ time complexity. Furthermore, because of the nested summation in equation (4) it may seem impossible to reuse previous calculations to get a linear time complexity. However, in a style reminiscent of [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) we can rewrite equation 4 by factoring out the denominator, i.e.

$$
	(5)\ \boldsymbol{g_i} =  {\sum\limits_{\ell=0}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) *\boldsymbol{x_\ell} \over \sum\limits_{\ell=0}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell})}
$$

- <div></div>Where the summation terms in the numerator and denominator for each $i$ can be calculated simply by cumulative sum. To elaborate on this and the linear complexity, we can write this as a kind of RNN as Katharpoulos et al. (2020) did.

$$
	(6)\ \boldsymbol{s_i} = \boldsymbol{s_{i-1}} +exp(\boldsymbol{w}^T \boldsymbol{x_i}) *\boldsymbol{x_i}
$$

$$
	(7)\ z_i = z_{i-1} +exp(\boldsymbol{w}^T \boldsymbol{x_i}) 
$$

$$
	(8)\ \boldsymbol{g_i} = {\boldsymbol{s_i} \over z_i}
$$

Note that Katharpoulos et al. didn't quite have the same process, they instead had to use a kernel to approximate the similarity scores of the keys and queries (and the associativity property) and likewise didn't and couldn't use a softmax. This Additive Attention math still uses a softmax, but the similarity between the two approaches/styles is how they both come from rewriting the equations and using some simplification techniques to achieve linear in complexity attention that can be represented as an RNN/cumulative sum with a vector that has sequence information in the numerator and a scalar normalization term in the denominator.

## Local Additive Attention (or Windowed Attention)

<div></div>While it may make sense to "generate a global attention vector for each token in a sequence", this runs into trouble when potentially more local information is needed (like part-of-speech, subject in a sentence, modifiers for nouns, correct conjugation for verbs, preposition agreement with nouns, etc.). To remedy this, we can simply apply the Additive Attention mechanism to a limited backward context. The principle for this is exactly the same as local scaled-dot product Attention where you can find [a diagram and code from lucidrains](https://github.com/lucidrains/local-attention) this is also explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf). Unfortunately though, local attention will cost $O(N*k)$ where $k$ is the size of the backwards context (or in other words, the number of previous tokens to attend to for each token), and it can be unclear whether this *really* improves over $O(N^2)$  complexity (with full scaled dot product attention) when taking into consideration that certainly longer sequences (i.e. bigger $N$) will require larger $k$ to get the good performance (this is discussed in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) as a natural tradeoff). This would even apply to our linear Additive Attention method *if implemented naively*, though again we can use some mathematical trickery (like in the previous section) to achieve $O(N)$ *independent* of $k$ (!!!) something that Katharpoulos et al. (2020) had not considered.<br/><br/>

- <div></div>First let us reconsider equation 5 (which is the Additive Attention equation) for a local/windowed version which only considers the previous $k$ tokens (including itself)

$$
	(9)\ \boldsymbol{g_i} =  {\sum\limits_{\ell=max(i-k+1,\ 0)}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} ) *\boldsymbol{x_\ell} \over \sum\limits_{\ell=max(i-k+1,\ 0)}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell})}
$$

- <div></div>Note that we need to use a $max$ in case the backwards context or "window" is non-existent or paritally non-existent for $i < k$. Then to calculate this with linear time complexity, we simply rewrite the sum, but this time, un-simplifying!


$$
	(10)\ \ 
\boldsymbol{g_i} =  {\sum\limits_{\ell=0}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) *\boldsymbol{x_\ell} - \sum\limits_{\ell=0}^{max(i-k,\ -1)}exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) *\boldsymbol{x_\ell}
		\over
\sum\limits_{\ell=0}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell}) -\sum\limits_{\ell=0}^{max(i-k, -1)} exp(\boldsymbol{w}^T \boldsymbol{x_j})}
$$

- Where again, we can easily keep track of all 4 summation terms using cumulative sums. To show this in a different way, we can rewrite this as an RNN again to also emphasize the linearity.

$$
	(11)\ \boldsymbol{s_{i}} = \boldsymbol{s_{i-1}} +exp(\boldsymbol{w}^T \boldsymbol{x_{i}}) *\boldsymbol{x_{i}}
$$

$$
	(12)\ 
	\begin{cases}
		\boldsymbol{s_{i}'} = \boldsymbol{s_{i-1}'} +exp(\boldsymbol{w}^T \boldsymbol{x_{i-k}}) *\boldsymbol{x_{i-k}}, & \text{if }\ i \geq k\\
		\boldsymbol{s_{i}'} = 0, & \text{if  }\ i < k
	\end{cases}
$$

$$
	(13)\ z_{i} = z_{i-1} +exp(\boldsymbol{w}^T \boldsymbol{x_{i}}) 
$$

$$
	(14)\ 
	\begin{cases}
		z_{i}' = z_{i-1}' +exp(\boldsymbol{w}^T \boldsymbol{x_{i-k}}) , & \text{if  }\ i \geq k\\
		z_{i}' = 0, & \text{if  }\ i < k\\
	\end{cases}
$$

$$
	(15)\ \boldsymbol{g_i} = {\boldsymbol{s_i} - \boldsymbol{s_i'} \over z_i -z_i'}
$$

Note that this should be doable with [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) too, though it seems like this is a novel idea that comes from an understanding that Additive Attention is global.  We specifically wanted to have some kind of local attention in this project whereas Katharpoulos et al. (2020) came from a mathematical perspective that kernel-izing the similarity function should roughly approximate softmax similarity scores.

<div></div>For implementation in a language model, we need to pick the right $k$ value, though this project proposes to consider different $k$ values for each layer. After some mild experimentation, the heuristic chosen for this project is $k_\ell = 4(2^{\ell})$ where $\ell$ is the hidden layer number (starting from 0), then the last layer will be "global attention" (unlimited window size). The intuition here would be that the lower levels are parsing lower-level semantic information that is accumulated with a final global attention layer. Note that because bigger models already use more layers, this should probably only help the scaling potential of this model. 

## Minor implementation details

<div></div>A numerical stability term should added to the denominator of equations denominators of these equations in case the denominator gets very close to 0 (including the normalizing equations of the rescaled dot-product). Also to reiterate, recaled dot-products are used in place of the down projection (which is the same operation as a dot-product), so instead of $\boldsymbol{w}^T \boldsymbol{x_i}$ (i.e. the down projection) we instead use $\boldsymbol{w}\ \bar{\cdot}\ \boldsymbol{x_i}$ where the definition of  " $\bar{\cdot}$ " can be found in the Rescaled Dot-Product section. 

## Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2 [paper for reference by Radford et al. (2019)](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf):

- Pre-norming with a layernorm before projecting to logits like GPT2
- Dropout of .1 on embeddings, feedforward layers, and this project applies dropout after each usage of Additive Attention (to mimic attention dropout) just like GPT2
- Learned positional embeddings ([GPT1 paper by Radford et al. (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) which carries over to GPT2, though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) we considered, but decided against because it would unfairly give an advantage to the model when compared against normal transformers/gpt2 which uses learned absolute positional embeddings
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by GPT2
- Label smoothing ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762)) is forgone because huggingface seems to oddly apply label smoothing during validation (so the loss that comes out when exponentiated would not be perplexity)
- Attention masking of pad tokens ([Attention is All you Need by Viswani et al. (2017)](https://arxiv.org/abs/1706.03762)) which is carried over to GPT2
- <div></div>Multihead Attention where the vectors in each head are simply a down projection to size $d_{model} \over n_{heads}$ with the same number of parameters as a single-head also as per Attention is All you Need by Viswani et al. (2017) which is carried over to GPT2

## Results

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/src/leap/fastformerLM/preliminary_results.png?raw=True)

Plotted is the validation perplexity of the Additive Attention models (blue and orange) compared to full attention model (GPT2 in green) with the model sizes stated in the legend trained on Wikitext-2 using a T5 tokenizer with sequence lengths of 2048. The "Windowed Additive Attention" uses local Additive Attention explained in the "Local Additive Attention (or Windowed Attention)" section.  After loading the model with the lowest validation perplexity, the test set perplexity for Additive Attention was 71.0, for Windowed Additive Attention 50.0, and for GPT2 59.7.

As we can see on this small scale experiment, the Windowed Additive Attention strongly outcompetes the standard Additive Attention and converges faster with less perplexity compared to even GPT2 (with slightly less parameters too). Even though these results are preliminary, the long sequence length of 2048 should already be enough to test the abilities of this model as being better than an RNN like LSTMs as found by this [Scaling Laws paper](https://arxiv.org/abs/2001.08361) (Figure 7 finds that LSTM scaling bends at around 1M parameters, and at context lengths of >1000, the LSTM should be unable to compete). Also because of the linear local attention, it may be more reasonable to believe that this model can scale up (as the combinations of local and global attentions should be able to model complex sequence information from short-range to long-range). Furthermore, this model beats both [Mogrifier LSTM](https://arxiv.org/abs/1909.01792v2) and [AWD LSTM](https://arxiv.org/abs/1708.02182v1) (when not using dynamic eval) on Wikitext-2 perplexity even though those models use >30M parameters (see the [leaderboard on paperswithcode](https://paperswithcode.com/sota/language-modelling-on-wikitext-2)).

**Speed:** Both Additive Attention training runs took ~30 minutes while GPT2 took ~48 minutes (1.6x the time) which should become more pronounced at context lengths greater than 2048 and larger model sizes. Also the convergence was faster.

### Training details

All models were trained on a single NVIDIA GeForce RTX 2060 with batch sizes of 2 and fp16 mixed precision. Code can be found in ``FastLM.ipynb`` which uses the [HuggingFace 🤗 Transfomers](https://huggingface.co/) specifically the [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer).

Model details: all models had a model dimension of 128 (feedforward dimension is 4x the model dimension), 4 attention heads, 6 layers and dropout probability of .1 on embedding, attention, and feedforward layers. The window sizes for Local Additive Attention were [4, 8, 16, 32, 64, 2028] as per the heurstic described.

Optimizer: [AdamW](https://arxiv.org/abs/1711.05101) with learning rate of 5e-4 to start and linear annealing, betas=(.9, .999) and weight decay = .01. Mixed precision with gradient clipping = 1 (max norm) is also used.

**Tuning** Very little tuning has been done to optimize performance, only learning rates of [1e-4, 5e-4, 1e-3] have been tried (5e-4 is best for both GPT and Additive Attention in preliminary experiments) and only a few different kind of window size settings have been explored (<20). No tuning on the test set was performed.


## References
Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). Huggingface's transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_.

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., ... & Ahmed, A. (2020). Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_, _33_, 17283-17297.

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-lm: Training multi-billion parameter language models using model parallelism. _arXiv preprint arXiv:1909.08053_.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2022). Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_.

Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020, November). Transformers are rnns: Fast autoregressive transformers with linear attention. In _International Conference on Machine Learning_ (pp. 5156-5165). PMLR.

Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. _arXiv preprint arXiv:1409.0473_.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help?. _Advances in neural information processing systems, 32_.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. _Advances in neural information processing systems, 30_.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

Press, O., & Wolf, L. (2016). Using the output embedding to improve language models. _arXiv preprint arXiv:1608.05859_.

Kaplan, Jared, et al. "Scaling laws for neural language models." _arXiv preprint arXiv:2001.08361_ (2020).

Melis, G., Kočiský, T., & Blunsom, P. (2019). Mogrifier lstm. arXiv preprint arXiv:1909.01792.

Merity, S., Keskar, N. S., & Socher, R. (2017). Regularizing and optimizing LSTM language models. arXiv preprint arXiv:1708.02182.

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_.
