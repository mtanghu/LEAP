# Linear Explainable Attention in Parallel (LEAP)

This project implements a novel linear attention mechanism based on the "softmax-weighted cumulative sum" which has suprisingly favorable properties in computational complexity, explainability, and theoretical expressiveness. This project strongly believes that this linear attention mechanism can replace full attention with virtually no tradeoffs, if not actually having even better performance (because it's a more simple attention mechanism). This was originally inspired by adapting [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) (where they don't use any kind of cumulative sum)  for causal language modeling which we also implement with documentation and a comprehensive README that can be found in `src/leap/fastformerLM`. 

Reasons why LEAP may be able to replace full attention:

1. The models considered in this project run faster than a standard Transformer when run with the same # of layers and layer sizes even on small sequence lengths (the math allows for *highly parallelizeable* operations which is not always the case with linear attention) which offers extra ease of use

2. **Dot-product rescaling**, we find that the current dot-product attention scaling method can lead to training instability especially in this more simple form of attention. We introduce a new dot product scaling method that should stop dot product similarities from scaling with model size that *may help the training stability of full attention as well* but will allow LEAP to scale to large model sizes stably

3. **Linear in time local attention**, this concept has not been seen before in the literature as local attention typically has to scale in time complexity with the size of the local window. This project uses some simple mathematics and reuse of computations to get around this (and still be parallelizeable). This gets around the issue that longer sequences will typically need bigger local attention windows, but also builds upon the suprising strength of local + global attention (previously explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/abs/2007.14062) with the addition of random attention) with added mid-range sequence modeling. This project contends that this will give enough representational complexity to match full attention

4. **Built-in Explainability**, while explainability is not supported yet in this project, as we'll see later, each token will be assigned an "focus weight" (which is softmaxed over the sequence) which can be used to explain what tokens the model is paying attention to, and which tokens are ignored similar to the explainability offered by the original [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) paper, though more simplified

5. **O(1) Path Length/Flexibility**, A great strength of full attention Transformers is the flexibility provided by the $O(1)$ path length. An example where many linear attention mechanisms would likely fail (ie. if they only use local/convolutional attention or time-decaying factors or a recurrent vector that will get overloaded with information over time) would be when there is "*task metadata*" at the beginning of the sequence. Example: "Read the following story paying special attention to how Alice treats Bob as you will write an essay on this after: \<very long story here\>". This task information may not make it all the way through the story and writing the essay with the previously mentioned approaches, but with this project's approach, tokens from the beginning of the sequence can directly transfer information to tokens at the end of the sequence with a $O(1)$ path length (like full-attention) through global LEAP

6. **O(1) Inference**, the math of LEAP can be represented as an RNN (while still maintaining the $O(1)$ path length). Thus, you only need the previous token's embeddings (i.e. $O(1)$ space) to calculate the next token (as per being an RNN) which only takes $O(1)$ computations with no matrix-matrix operations (all with respect to sequence length holding model size/dimension constant). This was originally shown in [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) to increase inference time performance by thousands of times and could potentially *allow large language models to run on edge devices like mobile phones or consumer laptops!*

This README will describe a rescaled dot-product which may be of independent interest to full attention, summarize LEAP mathematics, and then show preliminary results which show that LEAP is potentially comparable to full attention, and there is plenty of room for development!

## Usage & Development

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/))

```bash
pip install leap-transformer
```

Then to use in python (setting the config how you want):
```python
from leap import LeapForCausalLM, LeapConfig

config = LeapConfig(
    hidden_size = 128, # size of embeddings
    vocab_size = 32100, # number of tokens, if you have a tokenizer use len(tokenizer) instead
    n_positions = 2048, # max number of tokens to process at once
    n_layer = 6, # how many stacked decoder layers to use
    use_local_att = True, # whether to use windowed/local LEAP
    window_sizes = None, # window sizes to use for windowed/local LEAP for each layer (set automatically if None)
    n_heads = 4, # number of heads to use in multi-head attention
    initializer_range = .02, # standard deviation for weight initialization
    hidden_dropout_prob = .1, # dropout value used for embeddings, attention, and feedforward layers
    rescale = 15 # what to rescale the focus values with, set lower if you have unstable/NaN loss
)

model = LeapForCausalLM(config)

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
A more complete training example with a dataset, tokenization, and evaluations can be found at ``FastLM.ipynb`` in this repository which can be run with only 6GB of VRAM (GPU memory).

### Development
 If you want to contribute, (optionally) make/address a github issue, or just send in a pull request! Use these installation instructions so that you'll have the latest repo and your edits will be reflected when you run the code

```bash
git clone https://github.com/mtanghu/LEAP.git
cd LEAP
pip install -e .
```

A number possibilities for development and usage come to mind:

1. LEAP may work much better on other & more specific NLP tasks (conversation, reasoning, genomics, speech/audio, vision, text-to-image, or even specific language domains) as per the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which especially applies to unique datasets. Other Linear Attention papers seem to use unique datasets too. This is especially the case with smaller datasets as LEAP does have stronger inductive bias
2. **Theory** to show that LEAP is Turing Complete? -- Under similar assumptions about what it means for a LM to be Turing Complete (i.e. it can model any kind of sequence computation) the global LEAP with $O(1)$ path length and multiple heads should be enough to easily prove that LEAP can replicate any computation within Turing Completeness simply by performing the same steps as Turing Machine (using similar ideas and assumptions as [Pérez, J., Marinković, J., & Barceló, P. (2019)](https://jmlr.org/papers/volume22/20-302/20-302.pdf) like arbitrary precision, infinite recursive steps, and hard attention). Then the local/windowed attention will just allow for more parallel computation if only local computations are needed. If this can be shown, it may be of less importance to perform one-to-one comparisions with GPT2 as there is theory to back up the expressiveness of the architecture/attention mechanism
3. **More ablation and feature studies** could be performed to improve performance. Currently this project is working on direct comparison (as one-to-one as possible) with GPT2, so more recent transformer advancements have not been implemented (Rotary or ALiBi embeddings, key-query embeddings, parameter sharing, token mixing, new initialization schemes, etc.). It is important to continue this direct comparison research as a matter of making sure the proposed attention mechanism works comparably while also implementing the latest techniques seperately especially if unique techniques work particularly well or are in tandem with specifically Additive Attention
4. **Infinite Context** as mentioned in the previous point, recent work into positional embeddings that can extrapolate to longer context sizes is making a lot of progress. Both local and global Additive Attention should extrapolate to longer sequences because it is simply a "learned softmax-ed weight sum of tokens" of differing lengths
5. **Reinforcement Learning** as noted by [OpenAI's Requests for Research](https://openai.com/blog/requests-for-research-2/) a good linear attention system (that can be represented as an RNN) is very attractive for RL rollouts. This project contends that specifically the local, mid-range, and global attention is much more *biologically plausible* as humans/animals are more likely to keep track or local, mid-range, and global sequence information (and their interactions) rather than considering all pairwise interactions in a sequence
6. **Explainability** A note which was unexplored in the original fastformer paper (because they didn't quite use the same formulation as this project) is how this system has *built-in explainability*, instead of pairwise interactions with full attention, each token is directly assigned an "importance" or "focus" scalar (see equations 1) which can directly be tracked for explaining what tokens the model paid attention to when predicting the next token. This should be explored with measurements and experiments!
8. **RNN Formulation** as stated in the positives section, and the math will show, we can represent Additive Attention as an RNN for fast $O(1)$ inference. Current work is primarily focusing on getting parallel large scale training done, but this is still necessary for the future of bringing this project and large language models to the public


## Rescaled Dot-Product

This project encounted some training instability where some preliminary investigation found that the attention/focus scores that were being generated were absurdly large/small. This may be of independent interest to training instability in large language models (say in [Megatron](https://arxiv.org/abs/1909.08053) or [PALM](https://arxiv.org/abs/2204.02311)) as the arguments shown in this section may offer a reasonable explaination for why there is training instability particularly in large models.

### Normal Scaled Dot-Product Attention 

<div></div>Let's consider the normal context where an "Attention score" $A_{ij}$ of a query and a key is calculated as follows 

$$
A_{ij} = {exp(Q_i \cdot K_j / \sqrt{d_{model}}) \over \sum\limits_{j= 0}^{N} exp(Q_i \cdot K_j / \sqrt{d_{model}})}
$$

<div></div>To break this down, we simply measure the "similarity" of Query vector $i$ with Key vector $j$ (measured through dot product), then scale by a factor of $1 \over \sqrt{d_{model}}$ (we will get back to this). To ensure a "limited attentional span" we apply a softmax (i.e. dividing the similarity score of Query $i$ with Key $j$ by that Query's similarities with all the other Keys, an analgous situation would be calculating "the weight of a test on the final grade") strengthening this "limited attentional span" effect with exponentiation where a strong/high similarity between Query $i$ and Key $j$ will get exploded to a very large number and very negative similarity scores will mapped to an exponentially small number.

### Why scale by $1 \over \sqrt{d_{model}}$?

<div></div>According to [Attention is All you Need](https://arxiv.org/abs/1706.03762) (Viswani et al. 2017), the reason is simple. Consider that $Q_i$ and $K_j$ are random normal vectors (which may seem reasonable given random initialization and the use of LayerNorm), it is easy to show then that the dot product of $Q_i$ with $K_j$ will have mean of 0 and std of $\sqrt{d_{model}}$ which is remedied simply by scaling by $1 \over \sqrt{d_{model}}$ bringing the std back to 1. The authors note that this can "(push) the softmax function into regions where it has extremely small gradients" when $d_{model}$ is large.

### What is the issue?

<div></div>While the reason provided for scaling by the original paper/authors is valid and makes sense, it only considers where $Q_i$ and $K_j$ are "random and normal". In fact, as training happens and parameters are updated through optimization/gradient descent, we can be more and more assured that $Q_i$ and $K_j$ will be neither be random nor normal!
</br>
</br>

- <div></div>On the matter of normality , even if LayerNorm is used to normalize the embedding vector $x_i$ before be transformed into $Q_i = W_q \boldsymbol{x_i}$ and $K_j = W_k \boldsymbol{x_i}$, these projections are likely not going to be normal after even mild training. For example, a particular $K_j$ vector may be learned to become especially large if the token it represents is particularly "important" for the entire sequence, thus maximizing its dot products with query vectors and violating normality

-  <div></div>On the matter of randomness, there is an especially problematic scenario when $Q_i == K_j$. This is another realistic scenario when there are highly deterministic token interactions like verb conjugation or subject-verb agreement where there is only one right answer and a token only needs to pay attention to one and only one previous token. Even if we assume normality, it is easy to show that this case will cause the dot-product of $Q_i$ with $K_j$ to have mean of $d_{model}$ (!!) (because $Q_i \cdot K_j = \sum\limits_{z=0}^{d_{model-1}} r_z^2$ where $r$ is a random normal variable which would likewise have mean of 1, also consider the case that $Q_i == -K_j$ when the there should be no alignment)

<div></div>This general argument for the second point applies even when $Q_i$ and $K_j$ are only *slightly* correlated, and at large scales with huge $d_{model}$ values it can be very easy for there to be spurious correlations between parts of $Q_i$ and $K_j$ after only mild training that will blow up the dot-product similarity. While there is no conclusive argument that this will necessarily cause loss spikes, this does create training instability for the simplified versions of attention explored in this project, and in general having extremely large or extremely small attention values that get more extreme with scale would likely create some kind of training issues that at the very least "(push) the softmax function into regions where it has extremely small gradients" (the original rationale for scaled dot-product attention).

### The fix

<div></div>This project simply enforces normality (to the extent possible) using an unparameterized LayerNorm, dividing by $d_{model}$ and not $\sqrt{d_{model}}$, and multiplying by a set constant $c$.  In rigorous terms, let us define this "rescaled dot-product" using the symbol " $\bar{\cdot}$ " where for two vectors $\boldsymbol{x}, \boldsymbol{y}$ of dimension $d_{model}$:

$$\boldsymbol{x}\ \bar{\cdot}\ \boldsymbol{y} = \left({ \boldsymbol{x} - E[\boldsymbol{x}] \over \sqrt{Var[\boldsymbol{x}]}} \cdot  {\boldsymbol{y} - E[\boldsymbol{y}] \over \sqrt{Var[\boldsymbol{y}]} }\right) *{c \over d_{model}}$$

<div></div>This enforces that even if $\boldsymbol{x} ==\boldsymbol{y}$, after normalization, their dot product will have mean $d_{model}$, which we divide by to bring the mean back to 1. Then, to allow for larger dot-product similarity values, we multiply by the set constant $c$ (15 seems to work fine and, after exponentiation in the softmax, e^15 should be larger than what anyone would reasonably need) to rescale the the size of the dot-product. Thus, this "recaled dot-product" will not produce larger dot product similarities when $d_{model}$ is larger. As a slight tangent, it should be recognized that LayerNorm/normalizing a vector does not make the vector a "normal" vector (where each element is drawn from the normal distribution). This doesn't seem to be a problem emperically though. This project will work on concrete experments to show this as well as the value of this technique in the future, however preliminary measurements find that this still keeps attention sparse (in fact the pre-softmax dot product similarities quickly approach 15 within the first few steps of training) and does effectively limit the maximum value for this dot product (only a max value of ~16 was ever found).

## Linear Explainable Attention in Parallel (LEAP) Math

LEAP is meant to replace the scaled dot-product Attention module. The principle concept of LEAP is that sequence information will be conferred between tokens using a "softmax-weighted cumulative sum". Cumulative sums are parallelizeable as explained by [this wikipedia on prefix sum aka cumulative sum](https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel) which is implemented with CUDA [as seen here](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html) (they claim that the primary operation for cumsum/prefix sum "typically proceeds at 'memcpy' speeds"), and of course cumulative sums are linear in complexity. The "softmax-weighting" is to maintain a kind of "Attention" where there can still be O(1) path length and also offers explainability (as we can see what tokens the model is paying attention to). We will present the equations and ideas in steps to try to motivate and explain an interpretation for LEAP.

### Focus weighting

<div></div>Normal full attention is pairwise between all tokens (thus giving quadratic complexity) allowing all tokens to attend to each other. However to make a biological plausibility argument, a human would likely not read/predict a token in a sequence by considering that token's interactions with all other tokens (and humans seem to do just fine with sequence information). More realistically, they would have a more "focused" kind of attention where their focus is drawn to specific important aspects of the sequence which they will use to make sense of reading/predicting the next token. We will calculate this "focus weight" $f_i$ for token embedding $\boldsymbol{x_i}$ using a self dot-product just like full attention to create a "focus logit" that is softmaxed over the sequence up to token $i$ (to maintain causality):

$$
(1)\ \ f_i = \textit{softmax}(F_i\ \bar{\cdot}\ F_i') = {exp(F_i\ \bar{\cdot}\ F_i') \over \sum\limits_{k=0}^i exp(F_k\ \bar{\cdot}\ F_k')}
$$

<div></div>where $F_i = W_{F}\boldsymbol{x_i}$ and $F_i' = W_{F'}\boldsymbol{x_i}$ and " $\bar{\cdot}$ " is the rescaled dot product explained in the section above. One can think of this equation as calculating "whether the token contains information that should be conferred to future tokens, and should thus receive more 'attention'/'focus' compared to the other previous tokens" which will be learned by the weight matrices $W_{F}$ and $W_{F'}$ (note we use two weight matrices/projections and calculate their dot-product to allow for a kind of flexible "soft-weights" that are dynamic at runtime).

### Softmax-weighted cumulative sum

<div></div>As the natural next step, we will multiply "values" $V_j = W_V\boldsymbol{x_j}$ (similar to full attention) by these "focus weights" so that more focus will be given to values tokens with corresponding high focus weights, and less focus will be given to values tokens with corresponding low focus weights, and then add these "focus weighted values" together to get a focus vector for token $i$ that contains all sequence information that the model "focuses on" up to and including token $i$. Let the function that computes these "focus vectors" be $\textit{Focus}(F, F', V)$ where each row  $\textit{Focus}(F, F', V)_i$ can be computed as

$$
(2)\ \ \textit{Focus}(F, F', V)\_{i} = \sum\limits_{j = 0}^{i}  {exp(F_j\ \bar{\cdot}\ F_j') \over \sum\limits_{k=0}^i exp(F_k\ \bar{\cdot}\ F_k')}*V_j
$$

<div></div>Note the fraction term is just the "focus weight" formula presented in Equation 1, however it is slightly different because we *need to recompute the softmax* at every $i$ so that all "focus scores" are properly normalized (where each exponentiated "focus logit" is divided by the sum of ALL focus logits up to and including $i$) which stops us from being able to precompute the focus weights. This is a problem time complexity wise as needing to recompute the softmax for every token $i$ would yield $O(N^2)$ or quadratic complexity. However, we can reuse old calculations simply by applying the Distributive Property of summation to factor out the denominator and distribute the $V_j$ term


$$
(3)\ \ \textit{Focus}(F, F', V)\_{i} = {{\sum\limits_{j = 0}^{i}  exp(F_j\ \bar{\cdot}\ F_j')}*V_j \over \sum\limits_{k=0}^i exp(F_k\ \bar{\cdot}\ F_k')}
$$

where both the numerator and denominator terms can be calculated with cumulative sums (if this is unclear, the RNN formulation in the later subsections might help). The mathematics here bears some resemblance to [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) where they also propose a cumulative sum formulation of attention, however their method does not and cannot use softmax (on top of the fact that structurally, they were aiming at maintaining pairwise attention). To the knowledge of the project authors, this "softmax weighted cumsum" idea where a softmax and cumsum steps are merged to achieve linear complexity is novel.

### Local Attention/Windowing

<div></div>All cumulative sums can be made "local" through a "sliding window technique" that keeps the time complexity linear by reusing computations while still maintaining parallel computation. This is novel as almost all "local Attention" methods (like the ones used in  [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/abs/2007.14062)) or more commonly, convolution will scale in time complexity with the size of the local window used.
</br>
</br>

- <div></div>Consider the following arbitrary local cumulative sum for some window size $w$ over some input sequence of tokens with the z-th token denoted $\boldsymbol{x_z}$ and some arbitrary function $f$:

$$
\boldsymbol{g_i} = \sum\limits_{z = max(i-w+1,\ 0)}^{i} f(\boldsymbol{x_z})
$$

- <div></div>The $max$ is just there so the lower limit doesn't go below 0. While it would take $O(N*w)$ time complexity to recalculate the summation for every token in a sequence of length $N$, we can simply write this as two cumulative sums that run in linear time in $N$ and can of course be parallelized as per a standard cumulative sum

$$
\boldsymbol{g_i} = \sum\limits_{z = 0}^{i} f(\boldsymbol{x_z}) - \sum\limits_{z = 0}^{max(i-w, -1)} f(\boldsymbol{x_z})
$$

- <div></div>Which is equivalent to the original local cumulative sum as per the definition of summation. The $max(i-w, -1)$ in the upper limit of the second term is just to stop any subtraction when $i < w$ and thus no "windowing" is needed. Now let us apply this technique to Equation 3 and define $\textit{w-Focus}(F, F', V)$ to be the function that calculates the "focus vectors" using only a local window of size $w$ where the $i$-th row is calculated as

$$
(4)\ \ \textit{w-Focus}\left(F, F', V\right)\_i =  {
	{\sum\limits_{j = 0}^{i}  exp(F_j\ \bar{\cdot}\ F_j')}*V_j - \sum\limits_{j = 0}^{max(i-w, -1)}  exp(F_j\ \bar{\cdot}\ F_j')*V_j
	\over
	\sum\limits_{k=0}^i exp(F_k\ \bar{\cdot}\ F_k') - \sum\limits_{k=0}^{max(i-w, -1)} exp(F_k\ \bar{\cdot}\ F_k')
}
$$

<div></div>This project will use different $w$ at each layer, which can be set as hyperparameters. The heuristic we will use based on preliminary testing is that $w_\ell =4(2)^\ell$ for layer number $\ell$ starting at 0. In general these window sizes can easily be tuned using hyperparameter sweeps/bayesian optimization, or just using domain knowledge as to whether there is local structure that could be benefited from being modeled separately (which is certainly the case in text, audio, and image). This project speculates multiple rounds of local, mid-range, and global attention should be performed.

### LEAP Equation

<div></div>While it may make sense to just add the $\textit{w-Focus}$ output to the inputs as a residual connection, we make one final addition to allow for "querying" to serve the simple practical purpose of allowing a token to ignore sequence information if it does not match the "query" defined as $Q_i = W_Q\boldsymbol{x_i}$ which allows for greater representational complexity as well as offering even more explainability (especially with the multi-layer and multihead formulation) where each token will uniquely only pay attention to certain focus vectors.

$$
(5)\ \ \textit{LEAP}_i = \sigma(Q_i\ \bar{\cdot}\ \textit{w-Focus}\left(F, F', V\right)_i) * \textit{w-Focus}\left(F, F', V\right)_i
$$

where $\sigma$ is the sigmoid function. To wrap up, all equations shown so far are only for one row vector at a time, though of course, in the implementation all the equations are applied to all rows in parallel using basic pytorch/CUDA tensor operations.

### RNN Formulation

<div></div>To show the linearity of LEAP as well as follow through on the O(1) time and memory at inference time claim, we will rewrite the LEAP equation and likewise the $\textit{w-Focus}$ equations equivalently as an RNN

$$
	\boldsymbol{s_{i}} = \boldsymbol{s_{i-1}} + exp(F_i\ \bar{\cdot}\ F_i')*V_i
$$

$$
	\begin{cases}
		\boldsymbol{s_{i}'} = \boldsymbol{s_{i-1}'} +exp(F_{i-w}\ \bar{\cdot}\ F_{i-w}')*V_{i-w}, & \text{if }\ i \geq w\\
		\boldsymbol{s_{i}'} = 0, & \text{if  }\ i < w
	\end{cases}
$$

$$
z_{i} = z_{i-1} + exp(F_i\ \bar{\cdot}\ F_i')
$$

$$
	\begin{cases}
		z_{i}' = z_{i-1}' + exp(F_{i-w}\ \bar{\cdot}\ F_{i-w}') , & \text{if  }\ i \geq w\\
		z_{i}' = 0, & \text{if  }\ i < w\\
	\end{cases}
$$

$$
	(6)\ \ \boldsymbol{g_i} = \left( Q_i \ \bar{\cdot}\  {\boldsymbol{s_i} - \boldsymbol{s_i'} \over z_i -z_i'} \right) *{\boldsymbol{s_i} - \boldsymbol{s_i'} \over z_i -z_i'}
$$

<div></div>Short Discussion: The RNN formulation should only be used at inference time to provide O(1) time and space complexity for generating the next token. This is because when training, equation 4 can be implemented with parallel cumulative sums to calculate each of the summation terms for each token index $i$. Also, as a slight tangent, to calculate $\boldsymbol{s_{i+1}'}$ and $z_{i+1}'$ you will technically need to keep a buffer of the previous *tokens* and not their embeddings, as those can be recomputed "on the fly". This technically means you need $O(N)$ space but NOT $O(N*d_{model})$ "space". Though in general, this going to be a trivial matter since almost every sequence task will store all previous sequence tokens somewhere (like for text generation, all previously generated text will be stored). So if we only consider the amount of RAM/VRAM needed, it is still $O(1)$, as the buffer of previous tokens could be stored on disk and can be retrieved efficently with cached IO calls (without any special IO management).


### Numerical Stability

<div></div>A numerical stability term should added to the denominator of all equations (we use 1e-5, though in general it should have little effect) in case any denominator gets very close to 0 and may cause floating point overflow (which will show up as NaN loss). This includes the normalizing equations of the rescaled dot-product.

## Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2 [paper for reference by Radford et al. (2019)](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf) where LEAP just replaces the scaled-dot product Attention module in a Transformer:

- Pre-norming with a layernorm before projecting to logits like GPT2
- Dropout of .1 on embeddings, feedforward, and attention layers like GPT2
- Learned positional embeddings as per [GPT1 paper by Radford et al. (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) which carries over to GPT2 (though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) were considered, but decided against because it would unfairly give an advantage to the model when compared against normal Transformers/gpt2 which uses learned absolute positional embeddings
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by Attention is All you Need, GPT1 and likewise GPT2
- Label smoothing of .1 ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762) is forgone because huggingface seems to oddly apply label smoothing during validation (so the loss that comes out when exponentiated would not be perplexity)
- Attention masking of pad tokens ([Attention is All you Need by Viswani et al. (2017)](https://arxiv.org/abs/1706.03762)) which is carried over to GPT2
- <div></div>Multihead Attention where LEAP is simply performed on down projected vectors of size $d_{model} \over n_{heads}$ in parallel with the same number of parameters as a single-head also as per Attention is All you Need by Viswani et al. (2017) which is carried over to GPT2

## Preliminary Results

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/preliminary_results.png?raw=True)

Plotted is the validation perplexity of the LEAP Transformer (blue) and GPT2 (orange) when trained on Wikitext-2 with a T5 tokenizer. The final test set perplexity of the LEAP transformer was 47.5 and the final test set perplexity of GPT2 was 59.7.

As we can see on this small scale experiment, the LEAP Transformer stably with less perplexity compared to even GPT2 (of the same size). Even though these results are preliminary, the long sequence length of 2048 should already be enough to test the abilities of this model as being better than an RNN like LSTMs as found by this [Scaling Laws paper](https://arxiv.org/abs/2001.08361) (Figure 7 finds that LSTM scaling bends at around 1M parameters, and at context lengths of >1000, the LSTM should be unable to compete). Furthermore, this model already beats both [Mogrifier LSTM](https://arxiv.org/abs/1909.01792v2) and [AWD LSTM](https://arxiv.org/abs/1708.02182v1) (when not using dynamic eval) on Wikitext-2 perplexity even though those models use >30M parameters (see the [leaderboard on paperswithcode](https://paperswithcode.com/sota/language-modelling-on-wikitext-2)).

**Speed:** The LEAP Tranformer training run took ~26 minutes while GPT2 took ~48 minutes (1.8x) which should become more pronounced at context lengths greater than 2048 and larger model sizes.

### Training details

All models were trained on a single NVIDIA GeForce RTX 2060 with batch sizes of 2. Code can be found in ``FastLM.ipynb``   which uses the [HuggingFace 🤗 Transfomers](https://huggingface.co/) specifically the [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer).

Model details: all models had a model dimension of 128 (feedforward dimension is 4x the model dimension), 4 attention heads, 6 layers and dropout probability of .1 on embedding, attention, and feedforward layers. The window sizes for Local Additive Attention were [4, 8, 16, 32, 64, 2028] as per the heurstic described.

Optimizer: [AdamW](https://arxiv.org/abs/1711.05101) with learning rate of 5e-4 to start and linear annealing, betas=(.9, .999) and weight decay = .01. Mixed precision with gradient clipping = 1 (max norm) is also used.

**Tuning** Very little tuning has been done to optimize performance, only learning rates of [1e-4, 5e-4, 1e-3] have been tried (5e-4 is best for both GPT and Additive Attention in preliminary experiments) and only a few different kind of window size settings have been explored (<20). No tuning on the test set was performed and the only evaluation on the test set performed was for the results shown.


## References
Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). Huggingface's transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_.

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., ... & Ahmed, A. (2020). Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_, _33_, 17283-17297.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. _arXiv preprint arXiv:1810.04805_.

Pérez, J., Marinković, J., & Barceló, P. (2019). On the turing completeness of modern neural network architectures. _arXiv preprint arXiv:1901.03429_.

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

