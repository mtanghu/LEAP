# Linear Explainable Attention in Parallel (LEAP)

This project implements a novel linear attention mechanism based on "softmax-weighted cumulative sums" which has surprisingly favorable properties in computational complexity, explainability, and theoretical expressiveness. This project strongly believes that this linear attention mechanism can replace full attention with virtually no tradeoffs, if not actually having even better performance (because it's a more simple attention mechanism). This was originally inspired by adapting [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) (where they don't use any kind of cumulative sum)  for causal language modeling which we also implement with documentation and a comprehensive README that can be found in `src/leap/fastformerLM`. 

Reasons why LEAP may be able to replace full attention:

1. The models considered in this project run faster than a standard Transformer when run with the same # of layers and layer sizes even on small sequence lengths (the math allows for *highly parallelizeable* operations which is not always the case with linear attention) which offers extra ease of use

2. **Dot-product rescaling**, we find that the current dot-product attention scaling method can lead to training instability especially in this more simple form of attention. We introduce a new dot product scaling method that should stop dot product similarities from scaling with model size that *may help the training stability of full attention as well* but will allow LEAP to scale to large model sizes stably

3. **Linear in time local attention**, this concept has not been seen before in the literature as local attention typically has to scale in time complexity with the size of the local window. This project uses some simple mathematics and reuse of computations to get around this (and still be parallelizeable). This gets around the issue that longer sequences will typically need bigger local attention windows, but also builds upon the suprising strength of local + global attention (previously explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/abs/2007.14062) with the addition of random attention) with added mid-range sequence modeling. This project contends that this will give enough representational complexity to match full attention

4. **Built-in Explainability**, while explainability is not supported yet in this project, each token will be assigned an "focus weight" (which is softmaxed over the sequence) that can be used to explain what tokens the model is paying attention to, and which tokens are ignored. This is similar to the explainability offered by the original [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) paper, though more simplified

5. **O(1) Path Length/Flexibility**, A great strength of full attention Transformers is the flexibility provided by the $O(1)$ path length. An example where many linear attention mechanisms would likely fail (ie. if they only use local/convolutional attention or time-decaying factors or a recurrent vector that will get overloaded with information over time) would be when there is "*task metadata*" at the beginning of the sequence. Example: "Read the following story paying special attention to how Alice treats Bob as you will write an essay on this after: \<very long story here\>". This task information may not make it all the way through the story and writing the essay with the previously mentioned approaches, but with this project's approach, tokens from the beginning of the sequence can directly transfer information to tokens at the end of the sequence with a $O(1)$ path length (like full-attention) through global LEAP

6. **O(1) Inference**, the math of LEAP can be represented as an RNN (while still maintaining the $O(1)$ path length). Thus, you only need the previous token's embeddings (i.e. $O(1)$ space) to calculate the next token (as per being an RNN) which only takes $O(1)$ computations with no matrix-matrix operations (all with respect to sequence length holding model size/dimension constant). This was originally shown in [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) to increase inference time performance by thousands of times and could potentially *allow large language models to run on edge devices like mobile phones or consumer laptops!*

This README will describe a rescaled dot-product which may be of independent interest to full attention, summarize LEAP mathematics, and then show preliminary results which show that LEAP is potentially comparable to full attention, and there is plenty of room for development!

## Usage

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/) as a prerequisite)

```bash
pip install leap-transformer
```

Then to use in python (setting the config how you want):
```python
from leap import LeapForCausalLM, LeapConfig

config = LeapConfig(
    hidden_size = 128, # size of embeddings
    vocab_size = 32100, # number of tokens
    n_positions = 2048, # max number of tokens to process at once
    n_layer = 6, # how many stacked decoder layers to use
    use_local_att = True, # whether to use windowed/local LEAP
    window_sizes = None, # window sizes to use for windowed/local LEAP for each layer (set automatically if None)
    n_heads = 4, # number of heads to use in multi-head attention
    initializer_range = .02, # standard deviation for weight initialization
    hidden_dropout_prob = .1, # dropout value used for embeddings, attention, and feedforward layers
    rescale = 10 # what to rescale the focus values with, set lower if you have unstable/NaN loss
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

### Development and Contributing

This project needs your help! If you want to contribute, (optionally) make/address a github issue, or just send in a pull request! There will likely be a paper published for this where all contributors will be named, so please state your interested in this! Use these installation instructions so that you will have the latest repo and your edits will be reflected when you run the code

```bash
git clone https://github.com/mtanghu/LEAP.git
cd LEAP
pip install -e .
```


## Rescaled Dot-Product

This project encounted some training instability where some preliminary investigation found that the attention/focus scores that were being generated were absurdly large/small. This may be of independent interest to training instability in large language models (say in [Megatron](https://arxiv.org/abs/1909.08053) or [PALM](https://arxiv.org/abs/2204.02311)) as the arguments shown in this section may offer a reasonable explanation for why there is training instability particularly in large models.

### Normal Scaled Dot-Product Attention 

<div></div>Let's consider the normal context where an "Attention score" $A_{ij}$ of a query and a key is calculated as follows 

$$
A_{ij} = {exp(Q_i \cdot K_j / \sqrt{d_{model}}) \over \sum\limits_{j= 0}^{N} exp(Q_i \cdot K_j / \sqrt{d_{model}})}
$$

<div></div>To break this down, we simply measure the "similarity" of Query vector $i$ with Key vector $j$ (measured through dot product), then scale by a factor of $1 \over \sqrt{d_{model}}$ (we will get back to this). To ensure a "limited attention span" we apply a softmax (i.e. dividing the similarity score of Query $i$ with Key $j$ by that Query's similarities with all the other Keys) strengthening this "limited attention span" effect with exponentiation where a strong/high similarity between Query $i$ and Key $j$ will get exploded to a very large number and very negative similarity scores will mapped to an exponentially small number.

### Why scale by $1 \over \sqrt{d_{model}}$?

<div></div>According to [Attention is All you Need](https://arxiv.org/abs/1706.03762) (Viswani et al. 2017), the reason is simple. Consider that $Q_i$ and $K_j$ are random normal vectors (which may seem reasonable given random initialization and the use of LayerNorm), it is easy to show then that the dot product of $Q_i$ with $K_j$ will have mean of 0 and std of $\sqrt{d_{model}}$ which is remedied simply by scaling by $1 \over \sqrt{d_{model}}$ bringing the std back to 1. The authors note that this can "(push) the softmax function into regions where it has extremely small gradients" when $d_{model}$ is large.

### What is the issue?

<div></div>While the reason provided for scaling by the original paper/authors is valid and makes sense, it only considers where $Q_i$ and $K_j$ are "random and normal". In fact, as training happens and parameters are updated through optimization/gradient descent, we can be more and more assured that $Q_i$ and $K_j$ will be neither be random nor normal!
</br>
</br>

- <div></div>On the matter of normality, even if LayerNorm is used to normalize the embedding vector $x_i$ before be transformed into $Q_i = W_q \boldsymbol{x_i}$ and $K_j = W_k \boldsymbol{x_i}$, these projections are likely not going to be normal after even mild training. For example, a particular $K_j$ vector may be learned to become especially large if the token it represents is particularly "important" for the entire sequence, thus maximizing its dot products with query vectors and violating normality

-  <div></div>On the matter of randomness, there is an especially problematic scenario when $Q_i == K_j$. This is another realistic scenario when there are highly deterministic token interactions like verb conjugation or subject-verb agreement where there is only one right answer and a token only needs to pay attention to one and only one previous token. Even if we assume normality, it is easy to show that this case will cause the dot-product of $Q_i$ with $K_j$ to have mean of $d_{model}$ (!!) (because $Q_i \cdot K_j = \sum\limits_{z=0}^{d_{model-1}} r_z^2$ where $r$ is a random normal variable which would likewise have mean of 1, also consider the case that $Q_i == -K_j$ when the there should be no alignment)

<div></div>This general argument for the second point applies even when $Q_i$ and $K_j$ are only *slightly* correlated, and at large scales with huge $d_{model}$ values it can be very easy for there to be spurious correlations between parts of $Q_i$ and $K_j$ after only mild training that will blow up the dot-product similarity. While there is no conclusive argument that this will necessarily cause loss spikes, this does create training instability for the simplified versions of attention explored in this project, and in general having extremely large or extremely small attention values that get more extreme with scale would likely create some kind of training issues that at the very least "(push) the softmax function into regions where it has extremely small gradients" (the original rationale for scaled dot-product attention).

### The fix

<div></div>This project simply enforces normality (to the extent possible) using an unparameterized LayerNorm, dividing by $d_{model}$ and not $\sqrt{d_{model}}$, and multiplying by a set constant $c$.  In rigorous terms, we will replace the dot product of any two vectors $\boldsymbol{x} \cdot \boldsymbol{y}$ of dimension $d_{model}$ with:

$${\textit{Norm}(\boldsymbol{x}) \cdot \textit{Norm}(\boldsymbol{y}) \over d_{model} / c} = \left({ \boldsymbol{x} - E[\boldsymbol{x}] \over \sqrt{Var[\boldsymbol{x}]}} \cdot  {\boldsymbol{y} - E[\boldsymbol{y}] \over \sqrt{Var[\boldsymbol{y}]} }\right) *{c \over d_{model}}$$

<div></div>This enforces that even if $\boldsymbol{x} ==\boldsymbol{y}$, after normalization, their dot product will have mean $d_{model}$, which we divide by to bring the mean back to 1. Then, to allow for larger dot-product similarity values, we multiply by the set constant $c$ (10 seems to work well and, after exponentiation in the softmax, e^10 should be larger than what anyone would reasonably need) to rescale the the size of the dot-product. Thus, this "rescaled dot-product" will not produce larger dot product similarities when $d_{model}$ is larger. As a slight tangent, it should be recognized that LayerNorm/normalizing a vector does not make the vector a "normal" vector (where each element is drawn from the normal distribution). This doesn't seem to be a problem empirically though. This project will work on concrete experiments to show this as well as the value of this technique in the future, however preliminary measurements find that this still keeps attention sparse (in fact the pre-softmax dot product similarities quickly approach 10 within the first few steps of training) and does effectively limit the maximum value for this dot product (only a max value of ~12 was ever found).

## Linear Explainable Attention in Parallel (LEAP) Math

LEAP is meant to replace the scaled dot-product Attention module. The principle concept of LEAP is that sequence information will be conferred between tokens using a "weighted cumulative sum" that represents what tokens the model is "focusing on". Cumulative sums are parallelizeable as explained by [this wikipedia on prefix sum aka cumulative sum](https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel) which is implemented with CUDA [as seen here](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html) (they claim that the primary operation for cumsum/prefix sum "typically proceeds at 'memcpy' speeds"), and of course cumulative sums are linear in complexity. The "softmax-weighting" is to maintain a kind of "Attention" where there can still be O(1) path length and also offers explainability (as we can see what tokens the model is paying attention to). We will present the equations and ideas in steps to try to motivate and explain an interpretation for LEAP.

### Focus weighting

Normal full attention is pairwise between all tokens (thus giving quadratic complexity) allowing all tokens to attend to each other. However to make a biological plausibility argument, a human would likely not read/predict a token in a sequence by considering that token's interactions with all other tokens (and humans seem to do just fine with sequence information). More realistically, a human would have a more "focused" kind of attention where their focus is drawn to specific important aspects of the sequence which they will use to make sense of reading/predicting the next token. 

<div></div>To ground this in terms of token sequences, let $E$ by a matrix with where the rows are the token embeddings denoted $E_i$ for the $i$-th embedding vector. Then to implement "focus", let us consider a column vector of "focus weights" $f$ with $f_i$ signifying whether embedding $E_i$ should have high focus/attention (and thus have a big $f_i$) or low focus/attention (and thus have a small $f_i$. Let us define the "focus" (of a model) at position $i$ as the follow function:

$$
(1)\ \ \textit{Focus}(f, E)\_{i} = {\sum\limits_{j = 0}^{i}  f_j * E_j \over \sum\limits_{j=0}^i f_j}
$$

<div></div>The idea is that at position $i$, the tokens that the models is "focusing on" can simply be represented as a weighted sum of the embeddings at each previous position $E_j$ weighted by the scalar focus weight $f_j$. Then to ensure a "limited focus" or a "limited attention span", we simply divide by the sum of these focus weights. Both the numerator and denominator summation terms can be calculated using cumulative sums so that we have have linear complextiy with performing the focus calculation on all $i$ (if this is confusing, just continue as the later sections may help show this idea in different ways). Note that if $f$ was calculated as the exponentiation of a "focus logits" vector $l$ (i.e. $f = exp(l)$) which we will do later, this would be a "softmax-weighted cumulative sum" as we can rewrite equation 1 with a softmax:

$$
\textit{causal-softmax}(i, l)_j =  {exp(l_j) \over \sum\limits_{j=0}^i exp(l_j)}
$$

$$
\textit{Focus}(f, E)\_{i} = {\sum\limits_{j = 0}^{i} \textit{causal-softmax}(i, l) _j* E_j}
$$

<div></div>This concept of the $\textit{causal-softmax}$ is the primary innovation that is hidden away inside equation 1, but we will show here to elucidate why this concept may not have been explored before. Subsituting in the $\textit{causal-softmax}$ definition we see

$$
\textit{Focus}(f, E)\_{i} = {\sum\limits_{j = 0}^{i} {exp(l_j) \over \sum\limits_{k=0}^i exp(l_k)}* E_j}
$$

<div></div>which has a nested summation (note the limits in the denominator of the inner summation were changed from $j$ to $k$ since $j$ is already used in the limits of the outer summation). This nested summation presents a time complexity issue for calculation $\textit{Focus}(f, E)$ for all $i$ where even if we use a cumulative sum to calculate the inner summation in the denominator, the outer summation would still need to be recalculated for every $i$ thus yielding $O(N^2)$ or quadratic complexity for token sequence length of $N$. This can be rectified simply by applying the distributive property (to reuse previous computation) of summations to factor out the denominator.

$$
\textit{Focus}(f, E)\_{i} = {\sum\limits_{j = 0}^{i} {exp(l_j) \over \sum\limits_{k=0}^i exp(l_k)}* E_j} = {\sum\limits_{j = 0}^{i} exp(l_j)* E_j \over \sum\limits_{j=0}^i exp(l_j)} 
$$

<div></div>where as stated before, both the numerator and denominator calculated using cumulative sums when performing the focus calculation on all $i$.

### Local Attention/Windowing

All cumulative sums can be made "local" through a "sliding window technique" that keeps the time complexity linear by reusing computations while still maintaining parallel computation. This is novel as almost all "local Attention" methods (like convolution or the ones used in Longformer and BigBird) will scale in time complexity with the size of the local window used.
</br>
</br>

- <div></div>Consider the following arbitrary local cumulative sum for some window size $w$ over some input sequence of embedded tokens with the $j$-th token denoted $E_j$ and some arbitrary function $f$:

$$
\boldsymbol{g_i} = \sum\limits_{j = max(i-w+1,\ 0)}^{i} f(E_j)
$$

- <div></div>The $max$ is just there so the lower limit doesn't go below 0. While it would take $O(N*w)$ time complexity to recalculate the summation for every token in a sequence of length $N$, we can simply write this as two cumulative sums that run in linear time in $N$ and can of course be parallelized as per a standard cumulative sum

$$
\boldsymbol{g_i} = \sum\limits_{z = 0}^{i} f(\boldsymbol{x_z}) - \sum\limits_{z = 0}^{max(i-w, -1)} f(\boldsymbol{x_z})
$$

- <div></div>Which is equivalent to the original local cumulative sum as per the definition of summation. Note the $max(i-w, -1)$ in the upper limit of the second term is just to stop any subtraction when $i < w$ and thus no "windowing" is needed. Now let us apply this technique to Equation 3 and define $\textit{w-Focus}(f, E)$ to be the function that calculates the "focus vectors" using only a local window of size $w$ where the $i$-th row is calculated as

$$
(2)\ \ \textit{w-Focus}(f, E)_i =  {
	\sum\limits_{j = 0}^{i} f_j*E_j - \sum\limits_{j = 0}^{max(i-w, -1)} f_j*E_j
	\over
	\sum\limits_{k=0}^i f_j - \sum\limits_{k=0}^{max(i-w, -1)} f_j
}
$$

<div></div>As to the question of what $w$ to use, this project will use different $w$ at each layer which can be set as hyperparameters. The heuristic we will use based on preliminary testing is that $w_\ell = 2(2)^\ell$ for layer number $\ell$ starting at 0, though both the first and last layers will be reserved for global attention (i.e. $w_0 = N, w_{L-1} = 0$ where $L$ is the number of layers). This heuristic should work well enough for most applications, though in general these window sizes can easily be tuned using hyperparameter sweeps/bayesian optimization, or just using domain knowledge as to whether there is local structure that could be benefited from being modeled separately (which is certainly the case in text, audio, and image).

### LEAP Equation

<div></div> We put this all together using a Queries, Keys and Values formulation as per normal scaled dot-product attention. Letting the token embeddings be denoted $\boldsymbol{x_i}$ we first calculate Queries, Keys, and Values as normal though with the addition of a embedding $F$ used to generate focus weights:

$$
Q_i = \textit{Norm}(W_Q\boldsymbol{x_i})\\
F_i = \textit{Norm}(W_F\boldsymbol{x_i})\\
K_i = \textit{Norm}(W_K\boldsymbol{x_i})\\
V_i = W_V\boldsymbol{x_i}
$$

We apply norming to $Q, F, K$ as per the "Rescaled Dot-Product" section as they will be used in dot-products. The first example of this is when we calculate the focus weights column vector $f$:

$$
f_i = exp\left({F_i \cdot K_i \over d_{model} / c}\right)
$$

A "self dot-product" is used to calculate the focus weights so that this weight would be dynamic at runtime (potentially increasing flexibility just like normal dot-product attention). Now we can form the leap equation as follows

$$
(3)\ \ \textit{LEAP}_i = \sigma\left({Q_i \cdot \textit{w-Focus}(f, K)_i \over d_{model} / c}\right) * \textit{w-Focus}(f, V)_i
$$

where $\sigma$ is the sigmoid function and $LEAP$ is the attention output. To offer an interpretation of this equation: the Queries, Keys and Values have the same meaning as they do in full scaled dot-product attention, which is that a Query vector is what the token at index $i$ is "looking for", a Key vector is what 'kind' of information the token at index $i$ contains, and a Values vector is the information itself. At index $i$ the idea is to measure the (re-scaled) dot-product similarity of what the token is "looking for" (the query) with the kind of token information the model focused on (the focused keys). If the (re-scaled) dot-product similarity is high (i.e. what the token is looking for matches what the 'kind' of information focused on) then the sigmoid will output a value close to 1 to allow the information the model focused on (the values) to output. On the other hand, if the (re-scaled) dot-product similarity is low, the sigmoid will output a value close to 0 to stop the values information focused on to be output. A prototypical example of the former case would be pronoun filling, Ex. "Alice gave Bob a toy, and he was very happy". The query for the "he" token would be to look for "a male subject or object" where one of the attention heads should focus on "Bob" and should likewise produce a keys vector that encodes "a male subject or object" where then the dot-product similarity should be high and allow the "Bob" encoded in the focused values to be output (thus making sense of the pronoun "he" as referring to Bob). To clarify about "attention heads", Multihead Attention is used and implemented just like normal scaled-dot-product attention where instead of just having a single set of Queries, Keys, and Values, instead we have multiple Queries, Keys, and Values vectors are generated for each token so that there can be "multiple focuses" (each head would have its own "focus"/attention).

Note: all equations shown so far are only for one row vector at a time, though of course, in the implementation all the equations are applied to all rows in parallel using basic pytorch/CUDA tensor operations.

### RNN Formulation

<div></div>To show the linearity of LEAP as well as follow through on the O(1) time and memory at inference time claim, we will rewrite the LEAP equation and likewise the $\textit{w-Focus}$ equations equivalently as an RNN

$$
	\boldsymbol{k_{i}} = \boldsymbol{k_{i-1}} + exp\left({F_i \cdot K_i \over d_{model} / c}\right)*K_i
$$

$$
	\begin{cases}
		\boldsymbol{k_{i}'} = \boldsymbol{k_{i-1}'} + exp\left({F_{i-w} \cdot K_{i-w} \over d_{model} / c}\right)*K_{i-w}, & \text{if }\ i \geq w\\
		\boldsymbol{k_{i}'} = 0, & \text{if  }\ i < w
	\end{cases}
$$

$$
	\boldsymbol{v_{i}} = \boldsymbol{v_{i-1}} + exp\left({F_i \cdot K_i \over d_{model} / c}\right)*V_i
$$

$$
	\begin{cases}
		\boldsymbol{v_{i}'} = \boldsymbol{v_{i-1}'} + exp\left({F_{i-w} \cdot K_{i-w} \over d_{model} / c}\right)*V_{i-w}, & \text{if }\ i \geq w\\
		\boldsymbol{v_{i}'} = 0, & \text{if  }\ i < w
	\end{cases}
$$

$$
z_{i} = z_{i-1} + exp\left({F_i \cdot K_i \over d_{model} / c}\right)
$$

$$
	\begin{cases}
		z_{i}' = z_{i-1}' + exp\left({F_{i-w} \cdot K_{i-w} \over d_{model} / c}\right) , & \text{if  }\ i \geq w\\
		z_{i}' = 0, & \text{if  }\ i < w\\
	\end{cases}
$$

$$
	(6)\ \ \boldsymbol{g_i} = \left({Q_i \cdot  {\boldsymbol{k_i} - \boldsymbol{k_i'} \over z_i -z_i'} \over d_{model}/c}\right) *{\boldsymbol{v_i} - \boldsymbol{v_i'} \over z_i -z_i'}
$$

<div></div>Note: The RNN formulation should only be used at inference time to provide O(1) time and space complexity for generating the next token. This is because when training, equation 4 can be implemented with parallel cumulative sums to calculate each of the summation terms for each token index $i$. Also, as a slight tangent, to calculate $\boldsymbol{s_{i+1}'}$ and $z_{i+1}'$ you will technically need to keep a buffer of the previous *tokens* and not their embeddings, as those can be recomputed "on the fly". This technically means you need $O(N)$ space but NOT $O(N*d_{model})$ "space". Though in general, this going to be a trivial matter since almost every sequence task will store all previous sequence tokens somewhere (like for text generation, all previously generated text will be stored). So if we only consider the amount of RAM/VRAM needed, it is still $O(1)$, as the buffer of previous tokens could be stored on disk and can be retrieved efficiently with cached IO calls (without any special IO management).


### Numerical Stability

<div></div>A numerical stability term should added to the denominator of all equations (we use 1e-5, though in general it should have little effect) in case any denominator gets very close to 0 and may cause floating point overflow (which will show up as NaN loss). This includes the normalizing equations of the rescaled dot-product.

## Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2 [paper for reference by Radford et al. (2019)](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf) where LEAP just replaces the scaled-dot product Attention module in a Transformer:

- Pre-norming with a layernorm before projecting to token logits like GPT2
- Dropout of .1 on embeddings, feedforward, and attention layers like GPT2
- GELU activation is used in the feedforward layer like GPT2
- Learned positional embeddings as per [GPT1 paper by Radford et al. (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) which carries over to GPT2 (though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) were considered, but decided against because it would unfairly give an advantage to the model when compared against normal Transformers/gpt2 which uses learned absolute positional embeddings
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by Attention is All you Need, GPT1 and likewise GPT2
- Label smoothing of .1 ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762) is forgone because huggingface seems to oddly apply label smoothing during validation (so the loss that comes out when exponentiated would not be perplexity)
- Attention masking of pad tokens ([Attention is All you Need by Viswani et al. (2017)](https://arxiv.org/abs/1706.03762)) which is carried over to GPT2
- <div></div>Multihead Attention where LEAP is simply performed on down projected vectors of size $d_{model} \over n_{heads}$ in parallel with the same number of parameters as a single-head also as per Attention is All you Need by Viswani et al. (2017) which is carried over to GPT2
- The only slight difference is that no biases are used in any of the linear projection layers like [PALM](https://arxiv.org/abs/2204.02311) as it fits with the theme of the rescaled dot-product (to keep pre-attention logits low) for increased training stability. This shouldn't affect modeling performance much (if not decreasing performance) in the comparison against GPT2

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

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2022). Palm: Scaling language modeling with pathways. _arXiv preprint arXiv:2204.02311_.

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

