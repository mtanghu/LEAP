# Additive Attention Is All You Need?

This project **heavily** adapts Additive Attention first introduced in [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) specifically for the purposes of causal language modeling. We use a different structure and linearize the global attention mechanism originally described in a way that is parallelizable. Furthermore we use a different multi-head Additive Attention formulation as well add the novel contribution of linear local attention.

Positive aspects of this approach (some novel):

1. Most Linear Attention mechanisms cannot be used for parallel decoder language modeling or don't work that well (see Eleuther comments on ["Have you considered more efficient architectures or methods?](https://www.eleuther.ai/faq/))
2. The models considered in this project run faster than a standard Transformer when run with the same # of layers and layer sizes even on small sequence lengths (the math allows for *strongly parallelize-able* operations which is not always the case with linear attention)
3. Already integrated with [HuggingFace🤗 Transformers](https://huggingface.co/)
4. **Linear in time local attention**, this concept has not been seen before in the literature as local attention typically has to scale in time complexity with the size of the local window, which we use some math trickery to get around this (and still be parallelize-able). This gets around the issue that longer sequences will typically need bigger local attention windows, as well as the issue that local + global attention (previously explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/pdf/2004.05150.pdf)) may not have enough representational complexity at scale (not enough mid-range sequence modeling)
5. **Built-in Explainability**, while explainability is not supported yet in this project, as we'll see later, each token will be assigned an "importance weight" (which is softmax-ed) which can be used to explain what tokens the model is paying attention to, and which tokens are ignored similar to the explainability offered by the original [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) paper, though even more explainable as attention weights aren't pairwise anymore
6. **O(1) Path Length/Flexibility**, A great part about full attention Transformers the flexibility provided by the $O(1)$ path length. An example where many linear attention mechanisms would likely fail (ie. if they only use local/convolutional attention or time-decaying factors or a recurrent vector that will get overloaded with information over time) would be when there is "*task metadata*" at the beginning of the sequence. Example: "Read the following story paying special attention to how Alice treats Bob as you will write an essay on this after: \<very long story here\>". This task information may not make it all the way through the story and writing the essay with the previously mentioned approaches, but with this project's approach, tokens from the beginning of the sequence can directly transfer information to tokens at the end of the sequence with a $O(1)$ path length (like full-attention) through global Additive Attention
7. **O(1) Inference**, the math of Additive Attention can be represented as an RNN (while still maintaining the $O(1)$ path length). Thus because you only need the previous token's embeddings (i.e. $O(1)$ space) to calculate the next token (as per being an RNN) which only takes $O(1)$ computations with no matrix-matrix operations (all with respect to sequence length holding model size/dimension constant). This was which was originally shown in [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) to increase inference time performance by thousands and could potentially *allow large language models to run on edge devices like mobile phones or consumer laptops!*


This README will summarize Additive Attention and annotate a number of its details, show a unique linearization process/math (which allows for RNN formulation), show how this approach can be used for linear local attention, as well as preliminary results which show that Additive Attention is potentially comparable to full attention, and there is plenty of room for development!

## Usage & Development

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install (make sure you have [pytorch installed with CUDA](https://pytorch.org/get-started/locally/))

```bash
pip install fastformerLM
```

Then to use in python (setting the config how you want):
```python
from fastformer import FastformerForCausalLM, FastformerLMConfig

config = FastformerLMConfig(
    hidden_size = 128, # size of embeddings
    vocab_size = 32100, # number of tokens, if you have a tokenizer use len(tokenizer) instead
    n_positions = 2048, # max number of tokens to process at once
    n_layer = 6, # how many stacked decoder layers to use
    use_local_att = True, # whether to use windowed/local Additive Attention
    window_sizes = None, # window sizes to use for windowed/local Additive Attention for each layer (set automatically if None)
    n_heads = 4, # number of heads to use in multi-head attention
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
A more complete training example with a dataset, tokenization, and evaluations can be found at ``FastLM.ipynb`` in this repository which can be run with only 6GB of VRAM (i.e. GPU memory).

### Development
 If you want to contribute, (optionally) make a github issue and send in a pull request! A number possibilities for development and usage come to mind:

1. Additive Attention may work better on other & more specific NLP tasks (conversation, reasoning, genomics, speech/audio, vision, text-to-image, or even specific language domains) as per the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which especially applies to unique datasets. Other Linear Attention papers seem to use unique datasets too. This is especially the case with smaller dataset as Additive Attention does have stronger inductive bias.
2. **Theory** to show that local, mid-range, and global Additive Attention can replicate full Attention/is Turing Complete? -- This should be unequivocally True since it was already found to be true for [BigBird](https://arxiv.org/pdf/2004.05150.pdf) with local + random attention + global attention, where this project has local + mid-range + global attention which should be stronger (The proof will be tricky though). If this were true, we wouldn't necessarily need to worry about doing a one-to-one comparision with standard Transformers as the theory would support that this system is just as capable of modeling sequence information
3. **More ablation and feature studies** could be performed to improve performance. Currently this project is working on direct comparison (as one-to-one as possible) with GPT2, so more recent transformer advancements have not been implemented (Rotary or ALiBi embeddings, key-query embeddings, parameter sharing, token mixing, new initialization schemes, etc.). It is important to continue this direct comparison research as a matter of making sure the proposed attention mechanism works comparably while also implementing the latest techniques seperately especially if unique techniques work particularly well or are in tandem with specifically Additive Attention
4. **Infinite Context** as mentioned in the previous point, recent work into positional embeddings that can extrapolate to longer context sizes is making a lot of progress. Both local and global Additive Attention should have some ability to 
5. **Reinforcement Learning** as noted by [OpenAI's Requests for Research](https://openai.com/blog/requests-for-research-2/) a good linear attention system (that can be represented as an RNN) is very attractive for RL rollouts. This project contends that specifically the local, mid-range, and global attention is much more *biologically plausible* as humans/animals are more likely to keep track or local, mid-range, and global sequence information (and their interactions) as opposed to considering all pairwise interactions in a sequence
7. **Explainability** A note which was unexplored in the original fastformer paper (because they didn't quite use the same formulation as this project) is how this system has *built-in explainability*, instead of pairwise interactions with full attention, each token is directly assigned an "importance" scalar (see equations 1 and 2, $\alpha_{i}$ is the importance weight) which can directly be tracked for explaining what tokens were deemed as important for future predictions and what tokens weren't. This should be explored with some kind of experiments!
8. ***RNN Formulation*** as stated in the positives section, and the math will show, we can represent Additive Attention as an RNN for fast $O(1)$ inference. Current work is primarily focusing on getting parallel large scale training done, but this is still necessary for the future of bringing this project and large language models to the public.


## Brief Explanation of Additive Attention

<div></div>The general concept of Additive Attention is that is instead of allowing each embedded token to pairwise attend to every other embedded token which happens in normal full Attention (which is $O(N^2)$ complexity where $N$ is the sequence length), Additive Attention relies on “global attention vectors” which condense information about the entire sequence into a single vector through addition/summation (giving the name “Additive Attention”\*). A global attention vector then confers information about the entire sequence to individual embeddings through pointwise multiplying the global attention vector with each embedding vector. The specifics of this last step and other structural details are best explained [in the original paper]([https://arxiv.org/pdf/2108.09084.pdf](https://arxiv.org/pdf/2108.09084.pdf)). We will however dive deeper into the Additive Attention mechanism itself as we will need to adapt it for causal language modeling rather than classification (as was the purpose of the original paper).

This project considers the Additive Attentional mechanism described in [Wu et al. 2021](https://arxiv.org/pdf/2108.09084.pdf) to only be just the following equations (as all the other equations/steps do not apply to the sequence dimension):

- <div></div>Consider a sequence of embeddings $\boldsymbol{x_{i}}$ with $i$ from 1 to N

- <div></div>Get an “attention weight” $\alpha_{i}$ (which is just a scalar) for each embedding by projecting the embedding to a single dimension that will be scaled and softmax-ed over the sequence dimension, i.e.

$$
	(1)\ \alpha_i =  {exp(\boldsymbol{w}^T \boldsymbol{x_{i}} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{N} exp(\boldsymbol{w}^T \boldsymbol{x_{j}} / \sqrt{d_{model}})}
$$

- <div></div>Multiply the embeddings by their “attention weight” (so important embeddings are emphasized over unimportant embeddings which are pushed toward 0, note how this offers "explainability"), and sum over the sequence dimension to get a “global attention vector” ${\boldsymbol{g}}$ that contains information about the entire sequence, i.e.

$$ 
	(2)\ \boldsymbol{g} = \sum_{i=1}^{N} \alpha_{i} \boldsymbol{x_i}
$$

<div></div>Which is clearly $O(N)$ or linear in time complexity w.r.t the sequence length $N$. Note that this is also $O(1)$ in path length in the same way full attention is $O(1)$ path length as any token $\boldsymbol{x_{i}}$ can directly confer information to ${\boldsymbol{g}}$ without having to go through other tokens (and just like full attention, this is aided by the softmax which will clear away unimportant tokens by down weighting them). Slightly more mathematically, both full Attention and Additive Attention use a softmax-weighted sum of tokens (during decoding), full Attention just does this separately for each pairwise combination of previous tokens (as a tangent, this project will contend that a combination of local, mid-range, and global softmax-weighted sums will sufficiently replicate the sequence modeling of full (pairwise) Attention).
  
\* Not to be confused with [Additive Attention by Bahdanau et al. 2014](https://arxiv.org/abs/1409.0473v7)

## Additive Attention for Causal Language Modeling

Causal Language Modeling or decoder-based language modeling is where a language model is tasked with *generating* the next token given all previous tokens, though training is performed in parallel with all tokens present in training, BUT token embeddings are not allowed to receive information about future tokens. This restriction presents a challenge because, at a high level, a global attention vector that confers information about the entire sequence to each individual token embedding will certainly allow a token embedding to “see into the future” unduly. To remedy this, we need to create an equivalent sequence of global attention vectors, one for each token, that only contains sequence information **up to each token**.

- To do this rigorously, let's start by substituting equation (1) into equation (2)

$$
	(3)\ \boldsymbol{g} = \sum\limits_{\ell=1}^{N}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{N} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}*\boldsymbol{x_\ell}
$$


- <div></div>Now let us instead create $\boldsymbol{g_{i}}$, which would be the equivalent global attention vector for sequence information up to (and including) token $i$. This gives us:

$$
	(4)\ \boldsymbol{g_i} = \sum\limits_{\ell=1}^{i}  {exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) \over \sum\limits_{j=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}*\boldsymbol{x_\ell}
$$


- <div></div>Though we may have a time complexity issue. The original Additive Attention mechanism shown in equation (3) takes $O(N)$ time, so recalculating it for every token $i$ as equation (4) might suggest would yield a $O(N^2)$ time complexity. Furthermore, because of the nested summation in equation (4) it may seem impossible to reuse previous calculations to get a linear time complexity. However, in a style reminiscent** of [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) we can rewrite equation 4 by factoring out the denominator, i.e.

$$
	(5)\ \boldsymbol{g_i} =  {\sum\limits_{\ell=1}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) *\boldsymbol{x_\ell} \over \sum\limits_{\ell=1}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}})}
$$

- <div></div>Where the summation terms in the numerator and denominator for each $i$ can be calculated simply by cumulative sum. To elaborate on this and the linear complexity, we can write this as a kind of RNN as Katharpoulos et al. (2020) did.

$$
	(6)\ \boldsymbol{s_i} = \boldsymbol{s_{i-1}} +exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) *\boldsymbol{x_i}
$$

$$
	(7)\ z_i = z_{i-1} +exp(\boldsymbol{w}^T \boldsymbol{x_i} / \sqrt{d_{model}}) 
$$

$$
	(8)\ \boldsymbol{g_i} = {\boldsymbol{s_i} \over z_i}
$$

\*\*Note that Katharpoulos et al. didn't quite have the same process, they instead had to use a kernel to approximate the similarity scores of the keys and queries (and the associativity property) and likewise didn't and couldn't use a softmax. This Additive Attention math still uses a softmax, but the similarity between the two approaches/styles is how they both come from rewriting the equations and using some simplification techniques to achieve linear in complexity attention that can be represented as an RNN/cumulative sum with a vector that has sequence information in the numerator and a scalar normalization term in the denominator.

## Local Additive Attention (or Windowed Attention)

<div></div>While it may make sense to "generate a global attention vector for each token in a sequence", this runs into trouble when potentially more local information is needed (like part-of-speech, subject in a sentence, modifiers for nouns, correct conjugation for verbs, preposition agreement with nouns, etc.). To remedy this, we can simply apply the Additive Attention mechanism to a limited backward context. The principle for this is exactly the same as local scaled-dot product Attention where you can find [a diagram and code from lucidrains](https://github.com/lucidrains/local-attention) this is also explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf). Unfortunately though, local attention will cost $O(N*k)$ where $k$ is the size of the backwards context (or in other words, the number of previous tokens to attend to for each token), and it can be unclear whether this really improves over $O(N^2)$ (with full scaled dot product attention) when taking into consideration that certainly longer sequences (i.e. bigger $N$) will require larger $k$ to get the good performance (this is discussed in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) as a natural tradeoff). This would even apply to our linear Additive Attention method *if implemented naively*, again we can use some mathematical trickery (like in the previous section) to achieve $O(N)$ *independent* of $k$ (!!!) something that Katharpoulos et al. (2020) had not considered.<br/><br/>

- <div></div>First let us reconsider equation 5 (which is the Additive Attention equation) for a local/windowed version which only considers the previous $k$ tokens (including itself)

$$
	(9)\ \boldsymbol{g_i} =  {\sum\limits_{\ell=max(i-k+1,\ 0)}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) *\boldsymbol{x_\ell} \over \sum\limits_{\ell=max(i-k+1,\ 0)}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}})}
$$

- <div></div>Note that we need to use a $max$ in case the backwards context or "window" is non-existent or paritally non-existent for $i < k$. Then to calculate this with linear time complexity, we simply rewrite the sum, but this time, un-simplifying!


$$
	(10)\ \ 
\boldsymbol{g_i} =  {\sum\limits_{\ell=0}^{i}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) *\boldsymbol{x_\ell} - \sum\limits_{\ell=0}^{max(i-k,\ -1)}exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) *\boldsymbol{x_\ell}
		\over
\sum\limits_{\ell=0}^{i} exp(\boldsymbol{w}^T \boldsymbol{x_\ell} / \sqrt{d_{model}}) -\sum\limits_{\ell=0}^{max(i-k, -1)} exp(\boldsymbol{w}^T \boldsymbol{x_j} / \sqrt{d_{model}})}
$$

- Where again, we can easily keep track of all 4 summation terms using cumulative sums. To show this in a different way, we can rewrite this as an RNN again to also emphasize the linearity.

$$
	(11)\ \boldsymbol{s_{i}} = \boldsymbol{s_{i-1}} +exp(\boldsymbol{w}^T \boldsymbol{x_{i}} / \sqrt{d_{model}}) *\boldsymbol{x_{i}}
$$

$$
	(12)\ 
	\begin{cases}
		\boldsymbol{s_{i}'} = \boldsymbol{s_{i-1}'} +exp(\boldsymbol{w}^T \boldsymbol{x_{i-k}} / \sqrt{d_{model}}) *\boldsymbol{x_{i-k}}, & \text{if }\ i \geq k\\
		\boldsymbol{s_{i}'} = 0, & \text{if  }\ i < k
	\end{cases}
$$

$$
	(13)\ z_{i} = z_{i-1} +exp(\boldsymbol{w}^T \boldsymbol{x_{i}} / \sqrt{d_{model}}) 
$$

$$
	(14)\ 
	\begin{cases}
		z_{i}' = z_{i-1}' +exp(\boldsymbol{w}^T \boldsymbol{x_{i-k}} / \sqrt{d_{model}}) , & \text{if  }\ i \geq k\\
		z_{i}' = 0, & \text{if  }\ i < k\\
	\end{cases}
$$

$$
	(15)\ \boldsymbol{g_i} = {\boldsymbol{s_i} - \boldsymbol{s_i'} \over z_i -z_i'}
$$

Note that this should be doable with [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) too, though it seems like this is a novel idea that comes from an understanding that Additive Attention is global, and we specifically wanted to have some kind of local attention whereas Katharpoulos et al. (2020) came from a mathematical perspective that kernel-izing the similarity function should roughly approximate softmax similarity scores. Though because both this implementation and theirs rely on cumulative sums which may become overloaded with information after cumulative summing enough token embeddings on long token sequences, we suspect that their system could be improved with linear local attention too! However, this project suspects that actually this Additive Attention formulation will outperform their kernelized Attention because Additive Attention is still able to use a softmax to strongly prioritize Attention to important tokens.

<div></div>For implementation in a language model, we need to pick the right $k$ value, though this project proposes to consider different $k$ values for each layer. After some mild experimentation, the heuristic chosen for this project is $k_\ell = 4(2^{\ell})$ where $\ell$ is the hidden layer number (starting from 0), then the last layer will be "global attention" (unlimited window size). The intuition here would be that the lower levels are parsing lower-level semantic information that is accumulated with a final global attention layer. Note that because bigger models already use more layers, this should probably only help the scaling potential of this model. 

## Linear RNN or Parallel Cumulative Sum?

It may be confusing that there are two sets of equations for Additive Attention in both of the previous sections, however there is an important purpose.

- <div></div>**During training** when the entire sequence is presented, the key point of Transformers is that even though they have $O(N^2)$ complexity, the complexity comes from a matrix multiplications which are highly parallelizable on GPUs (and TPUs too of course). Lucky for us, a **cumulative sum is also parallelizable** as explained by [this wikipedia on prefix sum aka cumulative sum](https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel) which is implemented with CUDA [as seen here](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a16d7bc049ba8985dd89b10b3bcf0a8a3) which is directly called by Pytorch (used in this project) for cumulative sums (shown by [these CUDA profiles](https://github.com/pytorch/pytorch/issues/75240)). Thus the summation equations (5 and 10) make more sense as they can be directly translated to cumulative sums.

- <div></div>**During inference** each token is generated one at a time. Thus, at inference we can just focus on the linear (and non-parallel) RNN formulation which just requires us to keep track of the most recent $\boldsymbol{s_i}, \boldsymbol{s_i'}, z_i, z_i'$ (only 2 vectors and 2 scalars) to generate the next token as well as generate the next Additive Attention vectors (i.e. $\boldsymbol{s_{i+1}}, \boldsymbol{s_{i+1}'}, z_{i+1}, z_{i+1}'$). This only needs $O(1)$ complexity in terms of sequence length to generate new tokens, though the dimensionality of the model needs to be considered giving $O(d)$ complexity where $d$ is the width/hidden_size of the model.

<div></div>Slight tangent: In RNN mode for local attention, to calculate $\boldsymbol{s_{i+1}'}$ and $z_{i+1}'$ you will need to keep a buffer of the previous *tokens* and not their embeddings, as those can be recomputed "on the fly". This technically means you need $O(N)$ space but NOT $O(N*d)$ space. In general, this going to be a trivial matter since almost every sequence task will store all previous sequence tokens somewhere (like for text generation, all previously generated text will be stored). So of we only consider the amount of RAM/VRAM needed, it is still $O(1)$, as the buffer of previous tokens could be stored on disk and can be retrieved efficently with cached IO calls (without any special IO management).

## Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2:

- Ordering of residual connections, layer norms and dropouts follows [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf) by Shoeybi, Patwary & Puri et al. 2020. (shouldn't affect performance much, just stability)
- Learned positional embeddings ([Radford et al. 2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) we considered, but decided against because it would unfairly give an advantage to the model when compared against normal transformers/gpt2 which uses learned absolute positional embeddings.
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by GPT2.
- Label smoothing of .1 ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762)). (Though huggingface seems to oddly apply label smoothing during validation as well so later experiments will forgo label smoothing)
- Attention masking of pad tokens ([Viswani et al. 2017](https://arxiv.org/abs/1706.03762))
- <div></div>Scaling output projection (right before softmax) by $1 \over \sqrt{d_{model}}$ and no biases on linear layers (except for output projection) like [PALM](https://arxiv.org/pdf/2204.02311.pdf) (this shouldn't affect performance just stability)
- Due to some training instability, a residual connection and layer norm was placed in the middle of the full additive attention process (diagram on the way!)

## Results

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/preliminary_results.png?raw=True)

Plotted is the validation bits per character of the Additive Attention models (blue and orange) compared to full attention model (GPT2 in green) with the model sizes stated in the legend trained on Wikitext-2 using a T5 tokenizer with sequence lengths of 2048. The "Windowed Additive Attention" uses local Additive Attention explained in the "Local Additive Attention (or Windowed Attention)" section. 

As we can see on this small scale experiment, the Windowed Additive Attention strongly outcompetes the standard Additive Attention and converges faster with less perplexity compared to even GPT2 (of the same size). Even though these results are preliminary, the long sequence length of 2048 should already be enough to test the abilities of this model as being better than an RNN like LSTMs as found by this [Scaling Laws paper](https://arxiv.org/abs/2001.08361) (Figure 7 finds that LSTM scaling bends at around 1M parameters, and at context lengths of >1000, the LSTM should be unable to compete). Also because of the linear local attention, it may be more reasonable to believe that this model can scale up (as the combinations of local and global attentions should be able to model complex sequence information from short-range to long-range). Bigger tests are on the way!!

**Speed:** Both Additive Attention training runs took ~24 minutes while GPT2 took ~36 minutes (1.5x) which should become more pronounced at context lengths greater than 2048. Note again that it's tricky to do a lot better than full attention in terms of wall time because of highly optimized and paralellizable matrix multiplication algorithms.

### Training details

All models were trained on a single NVIDIA GeForce RTX 2060 with batch sizes of 2. Code can be found in ``FastLM.ipynb``   which uses the [HuggingFace 🤗 Transfomers](https://huggingface.co/) specifically the [Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer).

Model details: all models had a model dimension of 128 (feedforward dimension is 4x the model dimension), 4 attention heads, 6 layers and dropout probability of .1 on embedding, attention, and feedforward layers. The window sizes for Windowed Additive Attention were [4, 8, 16, 32, 64, 2028].

Optimizer: [AdamW](https://arxiv.org/abs/1711.05101) with learning rate of 5e-4 to start and linear annealing, betas=(.9,.999) and weight decay = .01. Mixed precision with gradient clipping = 1 (max norm) is also used.

**Tuning** Very little tuning has been done to optimize performance, only learning rates of [1e-4, 5e-4, 1e-3] have been tried (5e-4 is best for both GPT and Additive Attention in preliminary experiments) and a good way to set the window sizes has not been figured out.


## References
Wu, C., Wu, F., Qi, T., Huang, Y., & Xie, X. (2021). Fastformer: Additive attention can be all you need. _arXiv preprint arXiv:2108.09084_.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). Huggingface's transformers: State-of-the-art natural language processing. _arXiv preprint arXiv:1910.03771_.

Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. _arXiv preprint arXiv:2004.05150_.

Zaheer, M., Guruganesh, G., Dubey, K. A., Ainslie, J., Alberti, C., Ontanon, S., ... & Ahmed, A. (2020). Big bird: Transformers for longer sequences. _Advances in Neural Information Processing Systems_, _33_, 17283-17297.

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

Kaplan, Jared, et al. "Scaling laws for neural language models." _arXiv preprint arXiv:2001.08361_ (2020).

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_.
