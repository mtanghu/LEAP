# Linear Explainable Attention in Parallel (LEAP)

This project implements a novel linear attention mechanism based on "softmax-weighted cumulative sums" which has surprisingly favorable properties in computational complexity, explainability, and theoretical expressiveness. This project strongly believes that this linear attention mechanism can replace full attention with virtually no tradeoffs, if not actually having even better performance (because it's a more simple attention mechanism). This was originally inspired by adapting [Fastformer: Additive attention can be all you need](https://arxiv.org/abs/2108.09084) by Wu et al. (2021) (where they don't use any kind of cumulative sum)  for causal language modeling which we also implement with documentation and a comprehensive README that can be found in `src/leap/fastformerLM`. 

Reasons why LEAP may be able to replace full attention:

1. The models considered in this project run **faster** than a standard Transformer of the same size even on small sequence lengths (the math allows for *highly parallelizeable* operations which is not always the case with linear attention) which offers high ease of use

2. **Dot-product rescaling**, we find that the current dot-product attention scaling method can lead to training instability especially in this more simple form of attention. We introduce a new dot product scaling method that should stop dot product similarities from scaling with model size that *may help the training stability of full attention as well* but will allow LEAP to scale to large model sizes stably

3. **Linear in time local attention**, this concept has not been seen before in the literature as local attention typically has to scale in time complexity with the size of the local window. This project uses some simple mathematics and reuse of computations to get around this (and still be parallelizeable). This gets around the issue that longer sequences will typically need bigger local attention windows, but also builds upon the surprising strength of local + global attention (previously explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/abs/2007.14062) with the addition of random attention).

4. **Built-in Explainability**, while explainability is not supported yet in this project, each token will be assigned an "focus weight" (which is softmaxed over the sequence) that can be used to explain what tokens the model is paying attention to, and which tokens are ignored. This is similar to the explainability offered by the original [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) paper, though more simplified

5. **O(1) Path Length/Flexibility**, A great strength of full attention Transformers is the flexibility provided by the $O(1)$ path length. An example where many linear attention mechanisms would likely fail (ie. if they only use local/convolutional attention or time-decaying factors or a recurrent vector that will get overloaded with information over time) would be when there is "*task metadata*" at the beginning of the sequence. Example: "Read the following story paying special attention to how Alice treats Bob as you will write an essay on this after: \<very long story here\>". This task information may not make it all the way through the story and writing the essay with the previously mentioned approaches, but with this project's approach, tokens from the beginning of the sequence can directly transfer information to tokens at the end of the sequence with a $O(1)$ path length (like full-attention) through global LEAP

6. **O(1) Inference**, the math of LEAP can be represented as an RNN (while still maintaining the $O(1)$ path length). Thus, you only need the previous token's embeddings (i.e. $O(1)$ space) to calculate the next token (as per being an RNN) which only takes $O(1)$ computations with no matrix-matrix operations (all with respect to sequence length holding model size/dimension constant). This was originally shown in [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) to increase inference time performance by thousands of times and could potentially *allow large language models to run on edge devices like mobile phones or consumer laptops!*

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
    n_head = 4, # number of heads to use in multi-head attention
    initializer_range = None, # variance for weight initialization, defaults to 1 / sqrt(hidden_size)
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


## Brief LEAP description

The math tricky and overly verbose/complicated at the moment but can be found in this repo with a write-up  [here](https://github.com/mtanghu/LEAP/blob/main/src/leap/README.md). As stated the general concept is just to have a cumulative sum of the sequence that is weighted with values that are passed through a softmax over the sequence length (done causally though). What will be described here are just some high level details.

### Why cumulative sum?

Cumulative sums were used reasonably successfully in previous linear attention mechanisms like [Linear Transformers](https://arxiv.org/pdf/2006.16236.pdf) though they don't use the *parallel* cumulative sum that can be run in logarithmic time (w.r.t. sequence length) as noted by [Performer](https://arxiv.org/abs/2009.14794). This can be seen in the following circuit diagram (from [wikipedia prefix sum page](https://en.wikipedia.org/wiki/Prefix_sum)). This means that a model of this kind could actually **proportionally run faster on longer sequences** (i.e. doubling the sequence length only increases the wall time by less than double).


![alt text](https://upload.wikimedia.org/wikipedia/commons/8/81/Prefix_sum_16.svg)

Where each wire represents an element in the sequence as input (coming from the top) and where the output of each wire the cumulative sum up to that element in the sequence. Luckily this is already implemented by CUDA [as seen here](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html) where they report that the cumulative sum operation runs about as fast as copying! What might set this off as being a good choice for sequence modelling is how the diagram almost shows a kind of "residual connections through time" in a way that seems vaguely neural.

The concept for LEAP is just to weight each element in the sequence before cumulative summing as a kind of "attention" or "focus". This implemented in a multihead way with queries, keys, and values and is meant to be something of an analog to full attention.

### Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2 [paper for reference by Radford et al. (2019)](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf) where LEAP just replaces the scaled-dot product Attention module in a Transformer:

- Pre-norming with a layernorm before projecting to token logits like GPT2
- GELU activation is used in the feedforward layer like GPT2
- Learned positional embeddings as per [GPT1 paper by Radford et al. (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) which carries over to GPT2 (though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) were considered, but decided against because it would unfairly give an advantage to the model when compared against normal Transformers/gpt2 which uses learned absolute positional embeddings. Just as a note, positional embeddings are still "needed" as a cumulative sum would not necessarily encode position information.
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by Attention is All you Need, GPT1 and likewise GPT2
- <div></div>Multihead Attention where LEAP is simply performed on down projected vectors of size $d_{model} \over n_{heads}$ in parallel with the same number of parameters as a single-head also as per Attention is All you Need by Viswani et al. (2017) which is carried over to GPT2
- The only slight difference is that biases are not used in the attention projection like [PALM](https://arxiv.org/abs/2204.02311) as it fits with the theme of the rescaled dot-product (to keep pre-attention logits low) for increased training stability. This shouldn't affect modeling performance much (if not decreasing performance) in the comparison against GPT2

## Scaling Experiment

Following landmark papers [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf) which has been revisited by [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) we hope to show the scaling behavior of LEAP and how it's quite comparable to a vanilla Transformer like GPT2. Note that as found by [Scaling Laws vs Model Architectures](https://arxiv.org/pdf/2207.10551.pdf), few to no models can match the scaling performance of Transformers. The experiment shown are done on much less data and much less compute, but at least preliminarily show LEAP's capabilities. 

![alt text](https://raw.githubusercontent.com/mtanghu/LEAP/main/Experiments/wikitext103_powerlaws.png)
 The compute scaling law (left) is in line with [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf)  which reported a alpha/exponent of around -.05 which should reasonably validate this experimental setup where FLOPs are estimated the same way. Note that if the FLOPs approximation used was applied to LEAP (where the sequence length quadratic complexity is just ignored) than LEAP would just use the same amount of FLOPs as GPT2 on equivalently sized models and dataset size.

The parameters scaling law (right) has a higher alpha that what is reported in [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf) of -.076 because data and parameters were scaled in tandem (for speed and also to be closer to compute optimal). Only non-embedding parameters are reported following [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf) especially because the embedding parameters were a very significant proportion of the parameters. Following [Scaling Laws vs Model Architectures](https://arxiv.org/pdf/2207.10551.pdf), this test is meant to robustly compare a rather "exotic" architecture like LEAP to vanilla Transformers especially as "exotic" architectures can often get away with just having their hyperparameters/architectures tuned to match vanilla Transformer performance while not having the highly desirable scaling potential.

### Enwik8 second test

We try a similar test on the byte-level Enwik8 dataset to see if results transfer. Scaling laws are typically not studied for byte-level datasets (since tokenizers are generally more effective than either byte-level or word-level especially in terms of compute efficiency).

![alt text](https://raw.githubusercontent.com/mtanghu/LEAP/main/Experiments/enwik8_powerlaws.png)
Again we see LEAP about matches GPT2, though there does seem to be some "bending" of the curve on the two largest tests for LEAP. This is concerning though bigger tests will have to be done to get a conclusive answer. Especially given how noisy this test seems to be (the GPT2 curve is completely wild). Also interestingly enough compared to the [paperswithcode enwik8 leaderboard](https://paperswithcode.com/sota/language-modelling-on-enwiki8) the models trained here should be comparable in size yet the don't have nearly the same BPC as the majority of the leaderboard (i.e. not even close to being below 1).


## Training details

Exact training details and logs can be found in `/Experiments/Scaling.ipynb` of this notebook. 


- **Dataset:** subsets of Wikitext-103 so that the number of tokens would match the recommendation of  [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf) where the (# parameters) is directly proportional to the (# tokens). The largest test uses the shown in the figure does use the entirety of Wikitext-103. All the same is true for the enwik8 dataset just that the test and validations splits are created manually at a random 5% of the data each
- **Tokenizer** a word-level tokenizer was used, but due to memory and compute constraints, the vocab size was limited to 8192. This means that the losses shown cannot be directly compared to Wikitext-103 benchmarks, but shouldn't particularly change scaling behavior. The enwik8 test resolves this by just using a normal byte-level tokenizer
- **Hyperparameters:** LEAP uses all the same hyperparameters as GPT2, all of which were chosen to be *advantageous to GPT2 and not LEAP* (likely better hyperparameters can be found for LEAP). We use a layer number ratio according to [Levine 2020](https://proceedings.neurips.cc/paper/2020/file/ff4dfdf5904e920ce52b48c1cef97829-Paper.pdf) that are best for Transformers like GPT2, and head size as close to 64 as possible. LEAP introduces two new hyperparameters, though they were set automatically based on preliminary testing and not tuned (they don't seem to strongly affect performance either)
- **Training:** Training was performed for only 1 epoch on sequence lengths of 1024 (by splitting and concatenating articles) with cosine learning rate schedule with a warmup ratio of .05. This is all in line with [Scaling laws for neural language models](https://arxiv.org/pdf/2001.08361.pdf). The batch sizes were very small of just 2 because of memory constraints

**Finer details:** [AdamW](https://arxiv.org/abs/1711.05101) optimizer with default configuration and learning rate of 5e-4 (after warmup and is cosine annealed). No dropout or label smoothing for regularization was used due to only training for 1 epoch as per the recommendation of [One Epoch Is All You Need](https://arxiv.org/abs/1906.06669)


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

Choromanski, Krzysztof, et al. "Rethinking attention with performers." _arXiv preprint arXiv:2009.14794_ (2020).

Komatsuzaki, A. (2019). One epoch is all you need. _arXiv preprint arXiv:1906.06669_.

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. _arXiv preprint arXiv:1711.05101_.

