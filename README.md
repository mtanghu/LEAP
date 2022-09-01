# Linear Explainable Attention in Parallel (LEAP)

THIS PROJECT IS CURRENTLY UNDER CONSTRUCTION, PLEASE CHECK BACK IN 3 DAYS!!!

This project implements a novel linear attention mechanism based on cumulative sums that are "focus" weighted.

Positive aspects of this approach (some novel):

1. Most Linear Attention mechanisms cannot be used for parallel decoder language modeling or don't work that well (see Eleuther comments on ["Have you considered more efficient architectures or methods?](https://www.eleuther.ai/faq/))
2. The models considered in this project run faster than a standard Transformer when run with the same # of layers and layer sizes even on small sequence lengths (the math allows for *strongly parallelize-able* operations which is not always the case with linear attention)
3. Already integrated with [HuggingFace🤗 Transformers](https://huggingface.co/)
4. **Linear in time local attention**, this concept has not been seen before in the literature as local attention typically has to scale in time complexity with the size of the local window, which we use some math trickery to get around this (and still be parallelize-able). This gets around the issue that longer sequences will typically need bigger local attention windows, as well as the issue that local + global attention (previously explored in [Longformer](https://arxiv.org/pdf/2004.05150.pdf) and [BigBird](https://arxiv.org/pdf/2004.05150.pdf)) may not have enough representational complexity at scale (not enough mid-range sequence modeling)
5. **Built-in Explainability**, while explainability is not supported yet in this project, as we'll see later, each token will be assigned an "importance weight" (which is softmax-ed) which can be used to explain what tokens the model is paying attention to, and which tokens are ignored similar to the explainability offered by the original [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) paper, though even more explainable as attention weights aren't pairwise anymore
6. **O(1) Path Length/Flexibility**, A great strength of full attention Transformers is the flexibility provided by the $O(1)$ path length. An example where many linear attention mechanisms would likely fail (ie. if they only use local/convolutional attention or time-decaying factors or a recurrent vector that will get overloaded with information over time) would be when there is "*task metadata*" at the beginning of the sequence. Example: "Read the following story paying special attention to how Alice treats Bob as you will write an essay on this after: \<very long story here\>". This task information may not make it all the way through the story and writing the essay with the previously mentioned approaches, but with this project's approach, tokens from the beginning of the sequence can directly transfer information to tokens at the end of the sequence with a $O(1)$ path length (like full-attention) through global LEAP
7. **O(1) Inference**, the math of LEAP can be represented as an RNN (while still maintaining the $O(1)$ path length). Thus, you only need the previous token's embeddings (i.e. $O(1)$ space) to calculate the next token (as per being an RNN) which only takes $O(1)$ computations with no matrix-matrix operations (all with respect to sequence length holding model size/dimension constant). This was originally shown in [Transformers are RNNs](https://arxiv.org/pdf/2006.16236.pdf) by Katharpoulos et al. (2020) to increase inference time performance by thousands of times and could potentially *allow large language models to run on edge devices like mobile phones or consumer laptops!*


This README will summarize LEAP and annotate a number of its details, show a unique linearization process/math (which allows for RNN formulation), show how this approach can be used for linear local attention, as well as preliminary results which show that LEAP is potentially comparable to full attention, and there is plenty of room for development!

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
    rescale_value = 8 # what to rescale the focus values with, set lower if you have NaN loss
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
 If you want to contribute, (optionally) make a github issue and send in a pull request! A number possibilities for development and usage come to mind:

1. Additive Attention may work better on other & more specific NLP tasks (conversation, reasoning, genomics, speech/audio, vision, text-to-image, or even specific language domains) as per the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) which especially applies to unique datasets. Other Linear Attention papers seem to use unique datasets too. This is especially the case with smaller datasets as Additive Attention does have stronger inductive bias
2. **Theory** to show that local, mid-range, and global Additive Attention can replicate full Attention/is Turing Complete? -- This should be unequivocally True since it was already found to be true for [BigBird](https://arxiv.org/pdf/2004.05150.pdf) with local + random + global attention, where this project has local + mid-range + global attention which should be stronger (The proof will be tricky though because of the differences in mechanism). If this were true, we wouldn't necessarily need to worry about doing a one-to-one comparision with standard Transformers as much since the theory would support that this system is just as capable of modeling sequence information
3. **More ablation and feature studies** could be performed to improve performance. Currently this project is working on direct comparison (as one-to-one as possible) with GPT2, so more recent transformer advancements have not been implemented (Rotary or ALiBi embeddings, key-query embeddings, parameter sharing, token mixing, new initialization schemes, etc.). It is important to continue this direct comparison research as a matter of making sure the proposed attention mechanism works comparably while also implementing the latest techniques seperately especially if unique techniques work particularly well or are in tandem with specifically Additive Attention
4. **Infinite Context** as mentioned in the previous point, recent work into positional embeddings that can extrapolate to longer context sizes is making a lot of progress. Both local and global Additive Attention should extrapolate to longer sequences because it is simply a "learned softmax-ed weight sum of tokens" of differing lengths
5. **Reinforcement Learning** as noted by [OpenAI's Requests for Research](https://openai.com/blog/requests-for-research-2/) a good linear attention system (that can be represented as an RNN) is very attractive for RL rollouts. This project contends that specifically the local, mid-range, and global attention is much more *biologically plausible* as humans/animals are more likely to keep track or local, mid-range, and global sequence information (and their interactions) rather than considering all pairwise interactions in a sequence
7. **Explainability** A note which was unexplored in the original fastformer paper (because they didn't quite use the same formulation as this project) is how this system has *built-in explainability*, instead of pairwise interactions with full attention, each token is directly assigned an "importance" scalar (see equations 1 and 2, $\alpha_{i}$ is the importance weight) which can directly be tracked for explaining what tokens were deemed as important for future predictions and what tokens weren't. This should be explored with measurements and experiments!
8. ***RNN Formulation*** as stated in the positives section, and the math will show, we can represent Additive Attention as an RNN for fast $O(1)$ inference. Current work is primarily focusing on getting parallel large scale training done, but this is still necessary for the future of bringing this project and large language models to the public


## Linear RNN or Parallel Cumulative Sum?

It may be confusing that there are two sets of equations for Additive Attention in both of the previous sections, however there is an important purpose.

- **During training** when the entire sequence is presented, the key point of Transformers is that even though they have <div></div>$O(N^2)$ complexity, this comes from a matrix multiplication which is highly parallelizable on GPUs (and TPUs too of course). Lucky for us, a **cumulative sum is also parallelizable** as explained by [this wikipedia on prefix sum aka cumulative sum](https://en.wikipedia.org/wiki/Prefix_sum#Algorithm_1:_Shorter_span,_more_parallel) which is implemented with CUDA [as seen here](https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a16d7bc049ba8985dd89b10b3bcf0a8a3) which is directly called by Pytorch (used in this project) for cumulative sums (shown by [these CUDA profiles](https://github.com/pytorch/pytorch/issues/75240)). Thus the summation equations (5 and 10) make more sense as they can be directly translated to cumulative sums.

- **During inference** each token is generated one at a time. Thus, at inference we can just focus on the linear (and non-parallel) RNN formulation which just requires us to keep track of the most recent <div></div>$\boldsymbol{s_i}, \boldsymbol{s_i'}, z_i, z_i'$ (only 2 vectors and 2 scalars) to generate the next token as well as generate the next Additive Attention vectors (i.e. $\boldsymbol{s_{i+1}}, \boldsymbol{s_{i+1}'}, z_{i+1}, z_{i+1}'$). This only needs $O(1)$ complexity in terms of sequence length to generate new tokens (holding the model size/dimension constant).

<div></div>Slight tangent: In RNN mode for local attention, to calculate $\boldsymbol{s_{i+1}'}$ and $z_{i+1}'$ you will need to keep a buffer of the previous *tokens* and not their embeddings, as those can be recomputed "on the fly". This technically means you need $O(N)$ space but NOT $O(N*d)$ space (where $d$ is the hidden size of the model). In general, this going to be a trivial matter since almost every sequence task will store all previous sequence tokens somewhere (like for text generation, all previously generated text will be stored). So of we only consider the amount of RAM/VRAM needed, it is still $O(1)$, as the buffer of previous tokens could be stored on disk and can be retrieved efficently with cached IO calls (without any special IO management).

## Model Structure

Because this is a causal language model the code is structured like one and implements the following to be fair comparison against GPT2 [paper for reference by Radford et al. (2019)](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf):

- Pre-norming with a layernorm before projecting to logits like GPT2
- Dropout of .1 on embeddings, feedforward, and attention layers like GPT2
- Learned positional embeddings ([GPT1 paper by Radford et al. (2018)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) which carries over to GPT2, though [Rotary embeddings](https://arxiv.org/abs/2104.09864v2) we considered, but decided against because it would unfairly give an advantage to the model when compared against normal transformers/gpt2 which uses learned absolute positional embeddings
- Weight tying ([Press & Wolf 2017](https://arxiv.org/abs/1608.05859v3)) also used by GPT2
- Label smoothing ([Muller, Kornblith & Hinton 2019](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html), [Viswani et al. 2017](https://arxiv.org/abs/1706.03762) is forgone because huggingface seems to oddly apply label smoothing during validation (so the loss that comes out when exponentiated would not be perplexity)
- Attention masking of pad tokens ([Attnetion is All you Need by Viswani et al. (2017)](https://arxiv.org/abs/1706.03762)) which is carried over to GPT2

## Preliminary Results

![alt text](https://github.com/mtanghu/Additive-Attention-Is-All-You-Need/blob/main/preliminary_results.png?raw=True)

Plotted is the validation perplexity of the LEAP Transformer (blue) and GPT2 (orange) when trained on Wikitext-2 with a T5 tokenizer. The final test set perplexity of the LEAP transformer was 45.6 and the final test set perplexity of GPT2 was 59.7.

As we can see on this small scale experiment, the LEAP Transformer converges faster with less perplexity compared to even GPT2 (of the same size). Even though these results are preliminary, the long sequence length of 2048 should already be enough to test the abilities of this model as being better than an RNN like LSTMs as found by this [Scaling Laws paper](https://arxiv.org/abs/2001.08361) (Figure 7 finds that LSTM scaling bends at around 1M parameters, and at context lengths of >1000, the LSTM should be unable to compete). Also because of the linear local attention, it may be more reasonable to believe that this model can scale up (as the combinations of local and global attentions should be able to model complex sequence information from short-range to long-range). Furthermore, this model beats both [Mogrifier LSTM](https://arxiv.org/abs/1909.01792v2) and [AWD LSTM](https://arxiv.org/abs/1708.02182v1) (when not using dynamic eval) on even though those models use >30M parameters (see the [leaderboard on paperswithcode](https://paperswithcode.com/sota/language-modelling-on-wikitext-2))

**Speed:** The LEAP Tranformer training run took ~24 minutes while GPT2 took ~48 minutes (2x) which should become more pronounced at context lengths greater than 2048 and larger model sizes. Also the convergence was faster.

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
