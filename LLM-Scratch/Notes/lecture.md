# Language modeling from Scratch

- From Scratch: by building everything from ground up we can learn how things work
- Priortize high value-per-time concepts, dont loose the forest for trees
- More coverage of modern LM ingredients (mixture if experts, long context, agents)

## Underlying use of the course

- Problem: researchers are becoming disconnected from the underlying technology
- 2016: researchers implemented and build their 

Building small language models (<1B parameters) few issues noticed


| OPT Setups | Flops/Update | % FLops MHA | % FLops FFN | % FLops attn | %Flops logits |
| ---------- | -----------  | ----------- | ----------- | ------------ | ------------- |
| 760M       | 4.3E+15      | 35%         |  44%        | 14.8%        | 5.8%          |
| 1.3B       | 1.3E+16      | 32%         |  51%        | 12.7%        | 5.0%          |
| 2.7B       | 2.5E+16      | 29%         |  56%        | 11.2%        | 3.3%          |
| 6.7B       | 1.1E+17      | 24%         |  65%        | 8.1%         | 2.4%          |
| 13B        | 1.1E+17      | 22%         |  69%        | 6.9%         | 1.6%          |
| 30B        | 9.0E+17      | 20%         |  74%        | 5.3%         | 1.0%          |
| 66B        | 9.5E+17      | 18%         |  77%        | 4.3%         | 0.6%          |
| 175B       | 2.4E+18      | 17%         |  80%        | 3.3%         | 0.3%          |

Small scale models are not suitable for learning complex tasks while large models are able to learn (after a particular threshold)


## What we can learn

- *Mechanics*: how things work (what a transformerr is, how model parallelism work)
- *Mindset*: squeezing the most out of hardware taking scaling seriously
- *Intuitions*: which data and modeling decision yeild good accuracy

Some design decisions are simply not justifiable and just come from experimentation. E.g. Noam Shazeer paper that introduce Swiglu,


It is not true that the scale is all that matters, algorithms do matter. Right algorithm and scale that matter. 

$\text{accuracy} = \text{efficiency) \times \text{resources}$

In fact, efficiency is way more important at larger scales (we cant afford to be wasteful)

44x algorithmic efficiency on ImageNet between 2012 and 2019

Framing: what is the best model one can build on certain compute and data budget? 
In other words, maximize efficiency


## Current Landscape: 

1. Pre Neural (before 2010):

- Language model to measure the entrophy of English
- N-gram language models (used in the machine translation and speech recognition systems)

## Neural ingredients

- Long Short Term Memory (LSTM)
- Neural language model
- Sequence to Sequence modeling 
- Adam optimize
- Transformer Architecture
- Mixture of experts
- Model parallelism


## Early foundation models
- ELMO: pretraining LSTMs; finetuning improves downstream tasks
- BERT: pretraining with Transformers
- Google's T5 cast everything as text to text


## Embracing the scaling
- OpenAI's GPT-2 (1.5B): fluent text, first sign of zero-shot
- Scaling laws: provided hope/predictability
- GPT3 (175B): in context learning
- Google's PaLM (540B): massive scaling but undertrained
- DeepMind's (70B): compute scaling laws 

## Open Models: 

Early Attempts (attempts to replicate GPT-3)
- EleutherAI's open datasets (The pile) and the models (GPT-J)
- Meta's OPT(175B): GPT-3 replication, lots of hardware issues
- Hugging Face/BigScience's Bloom (176B): focused on data souring

Credible open weight models
- Meta's Llama models
- Mistral models
- DeepSeek's Models
- Alibaba's Qwen models
- Moonshot's Kimi
- Z.ai's GLM models
- Minimax models
- Xiami's MIMO models

Open-source models
- AI2;s Olmo models
- NVIDIA's Nemotrom models
- Marin's models (open development)

Openness is important for trust and innovation.

## What is a language Model?

- Something you finetune: (BERT)
- Something you prompt: GPT-3
- Something you talk to: ChatGPT-3
- Something that acts autonomously (agents)

The fundamentals are the same (attention, kernels, optimization). The specs are different (longer context, inference efficiency matters more)

## Structure

1. Basics: 

Train a basic language model. Tokenize, model architecting, training

Tokens are the atoms that the model operate on. Formally a tokenizer converts between raw bytes into a sequence of integers. Popular tokenizer: Byte pair encoding. Intuition break input into frequently occuring chunks. 

Efficiency lens
- Reduce context length 
- Adaptive computation (more modeling capacity on interesting parts of input)

Dream is to use a tokenizer free models which operates directly on bytes. 

Refinements on the current architecture
1. Activation functions: ReLU -> SwiGLU
2. Positional Encoding: sinusoidal -> RoPE
3. Normalization: LayerNorm, RMSNorm, QKNorm, pre-norm vs post norm
4. Attention: full, sparse/local attention, group query attention (CGQ), multi headed attention
5. Recurrence/state-space models/linear attention: Mamba, Gated Delta
6. MLP: dense, mixture of experts
7. Shape (hidden dimension, depthm number of heads, number of experts)


Refinements to training: 

How do we set the parameters of the model??
1. Loss function (e.g. multi-token prediction)
2. Optimizer (e.g. AdamW, SOAP, MuON)
3. Initialization scale (e.g. Xavier init, muP)
4. Learning rate schedule (e.g. cosinem WSD)
5. Regularization (e.g. dropout, weight decay)
6. Batch Size (e.g. critical batch size)
7. MoE specific: load balancing (e.g. aux-free)



