# A Mathematical Framework for Transformer Circuit

The article is inspired by anthrophic's original [article](https://transformer-circuits.pub/2021/framework/index.html). It is an attempt to understand mathematically how transformers work. The article focuses on studying trasnformers in two layers or less which has only the attention block. This is further in contrast to the mordern transformer like GPT-3 which has 96 layers and alernates attention blocks with MLP blocks. 

Some prominent findings

- **Zero Layer transformers model bigram statistics**: The bigram table can be access directly from the weights
- **One layer attention-only transformers are ensemble of bigram and "skip-gram" (sequences which form "A..BC") models**: The bigram and skip trigrams can be directly access from weights, without running the models. These skip trigrams can be suprisingly expressive. This includes implementing a kind of very simple in-context learning
- **Two layer attention only transformers can implement more complext algorithms using composition of heads**: These composition algorithms can also be detected directly from the weights. Two layer models use attention head composition to create "induction heads", a very general in context learning algorithm
- One layer and two layer attention only transformers use very different algorithms to perform in context learning. Two layer attention heads use qualitatively more spohisticated inference time algorithms - in particular, a special type of attention head we call an induction head - to perform incontext learning, performing an important transition point that will be relevant for larger models. 
- **Attention heads can be understood as independent operations, each outputing a result which is added to the residual stream**: Attention heads are often described in an alternate "concatenate and multiply" formulation for computational efficiency. but this is mathematically equivalent. 
- **Attention only models can be written as a sum of interpretable end to end functions** mapping tokens into changes in logits. These function corresponds to "paths" through the model, and are linear if one freezes the attention patterns
- **Transformers have an enormous amount of linear structure**: A lot can be learned by simply breaking apart the sumps and multiplying together chains of matrices. 
- Attention heads can be understood as having two independent computations: QK ("query-key") circuit which computes the attention pattern, and OV ("output-value") circuit which computes how each tokens affect the output it attended to
- Key, query and value vectors can be thought as intermediate results in the computation of the low rank matrices $W_Q^{T}Q_K$ and $W_O^{T}W_V$. It is useful to describe transformer without reference to them
- Composition of attention heads greatly increases the expressivity of transformers. There are three different ways attention heads can compose, corresponding to keys, queries and values. Key and query composition are very different from value composition
- All components of a transformer (the token embedding, attention heads, MLP layers and unmbedding) communicate with each other by reading and writing to different subspaces of the residual stream. Rather than analyze the residual stream vectors, it can be helpful to decompose the residual streams into all these different composition channels, corresponding to paths through the model.

## Transformer Overview:

The section only considers autoregressive, decoder only transformer models like the one used in GPT-3. 
A transformer starts with a token embedding, followed by series of "residual blocks" and finally token unembedding. Each block consist of an attention layer, followed by an MLP layer. Both the attention and MLP layer each "read" their input from residual stream (by performing linear projection), and then "write" their result to the residual stream by adding the linear projection back in. Each attention layer has multiple heads, which operate in parallel

1. Token Embedding
```math
x_0 = W_Et
```

2. Each attention head h, is run and added to the residual stream
```math
x_{i+1} = x_i + \sum_{h \in H_i} h(x_i)
```

3.  An MLP Layer, m, is run and added to the residual sum
```math
x_{i+2} = x_i + m(x_i)
```

4. The final logits are produced by applying the unembedding
```math
T(t) = W_Ux_{-1}
```

The above high level architecture of a transformer is called a residual stream. It is simply a sum of the output of all previous layers and original embedding. We generally think of residual stream as a communication channel, since it doesn't do any processing itself and all layer communicate through it. The residual structure is deeply linear structure. Every layer performs an arbitrary linear transformation to read in information from the stream at the start, and performs another arbitrary linear transformation before adding to write its output back into the residual stream. This linear additive structure has a lot of important feature one being it does not have a privileged basis. One can rotate it by rotating all matrices yet the model behavior does not change.

### Virtual Weights

