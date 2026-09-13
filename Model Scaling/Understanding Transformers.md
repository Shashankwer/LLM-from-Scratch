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

An especially useful consequence of the residual stream being linear is that one can think of implicit virtual weights directly connecting any pairs of layers (even when those are separated by many other layers), by multiplying out their interactions through the residual stream, These virtual weights are the product of the output weights of one layer with the input weights of the other layer (i.e. $W_I^2W_O^1$) and describe the extend to which a latent layer reads in the information written by a previous layer 

Since all the information is linear we can multiply through the residual streams. By using the different subspaces of the residual a layer can send different information to different layers or event not interact with other layers. 


### Subspaces and residual stream bandwidth

The residual stream is a high dimensional vector space. In small models, it may be hundreds of dimensionsl in large models it can go into tens of thousands. This means that layer can send different information to different layers by storing it in different subspaces. This is especially important in the case of attention heads, since every individual head operates on comparatively small subspaces (often 64 or 128 dimensions), and can very easily write to completely disjoint subspaces and not interact. 

Once added, information presists in a subspace unless another layer actively deletes it. From this prespective, dimensions of the residual stream become something like "memory" or "bandwidth". The original token embedding as well as the unembeddings, mostly interact with a relatively small fraction of the dimensions. This leaves most dimensions "free" for the other layers to store information in. 

Residual stream activation seem to be high in demand. There are generally far more "computational dimensoions" (such as neurons and attention head result dimensions) that the residual stream has dimension to move the information. Just a single MLP layer typically has 4 times more neuron than the residual stream. So at layer of 25 of 50 transformer, the residual stream has 100 times more neurons as it has dimensions before it, trying to communicate with 100 times as many neurons as it has dimensions after it, somehow communicating in superposition. We call the tensors like this "bottleneck activations" and expect them ti be unusually challenging to interpret

One typical explanation provided might be: some MLP neurons and attention heads may perform a kind of "memory management" role, clearing residual stream dimension set by other layers by reading in information and writing out negative version. 

### Attention heads are independent and additive

The output of the attention layer is descrived as stacking multiple result layers $r^{h_1}, r^{h_2}, ...,$ and then multiplying by the output matrix $W_O^H$. If we split the output layer into multiple blocks corresponding to each head then $[W_O^{h_1}, W_O^{h_2}, ...]$. Then its observed that 

$$
W_O^H \begin{bmatrix} r^{h_1} \\ r^{h_2} \\ ... \end{bmatrix} = \begin{bmatrix} W_O^{h_1} & W_O^{h_2} & ... \end{bmatrix}\begin{bmatrix} r^{h_1} \\ r^{h_2} \\ ... \end{bmatrix} = \sum_i{W_O^{h_i} r^{h_i}}
$$

Indicating running multiple heads independently and multiplying them by its own output matrix and adding them into the input stream. The concatenation is preferred because it produces a larger and more compute efficient matrix multiply.

### Attention Heads as Information Movement

The fundamental action of attention heads is moving information. They read information from the residual stream of one token, and write it to the residual stream of another token. The main observation to take away from this section is that which tokens to move information from is completely separable from what information is "read" to be moved and how it is "written" to the destination. 

To see this, its helpful to write the attention in a non standard way. Given an attention pattern, computing the output of an attention head is typically described in three steps

1. Compute the value vector for each token from the stream ($v_i = W_vx_i$)
2. Compute the "result vector" by linearly combining value vectors according to the attention pattern ($r_i = \sum_j{A_{i,j}v_j}$)
3. Finally, compute the output vector of the head for each token ($h(x)_i = W_Or_i$)

Each of these steps can be written as matrix multiply. Using tensor process one can define the process of applying attention as 


$$
h(x) = \frac{Id \otimes W_O}{\begin{array}{l}
\text{Project result} \\  
\text{vector out for} \\
\text{each token} \\
 (h(x_i) = W_o r_i)\end{array}} . \frac{A \otimes Id}{\begin{array}{l}\text{Mix value } \\ \text{vector across} \\ \text{tokens to} \\ \text{compute result} \\ \text{vectors} \\ (r_i = \sum_j{A_jv_j})\end{array}} . \frac{ Id \otimes W_v}{\begin{array}{l}\text{Compute value }\\ \text{vector for each} \\ \text{token} \\ (v_i = W_vx_i) \end{array}}
$$

This can be collapsed as 

$$
h(x) = \frac{ A \otimes W_oW_v}{\begin{array}{l}\text{A mixes tokens} \\ \text{while } W_o W_V \text{ acts on each vector} \\ \text{independently} \end{array}}. x
$$

Typically one compures the keys $k_i = W_Kx_i$, computes the query $q_i = W_Qx_i$ and then computes the attention pattern from the dot product of each key and query vector $A = \text{softmax}(q^Tk)$. This can be represented mathematically as 

$A = \text{softmax}(x^TW_Q^TW_kx)$

### Observation About Attention Heads

A major benefit of rewriting attention heads in this format is it helps to understand
- Attention head moves information from the residual stream of one token to another. A collory of this is that the residual vector space - which is often interpreted as "contextual word embedding" - will generate linear subspaces corresponding to information copied from other tokens and not directly about the present token
- An attention head is really applying two linear operations, A and $W_OW_V$, which operates on different dimensions and acts independently. A governs which token information is moved from where to where. $W_OW_V$ governs which information is read from source token and it is written to the destination token
- A is the only non linear part of this equation (Being computed using softmax). This means if we fix A, the attention pattern is fixed, and without it is half linear in a sense, since per token linear operation is constant
- $W_Q$ and $W_K$ always operate together. They are never independent. Similarly $W_O$ and $W_V$ also operate together. Although they are parameterised as a separate matrix $W_Q^TW_K$ and $W_OW_V$ they can always be thought as a low rank matrix. This means key, query and value are by products of computing low rank matrices. One can reparameterize both factors of the low rank matrices to create different vectors which can still function identically. Because $W_OW_V$ and $W_QW_K$ always operate together we like to define variables representing these combined matrices, $W_{OV} = W_OW_V$ and $W_{QK} = W_Q^TW_K$
- Products of attention heads behave much like attention heads themselves. By the distributive property, $(A^{h_2} \otimes W_{OV}^{h_2}).(A^{h_1} \otimes W^{h1}{OV}) = (A^{h_1}A^{h_2}) \otimes (W_{OV}^{h_2}W_{OV}^{h_1})$. The result of this product results into