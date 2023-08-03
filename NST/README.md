# Neural Style Transfer

## Content Cost Function

$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2  $$

* $C$: content image
* $G$: Generated image
* $a^{(C)}$: 3D volumes corresponding to a hidden layer's($l$) activations when $C$ was the input 
* $a^{(G)}$: 3D volumes corresponding to a hidden layer's($l$) activations when $G$ was the input 
* For the choose of $l$, it should not be too shallow or deep so that it can capture both low-level and high-level information

## Style Cost Function

### Style Cost Function for a layer
$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum \sum (G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2 $$

* $G_{gram}^{(S)}$ Gram matrix of the "style" image.
* $G_{gram}^{(G)}$ Gram matrix of the "generated" image.
* $G_{gram}(A) = AA^T$

### Whole Style Cost Function 
$$J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)$$

* $\lambda^{[l]}$ is the weights given for a particular layer


## Total Cost function
$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$
