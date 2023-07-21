# Principle Component Analysis using a face dataset

## Use Eigenvalue Decomposition
1. Let $X$ be the dataset and $X \in \mathbb{R}^{n \times d}$ ($n$ is number of data points and $d$ is number of features)

2. Let $\mu$ be the data mean $$\mu = \frac{1}{n} \sum_{i=1}^{n}x^i$$ 

3. Let $X_c \in \mathbb{R}^{n \times d}$ be the centralized dataset. 

4. Let $C \in \mathbb{R}^{d \times d}$ be the covariance matrix $$C = cov(x) = \frac{X_c^T X_c}{n-1}$$

5. Using Enigenvalue Decomposition, $C = U \Lambda U^T$
	* $U, \Lambda, U^T \in \mathbb{R}^{d \times d}$
	* $\Lambda = diagonal(\lambda_1, \lambda_2 \cdots \lambda_d)$ such that $\lambda_i > \lambda_{i+1}$
	* $U = [\vec{U_1} |  \vec{U_2} | \cdots | \vec{U_d}]$ ($\vec{U_i}$ are eigenvectores with corresponding eigenvalue $\lambda_i$)

6. Construct matrix $A \in \mathbb{R}^{q \times d}$ like the following:

$$A=\begin{bmatrix}\vec{U_1}^T \\ \vec{U_2}^T \\ \vdots \\ \vec{U_q}^T\end{bmatrix}$$


### Transform
$$z^i = A(x^i - u)$$
### Reconstruct
$$\hat{x^i} = A^Tz^i + u$$

## Singular Value Decomposition
1. $X \in \mathbb{R}^{n \times d}$ and by using SVD, $X = USV^T$
	* $U \in \mathbb{R}^{n \times n}$
	* $S \in \mathbb{R}^{n \times d}$
	* $V^T \in \mathbb{R}^{d \times d}$
	* $U^TU = I$
	* $V^TV = VV^T = I$

2. $X^{T}X = VS^{T}U^TUSV^T =VS^TSV^T = VDV^T$ 
	* $D = S^TS$
	* $D$ is the diagonal matrix with square of singular values

3. $X^TXV = VDV^TV = VD$
	* $V$ contains eigenvectors of $X^TX$
	* $D$ contains eigenvalues of $X^TX$

4. Scipy svd
	```
	[U, S, V] = svd(c_x)
	```
	* $\frac{S^TS}{n-1}$ are eigenvalues of $C$
	* $V$ are eigenvectors of $C$
