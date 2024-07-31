## Incorporating gradient observations into the Gaussian Process
Consider a simple Gaussian Process **A**= $\\{A(x)\\}_{x \in R^2}$ with zero mean and covariance function $k : R^2 \times R^2 \rightarrow R$ with three different training observations: $\\{(x_1, f(x_1)), (x_2, f(x_2)), (x_3, f(x_3))\\}$, their derivatives at those points $\\{\frac{\partial f(x_1)}{\partial x_1} , \frac{\partial f(x_2)}{\partial x_2}, \frac{\partial f(x_3)}{\partial x_3}\\}$ and one test point $x_4$.

To incorporate the gradient observations in to the Gram matrix $K_3$, we will construct it in four blocks. These four blocks are explained below:
1. **Covariance between function values**

    $k(x_i,x_j)$ represents the covariance between function values at $x_i$ and $x_j$. For our exapmle, $K_{values}$ will be a 3x3 matrix:
```math
K_{values} = \begin{bmatrix}
k(x_1, x_1) & k(x_1, x_2) & k(x_1, x_3) \\
k(x_2, x_1) & k(x_2, x_2) & k(x_2, x_3) \\
k(x_3, x_1) & k(x_3, x_2) & k(x_3, x_3)
\end{bmatrix}
```

2. **Covariance between function values and first order derivatives.**
   
   $\frac{\partial}{\partial x_j} k(x_i, x_j)$ represents the covariance between the function value at ${x_i}$ and derivative at $x_j$.
```math
k(f(x_i), \frac{\partial f(x_j)}{\partial x_j}) = \frac{\partial}{\partial x_j} k(x_i, x_j)
```
In our 2D example, each $\frac{\partial}{\partial x_j} k(x_i, x_j)$ is a two dimensional vector. That means $K_{values-derivatives}$ will be a 3x6 matrix:    
```math
K_{values-derivatives} = \begin{bmatrix}
(\frac{\partial}{\partial x_1} k(x_1, x_1))^T & (\frac{\partial}{\partial x_2} k(x_1, x_2))^T & (\frac{\partial}{\partial x_3} k(x_1, x_3))^T\\
(\frac{\partial}{\partial x_1} k(x_2, x_1))^T & (\frac{\partial}{\partial x_2} k(x_2, x_2))^T & (\frac{\partial}{\partial x_3} k(x_2, x_3))^T \\
(\frac{\partial}{\partial x_1} k(x_3, x_1))^T & (\frac{\partial}{\partial x_2} k(x_3, x_2))^T & (\frac{\partial}{\partial x_3} k(x_3, x_3))^T
\end{bmatrix}
```

3. **Covariances between first order derivatives and function values.**

    $\frac{\partial}{\partial x_i} k(x_i, x_j)$ represents the covariance between the derivative at $x_i$ and function value at $x_j$.
```math
k(\frac{\partial f(x_i)}{\partial x_i}, f(x_j)) = \frac{\partial}{\partial x_i} k(x_i, x_j)
```
In our 2D example, each $\frac{\partial}{\partial x_i} k(x_i, x_j)$ is a two dimensional vector. That means $K_{derivatives-values}$ will be a 6x3 matrix:
```math
K_{derivatives-values} = \begin{bmatrix}
\frac{\partial}{\partial x_1} k(x_1, x_1) & \frac{\partial}{\partial x_1} k(x_1, x_2) & \frac{\partial}{\partial x_1} k(x_1, x_3)\\
\frac{\partial}{\partial x_2} k(x_2, x_1) & \frac{\partial}{\partial x_2} k(x_2, x_2) & \frac{\partial}{\partial x_2} k(x_2, x_3) \\
\frac{\partial}{\partial x_3} k(x_3, x_1) & \frac{\partial}{\partial x_3} k(x_3, x_2) & \frac{\partial}{\partial x_3} k(x_3, x_3)
\end{bmatrix}
```

4. **Covariance between first-order derivatives.**

    $\frac{\partial^2}{\partial x_i \partial x_j} k(x_i, x_j)$ represents the covariance between derivative at $x_i$ and derivative at $x_j$.
```math
k(\frac{\partial f(x_i)}{\partial x_i}, \frac{\partial f(x_j)}{\partial x_j}) = \frac{\partial^2}{\partial x_i \partial x_j} k(x_i, x_j)
```
In our 2D example, each second order derivative $\frac{\partial^2}{\partial x_i \partial x_j} k(x_i, x_j)$ is a 2x2 matrix. That means $K_{derivatives}$ will be a 6x6 matrix:
```math
K_{derivatives} = \begin{bmatrix}
\frac{\partial^2}{\partial x_1^2} k(x_1, x_1) & \frac{\partial^2}{\partial x_1 \partial x_2} k(x_1, x_2) & \frac{\partial^2}{\partial x_1 \partial x_3} k(x_1, x_3)\\
\frac{\partial^2}{\partial x_1 \partial x_2} k(x_2, x_1) & \frac{\partial^2}{\partial x_2^2} k(x_2, x_2) & \frac{\partial^2}{\partial x_2 \partial x_3} k(x_2, x_3) \\
\frac{\partial^2}{\partial x_3 \partial x_1} k(x_3, x_1) & \frac{\partial^2}{\partial x_3 \partial x_2} k(x_3, x_2) & \frac{\partial^2}{\partial x_3^2} k(x_3, x_3)
\end{bmatrix}
```

---
  The final $K_3$ will be constructed blockwise as follows. It will be a 9x9 matrix for our example:
```math
K_3 = \begin{bmatrix}
K_{values} & K_{values-derivatives} \\
K_{derivatives-values} & K_{derivatives}
\end{bmatrix}
```

---
To construct the observation vector $y$, we concatenate all the function values and their derivatives at those 3 training points. This will be a 9 dimensional vector:
```math
y = \begin{bmatrix}
f(x_1) \\
f(x_2) \\
f(x_3) \\
\frac{\partial f(x_1)}{\partial x_1}  \\
\frac{\partial f(x_2)}{\partial x_2} \\
\frac{\partial f(x_3)}{\partial x_3}
\end{bmatrix}
```

---
To make a prediction at a new point $x_4$, we need to create the $k_2$ matrix. This matrix holds the covariances between $x_4$ and all the training data points and their derivatives. It can be constructed blockwise, just as the $K_3$ matrix.
```math
k_2 = \begin{bmatrix}
k(x_1, x_4) & (\frac{\partial}{\partial x_4} k(x_1, x_4))^T \\
k(x_2, x_4) & (\frac{\partial}{\partial x_4} k(x_2, x_4))^T \\
k(x_3, x_4) & (\frac{\partial}{\partial x_4} k(x_3, x_4))^T \\
\frac{\partial}{\partial x_1} k(x_1, x_4) & \frac{\partial^2}{\partial x_1 \partial x_4} k(x_1, x_4) \\
\frac{\partial}{\partial x_2} k(x_2, x_4) & \frac{\partial^2}{\partial x_2 \partial x_4} k(x_2, x_4) \\
\frac{\partial}{\partial x_3} k(x_3, x_4) & \frac{\partial^2}{\partial x_3 \partial x_4} k(x_3, x_4)
\end{bmatrix}
```
For our 2D example, each first derivative is a 2d vector, and each second derivative is a 2x2 matrix. That means our $k_2$ matrix is going to be a 9x3 matrix.

---
The predictive distribution of the Gaussian process is defined by the following mean and variance:  
* $\mu = k_2^TK_3^{-1}y$  
  Here, $\mu$ is a three-dimensional vector. Its first component represents the mean value of the function at $x_4$, while the remaining components represent the mean values of the function's derivatives at $x_4$.
* $\Sigma = k_1 - k_2^TK_3^{-1}k_2$  
  The $k_1$ matrix is defined as follows:
```math
k_1 = \begin{bmatrix}
k(x_4, x_4) & (\frac{\partial}{\partial x_4} k(x_4, x_4))^T \\
\frac{\partial}{\partial x_4} k(x_1, x_4) & \frac{\partial^2}{\partial x_4^2} k(x_4, x_4)
\end{bmatrix}
```
&nbsp; &nbsp; &nbsp; Here, the covariance matrix Σ is a 3x3 matrix. Its top-left entry represents the variance of the function value at x4.


