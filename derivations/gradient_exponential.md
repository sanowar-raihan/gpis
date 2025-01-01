Our covariance function is
```math
k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right), \quad \text{where} \ r = ||x_i - x_j||
```
1. The first partial derivative with respect to $x_i$ is calculated as follows:
```math
\frac{\partial k}{\partial x_i} = \frac{\partial k}{\partial r} \frac{\partial r}{\partial x_i} \\ 
```
```math
\begin{align}
\frac{\partial k}{\partial r} &= \frac{\partial}{\partial r} \left(\sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right)\right) \\
 & = \sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right) \cdot \frac{\partial}{\partial r} \left(-\frac{r^2}{2\ell^2}\right) \\
 & = \sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right) \cdot \left(-\frac{2r}{2\ell^2}\right)\\
 & = \left(-\frac{r}{\ell^2}\right) \cdot \sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right)\\
 & = \left(-\frac{r}{\ell^2}\right) k(x_i, x_j)
\\
\frac{\partial r}{\partial x_i} &= \frac{x_i - x_j}{r}
\end{align}
```
  
&nbsp; &nbsp; &nbsp;  Now from the chain rule we get:
```math
\begin{align}
\frac{\partial k}{\partial x_i} &= \left(-\frac{r}{\ell^2}\right) k(x_i, x_j) \frac{x_i - x_j}{r}\\
&= -\frac{k(x_i, x_j)}{\ell^2} (x_i - x_j)
\end{align}
```


2. The first partial derivative with respect to $x_j$ is calculated as follows:
```math
\frac{\partial k}{\partial x_j} = \frac{\partial k}{\partial r} \frac{\partial r}{\partial x_j} \\ 
```
```math
\begin{align}
\frac{\partial k}{\partial r} &= \left(-\frac{r}{\ell^2}\right) k(x_i, x_j)
\\
\frac{\partial r}{\partial x_j} &= \frac{x_j - x_i}{r}\\
 &= -\frac{x_i - x_j}{r}
\end{align}
```
  
&nbsp; &nbsp; &nbsp;  Now from the chain rule we get:
```math
\begin{align}
\frac{\partial k}{\partial x_j} &= \left(-\frac{r}{\ell^2}\right) k(x_i, x_j) \cdot \left(-\frac{x_i - x_j}{r}\right)\\
&= \frac{k(x_i, x_j)}{\ell^2} (x_i - x_j)
\end{align}
```


3. To find the second partial derivative we need to differentiate $\frac{\partial k}{\partial x_i}$ with respect to $x_j$:   
```math
\begin{align}
\frac{\partial^2}{\partial x_i \partial x_j} k(x_i, x_j) &= \frac{\partial}{\partial x_j} \left( \frac{\partial k}{\partial x_i} \right) \\
 &= \frac{\partial}{\partial x_j} \left( -\frac{k(x_i, x_j)}{\ell^2} (x_i - x_j) \right) \\
 &= -\frac{1}{\ell^2} \left( \frac{\partial k(x_i, x_j)}{\partial x_j} (x_i - x_j) + k(x_i, x_j) \frac{\partial}{\partial x_j} (x_i - x_j) \right) \\
 &= -\frac{1}{\ell^2} \left( \frac{k(x_i, x_j)}{\ell^2} (x_i - x_j) \cdot (x_i - x_j) + k(x_i, x_j) \cdot (-I) \right)\\
 &= -\frac{k(x_i, x_j)}{\ell^4} (x_i - x_j) (x_i - x_j)^T + \frac{k(x_i, x_j)}{\ell^2} I
\end{align}
```
