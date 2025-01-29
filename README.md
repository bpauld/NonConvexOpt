# Nonconvex Optimization via Frank-Wolfe and Caratheodory.

This repository contains code for solving nonconvex separable optimization problems of the form

$$
\begin{align}
\sum_{i=1}^n f_i(x_i) \\
\text{subject to }  \sum_{i=1}^n A_i x_i \leq b \\
x_i \in \text{dom}(f_i), \quad i=1, \dots, n
\end{align}
$$

where the functions $f_i$ and their domains need not be convex.
