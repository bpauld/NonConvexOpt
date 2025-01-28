# Nonconvex Optimization via Frank-Wolfe and Caratheodory.

This repository contains code for solving nonconvex separable optimization problems of the form
$$\begin{aligned}
\min_{x} & \sum_{i=1}^n f_i(x_i)\\
\text{subject to } & \sum_{i=1}^n A_i x_i \leq b\\
&x_i \in \text{dom}(f_i), \ i=1, \dots, n
\end{aligned}$$
where the functions $f_i$ and their domains need not be convex.
