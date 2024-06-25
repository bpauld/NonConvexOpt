# Nonconvex Optimization via Frank-Wolfe and Caratheodory. 

This repository contains code for solving nonconvex separable optimization problems of the form
```math
\begin{split}
\min_x \quad &\sum_{i=1}^n f_i(x_i) \\ \text{ subject to } \quad &Ax \leq b \\ &x_i \in \text{dom}(f_i), \quad i=1,\dots, n,
\end{split}
```
where the functions $f_i$ and their domains $\text{dom}(f_i)$ need not be convex.

The full theoretical background can be found in our paper:

Benjamin Dubois-Taine and Alexandre d'Aspremont. "Frank-Wolfe meets Shapley-Folkman: a systematic approach for solving nonconvex separable problems with linear constraints". In: arXiv preprint.


## Use the code

You should define a non convex separable problem by creating a child of the `NonConvexProblem` class defined in `code/non_convex_problem.py` and implementing the required functions.
You can use the file `UnitCommitmentSquared/unit_commitment.py` as template.

You can then run the first conditional gradient stage defined in `code/frank_wolfe_1.py`, followed by the Caratheodory step, defined in `sparsify_constructive.py` (exact Caratheodory) or in `sparsify_frank_wolfe.py` (approximate Caratheodory).
You can use the file `UnitCommitmentSquared/script.py` as template.

## Reproduce experiments

To reproduce the experiments on the Unit Commitment problem from the paper, run the following in the `UnitCommitmentSquared/` directory:

```
$ python script.py
```

## Citing our work

To cite our our work please use:
```

```
