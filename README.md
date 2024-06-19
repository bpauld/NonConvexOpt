# Nonconvex Optimization via Frank-Wolfe and Caratheodory. 

This repository contains code for solving nonconvex separable optimization problems of the form
```math
\begin{split}
\min_x \quad &\sum_{i=1}^n f_i(x_i) \\ \text{ subject to } \quad &Ax \leq b \\ &x_i \in \text{dom}(f_i), \quad i=1,\dots, n,
\end{split}
```
where the functions $f_i$ and their domains $\text{dom}(f_i)$ need not be convex.

The full theoretical background can be found in our paper:

Benjamin Dubois-Taine and Alexandre d'Aspremont. "Frank-Wolfe meets Shapley-Folkman: a systematic approach for solving nonconvex separable problems with linear constraints". In: arXiv preprint arXiv:2401.09961.

[[Paper]](https://arxiv.org/abs/2401.09961)

## 

## Reproduce experiments

To reproduce the experiments from the paper, run the following in the `UnitCommitmentSquared/` directory:

```
$ python script.py
```

## Citing our work

To cite our our work please use:
```
@article{dubois2024iteratively,
  title={Iteratively Reweighted Least Squares for Phase Unwrapping},
  author={Dubois-Taine, Benjamin and Akiki, Roland and d'Aspremont, Alexandre},
  journal={arXiv preprint arXiv:2401.09961},
  year={2024}
}
```
