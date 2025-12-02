# Nonconvex Optimization via Frank-Wolfe and Caratheodory.

This repository contains code for solving nonconvex separable optimization problems of the form

$$
\begin{align}
\text{minimize } \ &\sum_{i=1}^n f_i(x_i) \\
\text{subject to } \ &\sum_{i=1}^n A_i x_i \leq b \\
&x_i \in \text{dom}(f_i), \quad i=1, \dots, n
\end{align}
$$

where the functions $f_i$ and their domains need not be convex.

The full theoretical background can be found in our paper:

Benjamin Dubois-Taine and Alexandre d'Aspremont. "Frank-Wolfe meets Shapley-Folkman: a systematic approach for solving nonconvex separable problems with linear constraints". In: Mathematical Programming.

## Use the code

You should define a non convex separable problem by creating a child of the `NonConvexProblem` class defined in `code/non_convex_problem.py` and implementing the required functions. You can use the file `UnitCommitmentSquared/unit_commitment.py` as template.

You can then run the first conditional gradient method defined in `code/frank_wolfe_1.py`, followed by the second Caratheodory step, define in `exact_caratheodory.py` or `approximate_caratheodory.py`. You can use the file `UnitCommitmentSquared/script.py` as template.

## Reproduce experiments

To reproduce the experiments on the Unit Commitment problem, run one of the following in `UnitCommitmentSquared` folder:

```
$ python script.py
```
or 
```
$ python script_runtime.py
```

To reproduce the experiments on the charging of electric vehicles problem, run the following in `PEVs` folder:
```
$ python script.py
```

You can use the jupyter notebooks to reproduce the plots.

## Citing our work

To cite our work please use:

```
@article{dubois2025frank,
  title={Frank-Wolfe meets Shapley-Folkman: a systematic approach for solving nonconvex separable problems with linear constraints},
  author={Dubois-Taine, Benjamin and d’Aspremont, Alexandre},
  journal={Mathematical Programming},
  pages={1--51},
  year={2025},
  publisher={Springer}
}
```
