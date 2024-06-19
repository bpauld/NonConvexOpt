# Nonconvex Optimization via Frank-Wolfe and Caratheodory. 

This repository contains code for solving nonconvex separable optimization problems of the form
```math
\begin{split}
\min_x \quad &\sum_{i=1}^n f_i(x_i) \\ \text{ subject to } \quad &Ax \leq b \\ &x_i \in \text{dom}(f_i), \quad i=1,\dots, n,
\end{split}
```
where the functions $f_i$ and their domains $\text{dom}(f_i)$ need not be convex.


Benjamin Dubois-Taine and Alexandre d'Aspremont. "Frank-Wolfe meets Shapley-Folkman: a systematic approach for solving nonconvex separable problems with linear constraints". In: arXiv preprint arXiv:2401.09961.

[[Paper]](https://arxiv.org/abs/2401.09961)

## Installation

Once you have downloaded the repository, we suggest creating a `conda` environment named `phase_unwrapping`.
```
conda env create -f environment.yml
conda activate phase_unwrapping
```
We also provide a `requirements.txt` file for users wishing to use another virtual environment manager.

Once this is done, you need to compile SNAPHU. This step is not necessary if you plan to use your own weights or uniform weights in your objective function. However SNAPHU generated weights have shown good practical performance, so we encourage using them. In the root directory, run the following.

## Reproduce experiments

To reproduce the experiments from the paper, run the `UnitCommitmentSquared/` directory:

```
$ python script.py
```

This will load simulated and real images from a region in Lebanon from the `data` folder. It will then unwrap the images using the IRLS algorithm, and write the output to a `results` folder.

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
