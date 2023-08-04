# **Generalized Teacher Forcing for Learning Chaotic Dynamics [ICML 2023 Oral]**
![](misc/GraphSum_GTF_icml2023.png "Graphical Summary.")

## Setup and usage
The entire project is written in [Julia](https://julialang.org/) using the [Flux](https://fluxml.ai/Flux.jl/stable/) deep learning stack.
### Installation
Install the package in a new Julia environment:
```julia
julia> ]
(@v1.9) pkg> activate .
(GTF) pkg> instantiate
```
We recommend using the latest version of [Julia (>v1.9)](https://julialang.org/downloads/).

### Ready-to-use experiment scripts
The folder `paper_experiments` holds subfolders for each dataset evaluated in the paper, in which you can find ready-to-use Julia scripts, which start and evaluate specific experiments of the paper (such as reproducing Table 1 results). For example, to run experiments using shPLRNN+GTF on the EEG dataset as in Table 1, run 
```bash
julia -t 1 --project paper_experiments/EEG/Table1/shPLRNN_GTF.jl -p 20 -r 20
```
which will start 20 processes with 1 thread each to train 20 models using the settings provided in `paper_experiments/ECG/Table1/shPLRNN_GTF.jl`. All datasets are stored in the folder `ICML2023_datasets`. The list of all settings can be found in the [default settings](settings/defaults.json) file. Any argument not overwritten in the experiment scripts will fall back to the default value found in that setting file.

### General code documentation
*More details will follow...*

## Citation
If you find the repository and/or paper helpful for your own research, please cite [our work](https://proceedings.mlr.press/v202/hess23a.html).
```
@InProceedings{pmlr-v202-hess23a,
  title = 	 {Generalized Teacher Forcing for Learning Chaotic Dynamics},
  author =       {Hess, Florian and Monfared, Zahra and Brenner, Manuel and Durstewitz, Daniel},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {13017--13049},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/hess23a/hess23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/hess23a.html},
  abstract = 	 {Chaotic dynamical systems (DS) are ubiquitous in nature and society. Often we are interested in reconstructing such systems from observed time series for prediction or mechanistic insight, where by reconstruction we mean learning geometrical and invariant temporal properties of the system in question (like attractors). However, training reconstruction algorithms like recurrent neural networks (RNNs) on such systems by gradient-descent based techniques faces severe challenges. This is mainly due to exploding gradients caused by the exponential divergence of trajectories in chaotic systems. Moreover, for (scientific) interpretability we wish to have as low dimensional reconstructions as possible, preferably in a model which is mathematically tractable. Here we report that a surprisingly simple modification of teacher forcing leads to provably strictly all-time bounded gradients in training on chaotic systems, and, when paired with a simple architectural rearrangement of a tractable RNN design, piecewise-linear RNNs (PLRNNs), allows for faithful reconstruction in spaces of at most the dimensionality of the observed system. We show on several DS that with these amendments we can reconstruct DS better than current SOTA algorithms, in much lower dimensions. Performance differences were particularly compelling on real world data with which most other methods severely struggled. This work thus led to a simple yet powerful DS reconstruction algorithm which is highly interpretable at the same time.}
}
```

## Funding
This work was funded by the German Research Foundation (DFG) within Germany’s Excellence Strategy EXC 2181/1 – 390900948 (STRUCTURES), by DFG grants Du354/10-1 \& Du354/15-1 to DD, and by the European Union Horizon-2020 consortium SC1-DTH-13-2020 (IMMERSE).
![](misc/logos.png "Funding.")
