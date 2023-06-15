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
If you find the repository and/or paper helpful for your own research, please cite [our work](https://arxiv.org/abs/2306.04406).
```
@article{hess2023generalized,
  title={Generalized Teacher Forcing for Learning Chaotic Dynamics},
  author={Hess, Florian and Monfared, Zahra and Brenner, Manuel and Durstewitz, Daniel},
  journal={arXiv preprint arXiv:2306.04406},
  year={2023}
}
```
*Link and reference will be updated once published in PMLR*.

## Funding
This work was funded by the German Research Foundation (DFG) within Germany’s Excellence Strategy EXC 2181/1 – 390900948 (STRUCTURES), by DFG grants Du354/10-1 \& Du354/15-1 to DD, and by the European Union Horizon-2020 consortium SC1-DTH-13-2020 (IMMERSE).
![](misc/logos.png "Funding.")
