# SINDy vs Hard Nonlinearities and Hidden Dynamics: a Benchmarking Study

This is the code repository for the article *"SINDy vs Hard Nonlinearities and Hidden Dynamics: a Benchmarking Study"* (Aurelio Raffa Ugolini[^1], Valentina Breschi[^2], Andrea Manzoni[^3], Mara Tanelli[^1]), available on [arXiv](https://arxiv.org/abs/2403.00578)

## Introduction

In this work we analyze the effectiveness of the Sparse Identification of Nonlinear Dynamics (SINDy) technique on three benchmark datasets for nonlinear identification, to provide a better understanding of its suitability when tackling real dynamical systems. While SINDy can be an appealing strategy for pursuing physics-based learning, our analysis highlights difficulties in dealing with unobserved states and non-smooth dynamics. Due to the ubiquity of these features in real systems in general, and control applications in particular, we complement our analysis with hands-on approaches to tackle these issues in order to exploit SINDy also in these challenging contexts.

## Installation

Once you download or clone the repository, you can configure the environment via [`Pipenv`](https://pipenv.pypa.io/en/latest/) through the provided `Pipfile` and `Pipfile.lock`.

> **Note:** the lock file has been generated on a macOS machine, so you might need to delete it if you are running on a different OS. `Pipenv` will take care of the generation of a new `Pipfile.lock` and proceed with the installation.

## Usage

Each experiment detailed in the paper is implemented in a self-contained script under `src/experiments`. To execute the experiments, navigate to the `src/experiments` folder and run the script via 

```shell
python <EXPERIMENT_NAME>.py
```

where `<EXPERIMENT_NAME>` is replaced by:
- `pick_and_place` for the Pick and Place Machine experiment;
- `bouc_wen` for the [Bouc-Wen hysteresis model](https://www.nonlinearbenchmark.org/benchmarks/bouc-wen) experiment;
- `cascaded_tanks` for the [Cascaded Tanks dataset](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks).

Notice that the data have already been provided as part of the repository (under the `data` directory), except for the `pick_and_place` whose data is not public.

## Bug reporting, inquiries, and more

Please feel free to contact the corresponding author, [Aurelio Raffa](mailto:aurelio.raffa@polimi.it) for issues with the code or other inquiries.


[^1]: Dip. di Elettronica, Informazione e Bioingegneria, Politecnico di Milano, Via G. Ponzio 34/5 - 20133 Milano, Italy.
[^2]: Dept. of Electrical Engineering, Eindhoven University of Technology, 5600 MB Eindhoven, The Netherlands.
[^3]: Dip. di Matematica, Politecnico di Milano, P.zza Leonardo da Vinci 32- 20133 Milano, Italy.