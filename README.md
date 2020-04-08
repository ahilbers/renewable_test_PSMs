[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Test case renewable power system models




## Overview

#### Summary

This repository contains model files, time series data and example code for a class of simple test power system models to use as benchmarks in renewable energy, time series and optimisation method analysis.

#### Rationale

In many fields, standard benchmarks exist; notable examples are [MNIST](http://yann.lecun.com/exdb/mnist/) or [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html) in Computer Vision and the [Lorenz 63](https://en.wikipedia.org/wiki/Lorenz_system) system in Dynamical Systems. In contrast, test models used in power system research tend to differ per investigation, with each paper using a different (often not open-source) model. The closest thing in power system applications are the various IEEE n-bus test systems, but the code, technology characteristics and time series data are usually not standardised or provided open-source.

This repository provides a few simple test models to fill this gap. The models can be run “off-the-shelf”, containing pre-determined topologies, technologies and time series data. All that needs to be specified is the subset of time series data to use and a number of switches (e.g. integer or ramping constraints, whether to allow unmet demand) that ensure the model can contain contain most features seen in more complicated systems. These models are not modelling frameworks like [OseMOSYS](http://www.osemosys.org/) or [Calliope](https://calliope.readthedocs.io/en/stable/) (which can be used to create arbitrary power system models, but are not models themselves). The models are built and can run in Python using the [Calliope](https://calliope.readthedocs.io/en/stable/) package. Documentation and examples can be found below.

#### Models

<img align="right" src="documentation/6_region_diagram.jpg" alt="drawing" width="450" height="375">

The models are designed to be simple "toy" examples (and hence run fast in most settings), but have all the features of more complicated power system models. There are two base models:
- The `1_region` model has only one region in which supply and demand must be met.
- The `6_region` model has six regions with a transmission topology, and supply and demand must be matched across the model but transmitted between the regions. It is based on a renewable version of the *IEEE 6-bus test system*.

Both models can be run in two modes. In `plan` mode, both the optimal system design (generation and transmssion capacities) and subsequent operation (generation and transmission levels in each time step) are optimised. In `operate` mode, system design is user-defined and only the system's operation is optimised. Furthermore, integer and ramping constraints can be easily activated or deactivated depending on the modelling context. See `documentation/` for details on the models.


#### Data

The time series input data consists of hourly demand levels and wind capacity factors for different European countries, forming a subset of the data available [here](https://doi.org/10.17864/1947.239). See `documentation/` for details.






## How to cite

If you use this repository in your own research, please cite the following paper and dataset:

- AP Hilbers, DJ Brayshaw, A Gandy (2020, *in review*). Quantifying demand and weather uncertainty in power system models using the *m* out of *n* bootstrap. [arXiv:1912.10326](https://arxiv.org/abs/1912.10326).

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239).




## Usage

The easiest way to start using these models is by modifying the example script provided. The file `main.py` contains three example runs with different specs.
- `1_region_plan_LP`: run the `1 region` model in planning mode. For baseload technology, any nonnegative capacity is allowed and there is no ramping constraint. This makes the optimisation problem a continuous linear program. All demand must be met -- no unmet demand is allowed.
- `6_region_plan_MILP_unmet`: run the `6 region` model in planning mode. Baseload technology may only be installed in integer units of 3GW, and has a ramping constraint of 20%/hr. This makes the optimisation problem a mixed-integer linear program. Unmet demand is allowed at high cost. 
- `6_region_operate`: run the `6 region` model in operate mode. Baseload technology has a ramping constraint. The installed capacities of the generation and transmission technologies are defined in `models/6_region/model.yaml`, and only the generation levels are determined by the model. Unmet demand is allowed at high cost.

Each of these examples can be run from a unix command line via:

```
python3 main.py --run_name {RUN_NAME}
```

where `{RUN_NAME}` is either `1_region_plan_LP`, `6_region_plan_MILP` or `6_region_operate`. By default, the verbosity of logging (print statements) is `INFO`. This can be changed (e.g. to `WARNING`) via an additional `--logging_level` argument.

A run may be customised by creating a custom `run_dict` and `ts_data` in `main.py`. Runs may give various warnings from the `Calliope` backend. These can usually be ignored as they refer to deliberate choices that do not affect model results. If you want to check the models are behaving as expected, the `summary_outputs.csv` of the three test simulations above should match `1_region_plan_continuous_2017-01.csv`, `6_region_plan_integer_ramping_2017-01.csv` and `6_region_operate_integer_ramping_2017-01.csv` respectively in the `benchmarks/` directory.




## Contains

### Modelling & data files
- `models/`: the files that define the power system model, in the open-source modelling framework `Calliope` (see acknowledgements)
- `data/demand_wind.csv`: the full 38-year demand and wind time series data for the `6_region` model. For the `1_region` model, specific demand and wind columns must be chosen. By default these are both taken from `region_5`.


### Code
- `main.py`: example runs of the models
- `models.py`: the models as Python classes, along with some helper functions
- `tests.py`: tests used in development to check the models behave as expected. These tests can be modified and may be helpful if you want to customise the models.


### Miscellaneous
- `documentation/`: documentation on the models
- `benchmarks/`: benchmark outputs used in the tests. Helpful for debugging or when customising models.




## Requirements & Installation

Since `main.py` is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.6.5`:  see [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation
  - `pandas 0.24.2`
- Other:
  - `gurobi`: a solver, used to solve the optimisation problems. It is not open-source but is free with an academic license. An alternative is `cbc` (see [this link](https://projects.coin-or.org/Cbc)), which is fully open-source. The solver can be specified in `models/{MODEL_NAME}/model.yaml`.




## Use in bootstrap uncertainty quantification paper

The models in this repository were used in the following paper:

AP Hilbers, DJ Brayshaw, A Gandy (2020, *in review*). Quantifying demand and weather uncertainty in power system models using the *m* out of *n* bootstrap. [arXiv:1912.10326](https://arxiv.org/abs/1912.10326).

The models used in the paper were constructed using the version of this repository at the time of first release (`v1.0`). This release was given the doi [10.5281/zenodo.3612174](https://zenodo.org/badge/latestdoi/229025818). The models correspond as follows:

| Model in paper | Model in this repository |
| -- | -- |
| *1-region LP*   | `1_region, run_mode=plan, baseload_integer=False, baseload_ramping=False, allow_unmet=True` |
| *6-region LP*   | `6_region, run_mode=plan, baseload_integer=False, baseload_ramping=False, allow_unmet=True` |
| *6-region MILP*   | `6_region, run_mode=plan, baseload_integer=True, baseload_ramping=True, allow_unmet=True` |

Code for the results in this paper is available in [this repository](https://github.com/ahilbers/2020_bootstrap_uncertainty_quantification).





## Contact

[Adriaan Hilbers](https://ahilbers.github.io). Department of Mathematics, Imperial College London. [a.hilbers17@imperial.ac.uk](mailto:a.hilbers17@imperial.ac.uk).




## Acknowledgements

Models are constructed in the modelling framework `Calliope`, created by Stefan Pfenninger and Bryn Pickering. See [callio.pe](https://callio.pe) or the following paper for details:

- S Pfenninger, B Pickering (2018). Calliope: a multi-scale energy systems modelling framework. Journal of Open Source Software, 3(29), 825, doi:[10.21105/joss.00825](https://doi.org/10.21105/joss.00825).

The demand and wind time series is a subset of columns from the following dataset:

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239)

Details about the creation of this data can be found in the following paper:

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2019). Characterising the winter meteorological drivers of the European electricity system using Targeted Circulation Types. Meteorological Applications. ISSN 1469-8080 (in press). doi:[10.1002/met.1858](https://doi.org/10.1002/met.1858).

The `6_region` model topology is based on the IEEE 6-bus test system, used in many previous studies. The renewable-ready topology, including the links and locations of demand & supply technologies, is based on a renewable 6-bus model, introduced in the following paper:

- S Kamalinia, M Shahidehpour (2010). Generation expansion planning in wind-thermal power systems. IET Generation, Transmission & Distribution, 4(8), 940-951. doi:[10.1049/iet-gtd.2009.0695](https://doi.org/10.1049/iet-gtd.2009.0695)
