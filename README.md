[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Test case renewable power system models




## Overview

#### Summary

This repository contains model files, time series data and example code for a class of simple test power system models to use in renewable energy, time series and optimisation analysis. They include *generation & transmission expansion planning* (G/TEP), *economic dispatch* (ED) and *unit commitment* (UC) type power system models.

This is a beta version that includes solar power. Currently, this has only been implemented for the `1_region` model. The original models (which contain only wind power but no solar), including tests and extensive documentation, are available under the branch `master`.




## Usage

#### Getting started

For a (very quick) tutorial, see the `tutorial.ipynb` notebook. This shows how to use the models and gives a feel for the syntax.




## Requirements & Installation

Since `main.py` is a short file with only a few functions, it's probably easier to directly copy-paste any relevant code into a personal project as opposed to installing a new module. For this reason, this repository does not contain a `setup.py` file.

Running `main.py` requires:
- Python modules:
  - `Calliope 0.6.6`: the modelling framework creating and reading and in the results of the optimisation problem. See [this link](https://calliope.readthedocs.io/en/stable/user/installation.html) for installation.
  - `pandas 0.24.2`.
- Other:
  - `cbc`: an open-source solver, used to solve the optimisation problems. See [this link](https://projects.coin-or.org/Cbc) for installation. Faster solution times, especially when `baseload_integer=True`, can be achieved with `Gurobi`, a commercial solver that is free to use with an academic license. The solver can be specified in `models/{MODEL_NAME}/model.yaml`.




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
