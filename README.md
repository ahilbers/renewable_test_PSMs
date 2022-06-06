[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Renewable test power system models


## Overview

#### Summary

<a href="https://www.youtube.com/watch?v=-dk3CVzaGew"> <img align="left" width="300" src="documentation/youtube_link.png"/> </a> 

This repository contains simple test power (energy) system models to use in renewable energy, time series and optimisation analysis, designed specifically for climate scientists who want to get a feel for energy systems. They include *generation & transmission expansion planning* (GTEP), *economic dispatch* (ED) and *unit commitment* (UC) type power system models. 

For a video introduction to energy system models, as well as a (slightly outdated) tutorial on this repo, see the video link on the left. For a very quick start, see [tutorial without installing](#tutorial-without-installing).

**Note**: This is a relatively new version that includes solar power. The version of the models [used in papers](#use-in-papers) do not contain solar power and are available under the branch `2020_papers`.

#### Rationale

There is considerable research into methods for generation & transmission expansion planning (GTEP), economic dispatch (ED) and unit commitment (UC) models. This includes:
- Time series aggregation, see e.g. [this paper](https://doi.org/10.3390/en13030641)
- Uncertainty analysis, see e.g. [this paper](https://doi.org/10.1016/j.esr.2018.06.003)
- New solution methods.

In most such investigations, a different model is used for each paper. Furthermore, models and the data used are usually not made public. This makes results from different studies hard to compare or reproduce. The closest thing to a standard for such applications are the various IEEE n-bus test systems, but the code, generation technologies and time series data are usually not standardised or provided open-source.

This repository provides a few simple test models to fill this gap. The models can be run “off-the-shelf”, containing pre-determined topologies, technologies and time series data. All that needs to be specified is the subset of time series data to use and a number of switches (e.g. integer or ramping constraints, whether to allow unmet demand) that ensure the model can contain most features seen in more complicated systems. These models are not modelling *frameworks* like [OseMOSYS](http://www.osemosys.org/) or [Calliope](https://www.callio.pe/) (which can be used to create arbitrary power system models, but are not models themselves). The models are built and can run in Python using the [Calliope](https://www.callio.pe/) package. Documentation and examples can be found below.

#### Models

<img align="right" src="documentation/6_region_diagram.jpg" alt="drawing" width="450" height="375">

Simple "toy" examples that run fast in most settings but have the features of more complicated examples. There are two base toplogies:
- The `1_region` model has only one region in which supply and demand must be met.
- The `6_region` model has six regions with a transmission topology, and supply and demand must be matched across the model but transmitted between the regions. It is based on a renewable version of the *IEEE 6-bus test system*.

Models can be run in two modes. In `plan` mode, both the optimal system design (generation and transmssion capacities) and subsequent operation (generation and transmission levels in each time step) are optimised. In `operate` mode, system design is user-defined and only the system's operation is optimised. Furthermore, integer and ramping constraints can be easily activated or deactivated depending on the modelling context. See `documentation/` for details.


#### Data

The time series input data consists of hourly demand levels and wind capacity factors for different European countries, forming a subset of the data available [here](https://doi.org/10.17864/1947.239). See `documentation/` for details.




## How to cite

If you use this repository in your own research, please cite the following paper and dataset:

- AP Hilbers, DJ Brayshaw, A Gandy (2020). Efficient quantification of the impact of demand and weather uncertainty in power system models. *IEEE Transactions on Power Systems*. [doi:10.1109/TPWRS.2020.3031187](https://doi.org/10.1109/TPWRS.2020.3031187).

- HC Bloomfield, DJ Brayshaw, A Charlton-Perez (2020). MERRA2 derived time series of European country-aggregate electricity demand, wind power generation and solar power generation. University of Reading. Dataset. doi:[10.17864/1947.239](https://doi.org/10.17864/1947.239).




## Usage

#### Tutorial without installing

For a quick introduction to the models, see [this link](https://mybinder.org/v2/gh/ahilbers/renewable_test_PSMs/HEAD), then open `tutorial.ipynb`. It is a [binder](https://mybinder.readthedocs.io/en/latest/) instance of the tutorial (`tutorial.ipynb`) that you can run in the browser, without having to install any pacakges on your own machine. Thanks to [Anne Fouilloux](https://github.com/annefou) for setting this up.


#### Customising and running your own simulations

To use these models in your own code, or customise them, you'll have to install the package (see section below). Then, you can run a sample simulation via

```
python3 scripts/main.py
```

This file is a template run illustrating the models' functionality. You can customise it for your own simulations.




## Requirements & installation

The models in this repo are power system *optimisation* models, so each simulation involves three parts:
1. Creating an optimisation problem associated with the power system (in python)
2. Solving the optimisation problem (usually done outside python)
3. Reading the results back and presenting them in a meaningful way (in python)

Hence, to run a simulation, you'll have to install this package (the python files), as well as a solver that solves the optimisation problem in step 2.


#### Installing this package

A key package used in this repo is [Calliope](https://www.callio.pe/), an open-source energy modelling framework. At the time of writing, the newest version of Calliope is `0.6.7`, which experiences bugs when installed with python `3.9` or higher. Hence, we recommend installing this package into python `3.8`. As usual, doing this via a virtual environment is recommended. From there, run

```
pip install -e .
```


#### Installing a solver

We recommend using [CBC](https://projects.coin-or.org/Cbc), which is open source and free-to-use. If you have `conda`, you can install it via

```
conda install -c conda-forge coincbc
```

If not, see install instructions for your system on the [project website](https://projects.coin-or.org/Cbc). 

[Gurobi](https://www.gurobi.com/) is a faster solver, but it requires a license. A free academic license is available. To change the solver used, change the value of `run.solver` in `models/1_region/model.yaml` or `models/6_region/model.yaml`, e.g. to `gurobi`.


#### Running the tests
This repo contains some unit tests, located in the `test` directory. You can run all unit tests by
calling `pytest` from a command line. To do this, you'll need to have [pytest-cov](https://pypi.org/project/pytest-cov/) installed.




## Use in papers

Specific (modified) version of these models have been used in two papers:

- AP Hilbers, DJ Brayshaw, A Gandy (2020). Efficient quantification of the impact of demand and weather uncertainty in power system models. *IEEE Transactions on Power Systems*. [doi:10.1109/TPWRS.2020.3031187](https://doi.org/10.1109/TPWRS.2020.3031187).

- AP Hilbers, DJ Brayshaw, A Gandy (2020). Importance subsampling for power system planning under multi-year demand and weather uncertainty. In proceedings of the *16th International Conference on Probabilistic Methods Applied to Power Systems (PMAPS 2020)*. [doi.org/10.1109/PMAPS47429.2020.9183591](https://doi.org/10.1109/PMAPS47429.2020.9183591)




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
