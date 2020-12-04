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

To run `tutorial.ipynb`, you need [Calliope](https://www.callio.pe/), an open-source energy modelling framework. To install it, you can use the `anaconda` package manager. If you don't have this yet, download a minimal version [here](https://docs.conda.io/en/latest/miniconda.html). From there, run the following lines of code in a command line in the directory containing this repo:

```
conda create -c conda-forge -n calliope calliope
```

This creates a new virtual environment called `calliope`. Activate it using `conda activate calliope`. The next step is to install software that solves the optimisation problem. [CBC](https://projects.coin-or.org/Cbc) works well, and can be installed via

```
conda install -c conda-forge coincbc
```

Now, install the [jupyter notebook](https://jupyter.org/index.html) software using

```
conda install -c conda-forge jupyterlab
```

and, from here, call `jupyter notebook`. This opens a browser, and you should see `tutorial.ipynb`. You're all set!




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
