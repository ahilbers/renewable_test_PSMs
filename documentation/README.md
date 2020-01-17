# Model documentation




## Usage & customisation

This repository seeks to provide simple test cases and sample code for renewable energy and time series analysis. For this reason, the models are simple, but have all the features of full-scale models. The code has been written to make the models as accessible and quick-to-use as possible. Running a model is as simple as modifying `main.py` with different values for the following entries in the `run_dict`:

- `model_name`: which model to run `1_region` or `6_region`.
- `ts_data`: the demand and wind time series data to use in the model.
- `run_mode`:
  - `plan`: generation and transmission capacities are determined by minimising system (sum of installation and generation) costs.
  - `operate`: generation and transmission capacities are fixed and the system is operated to match supply and demand at minimum cost.
- `baseload_integer`
  - `False`: baseload may be built to any nonnegative capacity (i.e. a continuous variable).
  - `True`: baseload may be built only in blocks of 3GW, which makes a model in `plan` mode a mixed-integer linear program and slower to solve. This switch matters only in `plan` mode, since in `operate` mode the capacities are user-defined.
- `baseload_ramping`:
  - `False`: baseload generation can change at any rate
  - `True`: baseload generation can only ramp up or down at 20% of its installed capacity per hour

The models may be customised further by editing the model-defining files:
- Costs associted with each generation & transmission technologies: change in `models/{MODEL_NAME}/techs.yaml`.
- Fixed generation and transmission capacities used in `operate` mode: change in `models/{MODEL_NAME}/model.yaml`.




## Model details

<img align="right" src="6_region_diagram.jpg" alt="drawing" width="500" height="415">

#### `1_region`:

In this model, an hourly demand time series must be met by generation from a combination of baseload, peaking and wind technologies, with some allowance for unmet demand at high cost. The model takes two time series: hourly demand levels and wind capacity factors for the UK.


#### `6_region`:

This model has 6 regions. Supply and demand must meet across the model as a whole but electricity may be transmitted around the grid according to a topology inspired by the *IEEE 6-bus test system* and [Kamalinia & Shahidehpour (2010)](https://doi.org/10.1049/iet-gtd.2009.0695). The regions contain the following demand and generation technologies:
- Region 1: baseload & peaking generation
- Region 2: demand and wind generation
- Region 3: baseload & peaking generation
- Region 4: demand
- Region 5: demand and wind generation
- Region 6: baseload, peaking & wind generation
Transmission is permitted between regions 1-2, 1-5, 1-6, 2-3, 3-4, 4-5 and 5-6.





## Generation & transmission technologies

### Generation technologies

| Technology | Installation cost <br> (£m/GWyr) | Generation cost <br> (£m/GWh) | Carbon Emissions <br> (t CO2/GWh) |
| -- | -- | -- | -- |
| Baseload     | 300 | 0.005 | 200 |
| Peaking      | 100 | 0.035 | 400 |
| Wind         | 100 |     0 |   0 |
| Unmet demand |   0 |     6 |   0 |

### Transmission technologies (in `6 region` model)

| Regions | Installation cost <br> (£m/GWyr) |
| -- | -- |
| Region 1 to 5 | 150 |
| Other         | 100 |












## Additional information

Additional information, such as the precise mathematical optimisation problem solved for each model setting, are available in the following paper:

| Model in paper | Model in this repository |
| -- | -- |
| *1-region LP*   | `1 region, run_mode=plan, baseload_integer=False, baseload_ramping=False` |
| *6-region LP*   | `6 region, run_mode=plan, baseload_integer=False, baseload_ramping=False` |
| *6-region MILP*   | `6 region, run_mode=plan, baseload_integer=True, baseload_ramping=True` |
