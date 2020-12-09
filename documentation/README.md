# Model documentation




## Usage & customisation

This repository provides simple test models and sample code for renewable energy and time series analysis. For this reason, the models are simple but have most of the features of full-scale models. The code has been written to make the models as accessible and quick-to-use as possible: simply modify the call to `models.OneRegionModel` or `models.SixRegionModel` with different values for the following arguments:

- `ts_data`: the demand and wind time series data to use
- `run_mode`:
  - `plan`: generation and transmission capacities, as well as subsequent generation levels, are determined by minimising sum of installation and generation costs
  - `operate`: generation and transmission capacities are fixed and only the generation levels are optimised to determine the minimum generation cost
- `baseload_integer`
  - `False`: baseload may be built to any nonnegative capacity (as a continuous variable)
  - `True`: baseload may be built only in blocks of 3GW, which makes a model in `plan` mode a mixed-integer linear program and slower to solve. This switch matters only in `plan` mode, since in `operate` mode the capacities are user-defined.
- `baseload_ramping`:
  - `False`: baseload generation can change at any rate
  - `True`: baseload generation can ramp up or down at most 20% of its installed capacity per hour
- `allow_unmet`:
  - `False`: all demand must be met
  - `True`: some demand may be left unmet, with a cost equal to the "value of lost load". This is always true if the `run_mode=operate`.

The models may be customised further by editing the model-defining files:
- Costs associted with each generation & transmission technologies: `models/{MODEL_NAME}/techs.yaml`.
- Fixed generation and transmission capacities used in `operate` mode: `models/{MODEL_NAME}/model.yaml`.

**Important note**: Any model outputs that are extensive (becoming larger with increasing simulation length, e.g. costs, generation levels, but not capacities) are annualised when called from `get_summary_outputs`. This means that for a run of 1 year vs 2 years, the costs and generation levels do not double. To return to extensive values, multiply by the simulation length in years.




## Model details

#### `1_region`:

In this model, an hourly demand time series must be met by generation from a combination of baseload, peaking and wind technologies, with some allowance for unmet demand at high cost. The model takes two time series: hourly demand levels and wind capacity factors for the UK.


#### `6_region`:

<img align="right" src="6_region_diagram.jpg" alt="drawing" width="500" height="415">

This model has 6 regions. Supply and demand must match across the model as a whole but electricity may be transmitted around the grid according to a topology inspired by the *IEEE 6-bus test system* and [Kamalinia & Shahidehpour (2010)](https://doi.org/10.1049/iet-gtd.2009.0695). The regions contain the following demand and generation technologies:
- Region 1: baseload & peaking generation
- Region 2: demand, wind and solar generation, with time series from Germany
- Region 3: baseload & peaking generation
- Region 4: demand, with time series from France
- Region 5: demand, wind and solar generation, with time series from the United Kingdom
- Region 6: baseload, peaking, wind and solar generation, with time series from Spain
Transmission is permitted between regions 1-2, 1-5, 1-6, 2-3, 3-4, 4-5 and 5-6.





## Generation & transmission technologies

| Type | Technology | Installation cost <br> (£m/GWyr) | Generation cost <br> (£m/GWh) | Carbon Emissions <br> (t CO2/GWh) |
| -- | -- | -- | -- | -- |
| Generation   | Baseload      | 300 | 0.005 | 200 |
| Generation   | Peaking       | 100 | 0.035 | 400 |
| Generation   | Wind          | 100 |     - |   - |
| Generation   | Solar         |  30 |     - |   - |
| Generation   | Unmet demand  |   - |     6 |   - |
| Transmission | Regions 1-5   | 150 |     - |   - |
| Transmission | Other regions | 100 |     - |   - |

For modelling purposes, unmet demand is considered a generation technology with no installation cost but a highy generation cost ("value of lost load"). The installation costs are annualised to reflect the cost per year of technology lifetime.

In the `6_region` model, the costs of the same technologies in different regions (e.g. baseload in regions 1 and 3) are perturbed slightly to remove solution nonuniqueness between regions -- so baseload in region 1 is (very slightly) different than baseload in region 3. Details can be found in `models/6_region/techs.yaml`.












## Additional information

Additional information, such as the precise mathematical optimisation problem solved by each model, are available in `documentation.pdf`.
