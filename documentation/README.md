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

In each model, the allowed generation technologies are baseload, peaking, wind and solar. Baseload and peaking are conventional technologies whose generation levels can be controlled. Wind and solar have no generation cost but generation levels that are capped by the time-varying wind and solar capacity factors respectively. For modelling purposes, unmet demand is considered a generation technology with no installation cost but a highy generation cost ("value of lost load"). The installation costs are annualised to reflect the cost per year of technology lifetime. For parameter values, see `models/*_region/techs.yaml`.

In the `6_region` model, the costs of the same technologies in different regions (e.g. baseload in regions 1 and 3) are perturbed slightly to remove solution nonuniqueness between regions -- so baseload in region 1 is (very slightly) different than baseload in region 3. Details can be found in `models/6_region/techs.yaml`.




## Storage technologies

A single storage technology is considered. For parameter values, see `models/*_region/techs.yaml`. Storage technologies have both an energy power (KW) and storage (KWh) capacity, which can be priced and constrained individually. Alternatively, you can define an `energy_cap_per_storage_cap_equals` which fixes the ratio (see `models/1_region/techs.yaml`), and use only a single one of these costs.



## Time series inputs

Time series inputs consist of hourly demand levels, wind capacity factors and solar capacity factor in different European countries over the period 1980-2017. Demand levels are measured in GWh and capacity factors are dimensionless between 0 and 1. Long-term anthropogenic demand trends such as GDP growth and efficiency improvements are removed so that the time series can be viewed as being drawn from the same underlying demand distribution. Details on the construction of the time series can be found in [Bloomfield et al, 2019](https://doi.org/10.1002/met.1858).




## Mathematical details

| Term | Description |
| -- | -- |
| | **Indices** |
| $i$ | Generation technology |
| $r,r'$ | Region |
| t | Time step |
| | **Sets** |
|  $\mathcal{I}$ | Technologies: baseload ($b$), peaking ($p$), wind ($w$), solar ($s$), unmet demand ($u$) |
| $\mathcal{D}$ | Decision variables |
| | **Parameters** |
| $C_{i}^\text{gen}$ | Install cost, technology $i$ (£/KWyr)|
| $C_{r, r'}^\text{tr}$ | Install cost, transmission, region $r$ to $r'$ (£/KWyr) |
| $F_{i}^\text{gen}$ | Generation cost, technology $i$ (£/KWh) |
| | **Time series** |
| $d_{r, t}$ | Demand, region $r$, time step $t$ (KWh) |
| $w_{r, t}$ | Wind capacity factor, region $r$, time $t$ ($\in$ [0, 1]) |
| $s_{r, t}$ | Solar capacity factor, region $r$, time $t$ ($\in$ [0, 1]) |
| | **Decision variables** |
| $\text{cap}_{i, r}^\text{gen}$ | Generation capacity, technology $i$, region $r$ (KW) |
| $\text{cap}_{r, r'}^\text{tr}$ | Transmission capacity, region $r$ to $r'$ (KW) |
| $\text{gen}_{i, r, t}$ | Generation, technology $i$, region $r$, time $t$ (KWh) |
| $\text{tr}_{r, r', t}$ | Transmission, region $r$ to region $r'$, time $t$ (KWh) | \hline


### 1-region model

#### Planning model

Minimise

$$
\Bigg[ \frac{T}{8760}
  \Bigg( \! \underbrace{\sum_{i \in \mathcal{I}} C_i^\text{gen} \text{cap}^\text{gen}_{i}}_{\substack{\text{install cost,} \\ \text{generation}}}
  + \underbrace{ \vphantom{\sum_{i \in \mathcal{I}}} C^\text{sto} \text{cap}^\text{sto} }_{\substack{\text{install cost,} \\ \text{storage}}} \!\! \Bigg)
  + \underbrace{ \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} F_i^\text{gen} \text{gen}_{i,t}}_\text{generation cost} \Bigg]
$$
by optimising over decision variables $\textbf{D}$ and $(\textbf{O}_t)_{t \in \mathcal{T}}$, where
$$
  \textbf{D} = [\text{cap}^\text{gen}_{i}, \ \text{cap}^\text{sto} \ | \ i \in \mathcal{I}]
$$
$$
  \textbf{O}_t = [\text{gen}_{i,t},\ \ch_{t} \ | \ i \in \mathcal{I}]
$$
subject to
$$
  \sum_{i \in \mathcal{I}} \text{gen}_{i,t} = d_{t} + \ch_{t} \quad \text{for all} \ t
$$
$$
  \text{sto}_{0} = 0
$$
$$
  \text{sto}_{t+1} = (1 - l^\text{sto}) \text{sto}_{t} +
  \begin{cases}
    e^\text{sto} \ch_{t} \quad \text{if} \ \ch_{t} \ge 0 \\
    \frac{1}{e^\text{sto}} \ch_{t} \quad \text{if} \ \ch_{t} < 0
  \end{cases}
  \quad \text{for all} \ t
$$
$$
  0 \le \text{gen}_{i,t} \le \text{cap}^\text{gen}_{i} \quad \text{for all} \ i, t
$$
$$
  0 \le \text{gen}_{j,t} \le \text{cap}^\text{gen}_{j} \lambda_{j,t} \quad \text{for all} \ j=\{s, w\}, t
$$
$$
  0 \le \text{sto}_{t} \le \text{cap}^\text{sto} \quad \text{for all} \ t
$$
$$
  [\text{if baseload\_ramping = True}] \quad | \, \text{gen}_{b,t} - \text{gen}_{b,t+1} | \le 0.2 \text{cap}^\text{gen}_{b} \quad \text{for all} \ t
$$
$$
  [\text{if baseload\_integer = True}] \quad \text{cap}^\text{gen}_{b} \in 3\mathbb{Z}
$$
$$
  [\text{if allow\_unmet = False}] \quad \text{gen}_{u,t} = 0 \quad \text{for all} \: t.
$$

The vectors $\textbf{D}$ and $(\textbf{O}_t)_{t \ \in \mathcal{T}}$ are the design and operational decision variables. The factor $\frac{T}{8760}$ normalises install costs to the same temporal scale as generation costs, since $C_i^{gen}$ and $C_{r,r'}^{tr}$ are costs per year of plant lifetime and there are 8760 hours (time steps) in a year.


#### Operation model

The operation problem is the same as the planning problem, except that the design $\textbf{D}$ is fixed and its contribution removed from the objective function.


### 6-region model

#### Planning model

Minimise

$$
\sum_{r \in \mathcal{R}} \Bigg[ \frac{T}{8760}
  \Bigg( \! \underbrace{\sum_{i \in \mathcal{I}} C_i^\text{gen} \text{cap}^\text{gen}_{i,r}}_{\substack{\text{install cost,} \\ \text{generation}}}
  + \!\! \underbrace{ \sum_{r' \in \mathcal{R}} C_{r,r'}^\text{tr} \text{cap}^\text{tr}_{r,r'} }_{\substack{\text{install cost,} \\ \text{transmission}}}
  + \underbrace{ \vphantom{\sum_{i \in \mathcal{I}}} C^\text{sto} \text{cap}^\text{sto}_r }_{\substack{\text{install cost,} \\ \text{storage}}} \!\! \Bigg)
  + \underbrace{ \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} F_i^\text{gen} \text{gen}_{i,r,t}}_\text{generation cost} \Bigg]
$$
by optimising over decision variables $\textbf{D}$ and $(\textbf{O}_t)_{t \in \mathcal{T}}$, where
$$
  \textbf{D} = [\text{cap}^\text{gen}_{i,r}, \ \text{cap}^\text{tr}_{r,r'}, \ \text{cap}^\text{sto}_r \ | \ i \in \mathcal{I}; \ r \in \mathcal{R}; \ r' \in \mathcal{R}]
$$
$$
  \textbf{O}_t = [\text{gen}_{i,r,t}, \ \text{tr}_{r,r',t}, \ \ch_{r, t} \ | \ i \in \mathcal{I}; \ r \in \mathcal{R}; \ r' \in \mathcal{R}]
$$
subject to
$$
  \text{cap}^\text{gen}_{b, r} \big\rvert_{r \notin \{1,3,6\}} = \text{cap}^\text{gen}_{p,r} \big\rvert_{r \notin \{1,3,6\}} = \text{cap}^\text{gen}_{w, r} \big\rvert_{r \notin \{2,5,6\}} = 0
$$
$$
  \text{cap}^\text{tr}_{r,r'} \big\rvert_{(r,r') \notin  \{(1,2), (1,5), (1,6), (2,3), (3,4), (4,5), (5,6)\}} = 0
$$
$$
  \text{cap}^\text{sto}_r \big\rvert_{r \notin \{2,5,6\}} = 0
$$
$$
  \sum_{i \in \mathcal{I}} \text{gen}_{i,r,t} + \sum_{r' \in \mathcal{R}} \text{tr}_{r',r,t} = d_{r,t} + \ch_{r,t} \quad \text{for all} \ r, t
$$
$$
  \text{tr}_{r,r',t} + \text{tr}_{r,'r,t} = 0 \quad \text{for all} \ r, r', t
$$
$$
  \text{sto}_{r,0} = 0 \quad \text{for all} \ r
$$
$$
  \text{sto}_{r,t+1} = (1 - l^\text{sto}) \text{sto}_{r,t} +
  \begin{cases}
    e^\text{sto} \ch_{r,t} \quad \text{if} \ \ch_{r,t} \ge 0 \\
    \frac{1}{e^\text{sto}} \ch_{r,t} \quad \text{if} \ \ch_{r,t} < 0
  \end{cases}
  \quad \text{for all} \ r, t
$$
$$
  0 \le \text{gen}_{i,r,t} \le \text{cap}^\text{gen}_{i,r} \quad \text{for all} \ i, r, t
$$
$$
  0 \le \text{gen}_{j,r,t} \le \text{cap}^\text{gen}_{j,r} \lambda_{j,r,t} \quad \text{for all} \ j = \{w, s\}, r, t
$$
$$
  |\text{tr}_{r,r',t}| \le \text{cap}^\text{tr}_{r,r'} + \text{cap}^\text{tr}_{r',r} \quad \text{for all} \ r, r', t
$$
$$
  0 \le \text{sto}_{r,t} \le \text{cap}^\text{sto}_r \quad \text{for all} \ r, t
$$
$$
  [\text{if baseload\_ramping = True}] \quad | \, \text{gen}_{b,r,t} - \text{gen}_{b,r,t+1} | \le 0.2 \text{cap}^\text{gen}_{b,r} \quad \text{for all} \ r, t
$$
$$
  [\text{if baseload\_integer = True}] \quad \text{cap}^\text{gen}_{b,r} \in 3\mathbb{Z} \quad \text{for all} \ r
$$
$$
  [\text{if allow\_unmet = False}] \quad \text{gen}_{u,r,t} = 0 \quad \text{for all} \: r, t.
$$

The vectors $\textbf{D}$ and $(\textbf{O}_t)_{t \ \in \mathcal{T}}$ are the design and operational decision variables. The factor $\frac{T}{8760}$ normalises install costs to the same temporal scale as generation costs, since $C_i^{gen}$ and $C_{r,r'}^{tr}$ are costs per year of plant lifetime and there are 8760 hours (time steps) in a year.


#### Operation model

The operation problem is the same as the planning problem, except that the design $\textbf{D}$ is fixed and its contribution removed from the objective function.