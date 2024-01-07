'''
Info on run_config keys:
------------------------
model_name (str) : '1_region' or '6_region'
ts_first_period (str) : first period of time series, slice, e.g. '2017-06-08'
ts_last_period (str) : last period of time series slice, e.g. '2017-06-15'
run_mode (str) : 'plan' or 'operate': whether model determines optimal generation and
    transmission capacities or optimises operation with a fixed setup
baseload_integer (bool) : baseload integer capacity constraint (units of 3GW)
baseload_ramping (bool) : baseload ramping constraint
allow_unmet (bool) : allow unmet demand in 'plan' mode (always allowed in operate mode)
fixed_caps (dict[str, float]) : fixed generation and transmission capacities. See
    `tutorial.ipynb` for an example.
extra_override (str) : name of additional override, should be defined in relevant `model.yaml`
output_save_dir (str) : name of directory where outputs are saved
save_full_model (bool) : save all model properies and results in addition to summary outputs
'''

from .one_region_operate import config_one_region_operate
from .one_region_plan import config_one_region_plan
from .six_region_operate import config_six_region_operate
from .six_region_plan import config_six_region_plan
