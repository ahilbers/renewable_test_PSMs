import logging
import json
import pandas as pd


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


def get_logger(name: str, run_config: dict) -> logging.Logger:
    """Get a logger to use throughout the codebase.
    
    Parameters:
    -----------
    name: logger name
    config: dictionary with model, run and save properties
    """

    output_save_dir = run_config['output_save_dir']

    # Create the master logger and formatter
    logger = logging.getLogger(name=name)
    logger.setLevel(level=getattr(logging, run_config['logging_level']))
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-8s - %(name)s - %(filename)s - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Create two handlers: one writes to a log file, the other to stdout
    logger_file = logging.FileHandler(f'{output_save_dir}/model_run.log')
    logger_file.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=logger_file)
    logger_stdout = logging.StreamHandler()
    logger_stdout.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=logger_stdout)

    return logger


def load_time_series_data(model_name: str) -> pd.DataFrame:
    """Load demand, wind and solar time series data for model.

    Parameters:
    -----------
    model_name: '1_region' or '6_region'
    """

    ts_data = pd.read_csv('data/demand_wind_solar.csv', index_col=0)
    ts_data = ts_data.clip(lower=0.)  # Trim negative values, can come from floating point error
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # If 1_region model, take demand, wind and solar from region 5
    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5', 'solar_region5']]
        ts_data.columns = ['demand', 'wind', 'solar']

    logger.debug(f'Loaded raw time series data:\n\n{ts_data}\n')
    return ts_data


def has_missing_leap_days(ts_data: pd.DataFrame) -> bool:
    """Detect if a time series has missing leap days.

    Parameters:
    -----------
    ts_data : time series to check for missing leap days
    """
    index = ts_data.index
    feb28_index = ts_data.index[(index.year % 4 == 0) & (index.month == 2) & (index.day == 28)]
    feb29_index = ts_data.index[(index.year % 4 == 0) & (index.month == 2) & (index.day == 29)]
    mar01_index = ts_data.index[(index.year % 4 == 0) & (index.month == 3) & (index.day == 1)]
    if len(feb29_index) < min((len(feb28_index), len(mar01_index))):
        return True
    return False


def get_scenario(
    run_mode: str, baseload_integer: bool, baseload_ramping: bool, allow_unmet: bool
) -> str:
    """Get scenario name, a comma-separated list of overrides in `model.yaml` for Calliope model

    Parameters:
    -----------
    run_mode: 'plan' or 'operate'
    baseload_integer: activate baseload discrete capacity constraint
    baseload_ramping: enforce baseload ramping constraint
    allow_unmet: allow unmet demand in 'plan' mode (should always be allowed in 'operate' mode)
    """

    scenario = run_mode
    if run_mode == 'plan':
        if baseload_integer:
            scenario += ',integer'
        else:
            scenario += ',continuous'
        if allow_unmet:
            scenario += ',allow_unmet'
    if baseload_ramping:
        scenario += ',ramping'

    logger.debug(f'Created Calliope model scenario: `{scenario}`.')

    return scenario


def get_cap_override_dict(model_name: str, fixed_caps: dict) -> dict:
    """Create override dictionary used to set fixed fixed capacities in Calliope model.

    Parameters:
    -----------
    model_name: '1_region' or '6_region'
    fixed_caps: fixed capacities -- `model.get_summary_outputs(as_dict=True)` has correct format

    Returns:
    --------
    o_dict: Dict that can be fed as 'override_dict' into Calliope model in 'operate' mode
    """

    if not isinstance(fixed_caps, dict):
        raise ValueError('Incorrect input format for fixed_caps')

    o_dict = {}  # Populate this override dict

    # Add generation capacities capacities for 1_region model
    if model_name == '1_region':
        for tech, attribute in [
            ('baseload', 'energy_cap_equals'),
            ('peaking', 'energy_cap_equals'),
            ('wind', 'resource_area_equals'),
            ('solar', 'resource_area_equals')
        ]:
            idx = (f'locations.region1.techs.{tech}.constraints.{attribute}')
            o_dict[idx] = fixed_caps[f'cap_{tech}_total']

    # Add generation and transmission capacities for 6_region model
    elif model_name == '6_region':
        for region in [f'region{i+1}' for i in range(6)]:

            # Add generation capacities
            for tech, attribute in [
                ('baseload', 'energy_cap_equals'),
                ('peaking', 'energy_cap_equals'),
                ('wind', 'resource_area_equals'),
                ('solar', 'resource_area_equals')
            ]:
                # If this technology is specified in this region, add it to o_dict
                fixed_caps_key = f'cap_{tech}_{region}'
                if fixed_caps_key in fixed_caps:
                    idx = f'locations.{region}.techs.{tech}_{region}.constraints.{attribute}'
                    o_dict[idx] = fixed_caps[fixed_caps_key]

            # Add transmission capacities
            for region_to in [f'region{i+1}' for i in range(6)]:
                # If this technology is specified in this region, add it to o_dict
                fixed_caps_key = f'cap_transmission_{region}_{region_to}'
                if fixed_caps_key in fixed_caps:
                    idx_regions = f'{region},{region_to}.techs.transmission_{region}_{region_to}'
                    idx = f'links.{idx_regions}.constraints.energy_cap_equals'
                    o_dict[idx] = fixed_caps[fixed_caps_key]

    if len(o_dict) == 0:
        raise AttributeError('Override dict is empty. Check if something has gone wrong.')

    logger.debug(f'Created override dict:\n{json.dumps(o_dict, indent=4)}')
    return o_dict