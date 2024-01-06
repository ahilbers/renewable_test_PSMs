import sys
import os
import logging
import re
import json
import pandas as pd
import calliope


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


def get_logger(name: str, run_config: dict) -> logging.Logger:
    '''Get a logger to use throughout the codebase.

    Parameters:
    -----------
    name: logger name
    config: dictionary with model, run and save properties
    '''

    output_save_dir = run_config['output_save_dir']

    # Create the master logger and formatter
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)  # Set master to lowest level -- gets overwritten by handlers
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-8s - %(name)s - %(filename)s - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Create two handlers: one writes to a log file, the other to stdout
    logger_file = logging.FileHandler(f'{output_save_dir}/model_run.log')
    logger_file.setFormatter(fmt=formatter)
    logger_file.setLevel(level=getattr(logging, run_config['log_level_file']))
    logger.addHandler(hdlr=logger_file)
    logger_stdout = logging.StreamHandler(sys.stdout)
    logger_stdout.setFormatter(fmt=formatter)
    logger_stdout.setLevel(level=getattr(logging, run_config['log_level_stdout']))
    logger.addHandler(hdlr=logger_stdout)

    return logger


def load_time_series_data(
    model_name: str,
    path: str = os.path.join(os.path.dirname(__file__), '..', 'data/demand_wind_solar.csv')
) -> pd.DataFrame:
    '''Load demand, wind and solar time series data for model.

    Parameters:
    -----------
    model_name: '1_region' or '6_region'
    path: path to CSV file
    '''

    ts_data = pd.read_csv(path, index_col=0)
    ts_data = ts_data.clip(lower=0.)  # Trim negative values, can come from floating point error
    ts_data.index = pd.to_datetime(ts_data.index)

    # If 1_region model, take demand, wind and solar from region 5
    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5', 'solar_region5']]
        ts_data.columns = ['demand', 'wind', 'solar']

    logger.debug(f'Loaded raw time series data:\n\n{ts_data}\n')
    return ts_data


def has_missing_leap_days(ts_data: pd.DataFrame) -> bool:
    '''Detect if a time series has missing leap days.

    Parameters:
    -----------
    ts_data : time series to check for missing leap days
    '''
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
    '''Get scenario name, a comma-separated list of overrides in `model.yaml` for Calliope model

    Parameters:
    -----------
    run_mode: 'plan' or 'operate'
    baseload_integer: activate baseload discrete capacity constraint
    baseload_ramping: enforce baseload ramping constraint
    allow_unmet: allow unmet demand in 'plan' mode (should always be allowed in 'operate' mode)
    '''

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


def get_cap_override_dict(model_name: str, run_mode: str, fixed_caps: dict) -> dict:
    '''Create override dictionary used to set fixed capacities in Calliope model.

    Parameters:
    -----------
    model_name: '1_region' or '6_region'
    run_mode: 'plan' or 'operate'
    fixed_caps: fixed capacities -- `model.get_summary_outputs(as_dict=True)` has correct format

    Returns:
    --------
    o_dict: Dict that can be fed as 'override_dict' into Calliope model in 'operate' mode
    '''

    if not isinstance(fixed_caps, dict):
        raise ValueError('Incorrect input format for fixed_caps')

    o_dict = {}  # Populate this override dict

    if model_name == '1_region':
        # Add capacities
        for tech, attribute in [
            ('baseload', 'energy_cap_equals'),
            ('peaking', 'energy_cap_equals'),
            ('wind', 'resource_area_equals'),
            ('solar', 'resource_area_equals'),
            ('storage_energy', 'storage_cap_equals'),
            ('storage_power', 'energy_cap_equals')
        ]:
            # If this technology is specified, add it to o_dict
            fixed_caps_key = f'cap_{tech}_total'
            tech_name_in_model = re.sub(r'storage_.*', 'storage_', tech)
            if fixed_caps_key in fixed_caps:
                idx = f'locations.region1.techs.{tech_name_in_model}.constraints.{attribute}'
                o_dict[idx] = fixed_caps[fixed_caps_key]
            elif run_mode == 'operate':
                raise ValueError(f'In operate mode, must set fixed {tech} capacity.')

        # If storage capacity is 0, make initial storage level 0
        if fixed_caps.get('cap_storage_energy_total') == 0.:
            o_dict['techs.storage_.constraints.storage_initial'] = 0.

    elif model_name == '6_region':

        # Do some checks that catch if not enough capacities are not specified in operate mode
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'storage']:
            fixed_cap_names = [i for i in fixed_caps.keys() if tech in i]
            if len(fixed_cap_names) < 3 and run_mode == 'operate':
                raise ValueError(
                    f'Expected 3 fixed {tech} capacities, but only {fixed_cap_names} specified.'
                )
        fixed_transmission_cap_names = [i for i in fixed_caps.keys() if 'transmission' in i]
        if len(fixed_transmission_cap_names) < 7 and run_mode == 'operate':
            raise ValueError(
                'Expected 7 fixed transmission capacities, '
                f'but only {fixed_transmission_cap_names} specified.'
            )

        for region in [f'region{i+1}' for i in range(6)]:

            # Add generation and storage capacities
            for tech, attribute in [
                ('baseload', 'energy_cap_equals'),
                ('peaking', 'energy_cap_equals'),
                ('wind', 'resource_area_equals'),
                ('solar', 'resource_area_equals')
            ]:
                # If this technology is specified in this region, add it to o_dict
                fixed_caps_key = f'cap_{tech}_{region}'
                tech_name_in_model = re.sub(r'storage_.*', 'storage', tech)
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

            # Add storage energy (MWh) and power (MW) capacities
            for tech, attribute in [
                ('storage_energy', 'storage_cap_equals'),
                ('storage_power', 'energy_cap_equals')
            ]:
                fixed_caps_key = f'cap_{tech}_{region}'
                if fixed_caps_key in fixed_caps:
                    idx = f'locations.{region}.techs.storage_{region}.constraints.{attribute}'
                    o_dict[idx] = fixed_caps[fixed_caps_key]

            # If storage capacity is 0, make initial storage level 0
            if fixed_caps.get(f'cap_storage_energy_{region}') == 0.:
                o_dict[f'techs.storage_{region}.constraints.storage_initial'] = 0.

    logger.debug(f'Created override dict:\n{json.dumps(o_dict, indent=4)}')
    return o_dict


def get_technology_info(model: calliope.Model) -> pd.DataFrame:
    '''Get technology install & generation costs and emissions from model config.'''

    model_dict = model._model_run
    costs = pd.DataFrame(columns=['install', 'generation', 'emissions'], dtype='float')
    regions = list(model_dict['locations'].keys())

    # Add the technologies in each region
    for region in regions:
        region_dict = model_dict['locations'][region]

        # Add generation technologies
        techs = list(region_dict['techs'].keys())
        for tech in techs:
            if ('demand' in tech) or ('storage' in tech):
                continue  # Generation technologies only here -- not demand or storage
            tech_costs_dict = region_dict['techs'][tech]['costs']
            is_variable_renewable = ('wind' in tech) or ('solar' in tech)
            install_cost_name = 'resource_area' if is_variable_renewable else 'energy_cap'
            costs.loc[tech, 'install'] = (
                0. if 'unmet' in tech else float(tech_costs_dict['monetary'][install_cost_name])
            )
            costs.loc[tech, 'generation'] = float(tech_costs_dict['monetary']['om_prod'])
            costs.loc[tech, 'emissions'] = float(tech_costs_dict['emissions']['om_prod'])

        # Add storage technologies
        for tech in techs:
            if 'storage' in tech:
                if model.model_name == '1_region':
                    tech_costs_dict = region_dict['techs']['storage_']['costs']
                    index_energy = 'storage_energy'
                    index_power = 'storage_power'
                elif model.model_name == '6_region':
                    tech_costs_dict = region_dict['techs'][f'storage_{region}']['costs']
                    index_energy = f'storage_energy_{region}'
                    index_power = f'storage_power_{region}'
                costs.loc[index_energy, ['install', 'generation', 'emissions']] = (
                    [float(tech_costs_dict['monetary'].get('storage_cap', 0.)), 0., 0.]
                )
                costs.loc[index_power, ['install', 'generation', 'emissions']] = (
                    [float(tech_costs_dict['monetary'].get('energy_cap', 0.)), 0., 0.]
                )

        # Add transmission technologies
        regions_to = region_dict.get('links', [])
        for region_to in regions_to:
            tech = f'transmission_{region}_{region_to}'
            tech_reversed = f'transmission_{region_to}_{region}'
            if tech_reversed in costs.index:
                continue  # Only count links in one direction
            tech_costs_dict = region_dict['links'][region_to]['techs'][tech]['costs']
            costs.loc[tech, 'install'] = float(tech_costs_dict['monetary']['energy_cap'])
            costs.loc[tech, 'generation'] = 0.
            costs.loc[tech, 'emissions'] = 0.

    logger.debug(f'Costs read from model config:\n\n{costs}\n')

    return costs


def get_tech_regions(model: calliope.Model) -> list[tuple[str]]:
    '''Get list of tuples of generation, transmission and storage technologies.
    Ex: [('wind', 'region1'), ('transmission', 'region1', 'region2'), ('storage', 'region1')] '''

    # Get techs and regions from model via 'get_technology_info' function. Remove double entries
    # 'energy' and 'power' for 'storage' -- just one entry, called 'storage'
    tech_regions = [tuple(i.split('_')) for i in get_technology_info(model).index]
    tech_regions = [i for i in tech_regions if not (i[0] == 'storage' and i[1] == 'power')]
    tech_regions = [(i[0], i[2]) if i[0] == 'storage' else i for i in tech_regions]

    return tech_regions
