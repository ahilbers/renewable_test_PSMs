import sys
import os
import logging
import re
import json
import numpy as np
import pandas as pd
import calliope


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


def get_logger(name: str, run_config: dict) -> logging.Logger:  # pragma: no cover
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
) -> pd.DataFrame:  # pragma: no cover
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


def has_missing_leap_days(ts_data: pd.DataFrame) -> bool:  # pragma: no cover
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
        raise ValueError('Incorrect input format for fixed_caps')  # pragma: no cover

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
        for region in [f'region{i+1}' for i in range(6)]:

            # Add generation and storage capacities
            for tech, attribute in [
                ('baseload', 'energy_cap_equals'),
                ('peaking', 'energy_cap_equals'),
                ('wind', 'resource_area_equals'),
                ('solar', 'resource_area_equals'),
                ('storage_energy', 'storage_cap_equals'),
                ('storage_power', 'energy_cap_equals')
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

            # If storage capacity is 0, make initial storage level 0
            if fixed_caps.get('cap_storage_energy_total') == 0.:
                o_dict['techs.storage_.constraints.storage_initial'] = 0.

            # Do some checks that catch if not enough capacities are not specified in operate mode
            for tech in ['baseload', 'peaking', 'wind', 'solar', 'storage']:
                fixed_caps = [i for i in o_dict.keys() if tech in i]
                if len(fixed_caps) < 3 and run_mode == 'operate':
                    raise ValueError(f'Expected 3 fixed {tech} capacities, but got {fixed_caps}.')
            fixed_transmission_caps = [i for i in o_dict.keys() if 'transmission' in i]
            if len(fixed_transmission_caps) < 7 and run_mode == 'operate':
                raise ValueError(
                    f'Expected 7 fixed transmission capacities, but got {fixed_transmission_caps}.'
                )

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


def _has_consistent_outputs_1_region(model: calliope.Model) -> bool:  # pragma: no cover
    '''Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of OneRegionModel
    '''

    passing = True  # Changes to False if any outputs are found to be inconsistent
    cost_total_v1 = 0

    costs = get_technology_info(model=model)
    techs = list(costs.index)
    gen_techs = [i for i in techs if (('storage' not in i) and ('unmet' not in i))]
    storage_techs = [i for i in techs if 'storage' in i]
    unmet_techs = [i for i in techs if 'unmet' in i]

    sum_out = model.get_summary_outputs()
    ts_out = model.get_timeseries_outputs()
    inp = model.inputs
    res = model.results

    # Normalise install costs to same temporal scale as generation costs
    corrfac = model.num_timesteps / 8760

    if model.run_mode == 'plan':

        # Test if generation technology installation costs are consistent
        for tech in gen_techs:
            cost_v1 = corrfac * float(costs.loc[tech, 'install'] * sum_out.loc[f'cap_{tech}_total'])
            cost_v2 = float(res.cost_investment.loc['monetary', f'region1::{tech}'])
            if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e-1):
                logger.error(
                    f'Cannot recreate {tech} install costs -- manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

        # Test if storage technology installation costs are consistent
        if len(storage_techs) > 1:
            cost_v1 = corrfac * (
                float(costs.loc['storage_energy', 'install'] * sum_out.loc['cap_storage_energy_total'])
                + float(costs.loc['storage_power', 'install'] * sum_out.loc['cap_storage_power_total'])
            )
            cost_v2 = float(res.cost_investment.loc['monetary', 'region1::storage_'])
            if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e-1):
                logger.error(
                    f'Cannot recreate storage install costs -- manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

    # Test if generation costs are consistent
    for tech in gen_techs + unmet_techs:
        cost_v1 = float(costs.loc[tech, 'generation'] * sum_out.loc[f'gen_{tech}_total'])
        cost_v2 = float(res.cost_var.loc['monetary', f'region1::{tech}'].sum())
        if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e-1):
            logger.error(
                f'Cannot recreate {tech} generation costs -- manual: {cost_v1}, model: {cost_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if total costs are consistent
    if model.run_mode == 'plan':
        cost_total_v2 = float(res.cost.loc['monetary'].sum())
        if not np.isclose(cost_total_v1, cost_total_v2, rtol=1e-6, atol=1e-1):
            logger.error(
                f'Cannot recreate system cost -- manual: {cost_total_v1}, model: {cost_total_v2}.'
            )
            passing = False

    # Test if emissions are consistent
    for tech in gen_techs:
        emission_v1 = float(costs.loc[tech, 'emissions'] * sum_out.loc[f'gen_{tech}_total'])
        emission_v2 = float(res.cost_var.loc['emissions', f'region1::{tech}'].sum())
        if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e5):
            logger.error(
                f'Cannot recreate {tech} emissions -- manual: {emission_v1}, model: {emission_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test that generation levels in each time step are nonnegative and add up to demand
    for tech in gen_techs:
        if not (ts_out.loc[:, f'gen_{tech}'] >= -0.0001).all():
            logger.error(f'Generation levels for {tech} sometimes negative:\n\n{ts_out}\n.')
            passing = False
    if not np.allclose(
        ts_out.filter(regex=f'gen_({"|".join([*techs, "storage"])}).*', axis=1).sum(axis=1),
        ts_out.filter(regex='demand.*', axis=1).sum(axis=1),
        rtol=1e-6,
        atol=1e-1
    ):
        logger.error('Generation does not add up to demand in some time steps.')
        passing = False

    # Test if storage levels are consistent
    gen_storage_np = ts_out['gen_storage'].to_numpy()
    level_storage_np = ts_out['level_storage'].to_numpy()
    initial_storage_calliope = float(
        inp.storage_initial.loc['region1::storage_'] * res.storage_cap.loc['region1::storage_']
    )
    if not np.isclose(level_storage_np[0], initial_storage_calliope, rtol=1e-6, atol=1e-1):
        logger.error(
            f'Cannot recreate initial storage level -- '
            f'manual: {level_storage_np[0]}, model: {float(inp.storage_initial)}.'
        )
        passing = False
    self_loss = float(inp.storage_loss)
    efficiency = float(inp.energy_eff.loc['region1::storage_'])
    storage_levels_v1 = level_storage_np[1:]
    storage_levels_v2 = (
        (1 - self_loss) * level_storage_np[:-1]  # Self charge loss
        - efficiency * np.clip(gen_storage_np[:-1], a_min=None, a_max=0.)  # Storage charge
        - (1 / efficiency) * np.clip(gen_storage_np[:-1], a_min=0., a_max=None)  # Storage discharge
    )
    if not np.allclose(storage_levels_v1, storage_levels_v2, rtol=1e-6, atol=1e-1):
        logger.error('Cannot recreate storage levels in some time steps.')
        passing = False

    # Check consistency between summary and time series outputs
    for col_name in ['gen_baseload', 'gen_peaking', 'gen_wind', 'gen_solar', 'demand']:
        total_sum = sum_out.loc[f'{col_name}_total', 'output']
        total_ts = ts_out[col_name].sum()
        if not np.isclose(total_sum, total_ts, rtol=1e-6, atol=1e-1):
            logger.error(
                f'Summary and time series output total do not match for {col_name} -- '
                f'summary outputs: {total_sum}, time series outputs: {total_ts}.'
            )
            passing = False

    return passing


def _has_consistent_outputs_6_region(model: calliope.Model) -> bool:  # pragma: no cover
    '''Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of SixRegionModel
    '''

    passing = True  # Changes to False if any outputs are found to be inconsistent
    cost_total_v1 = 0.

    # Get list of regions, tech-location pairs, and costs
    regions = list(model._model_run['locations'].keys())
    tech_regions = get_tech_regions(model=model)
    gen_regions = [i for i in tech_regions if i[0] not in ['transmission', 'storage', 'unmet']]
    trans_regions = [i for i in tech_regions if i[0] in ['transmission']]
    storage_regions = [i for i in tech_regions if i[0] in ['storage']]
    unmet_regions = [i for i in tech_regions if i[0] in ['unmet']]
    demand_regions = [('demand', i[1]) for i in unmet_regions]  # Demand and unmet have same regions
    costs = get_technology_info(model=model)
    assert set(gen_regions + trans_regions + storage_regions + unmet_regions) == set(tech_regions)

    sum_out = model.get_summary_outputs()
    ts_out = model.get_timeseries_outputs()
    inp = model.inputs
    res = model.results

    # Normalise install costs to same temporal scale as generation costs
    corrfac = model.num_timesteps / 8760

    if model.run_mode == 'plan':

        # Test if generation technology installation costs are consistent
        for tech, region in gen_regions:
            cost_v1 = corrfac * float(
                costs.loc[f'{tech}_{region}', 'install'] * sum_out.loc[f'cap_{tech}_{region}']
            )
            cost_v2 = float(res.cost_investment.loc['monetary', f'{region}::{tech}_{region}'])
            if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e0):
                logger.error(
                    f'Cannot recreate {tech} install costs in {region} -- '
                    f'manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

        # Test if transmission technology installation costs are consistent
        for tech, region, region_to in trans_regions:
            cost_v1 = corrfac * float(
                costs.loc[f'{tech}_{region}_{region_to}', 'install']
                * sum_out.loc[f'cap_transmission_{region}_{region_to}']
            )
            cost_v2 = 2 * float(
                res.cost_investment.loc[
                    'monetary', f'{region}::{tech}_{region}_{region_to}:{region_to}'
                ]
            )
            if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e0):
                logger.error(
                    f'Cannot recreate {tech} install costs from {region} to {region_to} -- '
                    f'manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

        # Test if storage technology installation costs are consistent
        for tech, region in storage_regions:
            cost_v1 = corrfac * (
                float(
                    costs.loc[f'storage_energy_{region}', 'install']
                    * sum_out.loc[f'cap_storage_energy_{region}']
                )
                + float(
                    costs.loc[f'storage_power_{region}', 'install']
                    * sum_out.loc[f'cap_storage_power_{region}']
                )
            )
            cost_v2 = float(res.cost_investment.loc['monetary', f'{region}::storage_{region}'])
            if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=1e0):
                logger.error(
                    f'Cannot recreate storage install costs in {region} -- '
                    f'manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

    # Test if generation costs are consistent
    for tech, region in gen_regions + unmet_regions:
        cost_v1 = float(
            costs.loc[f'{tech}_{region}', 'generation'] * sum_out.loc[f'gen_{tech}_{region}']
        )
        cost_v2 = float(res.cost_var.loc['monetary', f'{region}::{tech}_{region}'].sum())
        if not np.isclose(cost_v1, cost_v2, rtol=1e-6, atol=(1e2 if 'unmet' in tech else 1e0)):
            logger.error(
                f'Cannot recreate {tech} generation costs in {region} -- '
                f'manual: {cost_v1}, model: {cost_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if emissions are consistent
    emissions_total_v1 = 0.
    for tech, region in gen_regions:
        emissions_v1 = float(
            costs.loc[f'{tech}_{region}', 'emissions'] * sum_out.loc[f'gen_{tech}_{region}']
        )
        emissions_v2 = float(res.cost_var.loc['emissions', f'{region}::{tech}_{region}'].sum())
        if not np.isclose(emissions_v1, emissions_v2, rtol=1e-6, atol=1e5):
            logger.error(
                f'Cannot recreate {tech} emissions in {region} -- '
                f'manual: {emissions_v1}, model: {emissions_v2}.'
            )
            passing = False
        emissions_total_v1 += emissions_v1

    # Test if total costs are consistent
    if model.run_mode == 'plan':
        cost_total_v2 = float(res.cost.loc['monetary'].sum())
        if not np.isclose(cost_total_v1, cost_total_v2, rtol=1e-6, atol=1e1):
            logger.error(
                f'Cannot recreate system cost -- manual: {cost_total_v1}, model: {cost_total_v2}.'
            )
            passing = False

    # Test if total emissions are consistent
    emissions_total_v2 = float(res.cost_var.loc['emissions'].sum())
    if not np.isclose(emissions_v1, emissions_v2, rtol=1e-6, atol=1e5):
        logger.error(
            f'Cannot recreate total emissions -- '
            f'manual: {emissions_total_v1}, model: {emissions_total_v2}.'
        )
        passing = False

    # Test if costs and emissions match those in summary outputs
    if model.run_mode == 'plan':
        cost_total_v3 = sum_out.loc['cost_total', 'output']
        emissions_total_v3 = sum_out.loc['emissions_total', 'output']
        if not np.isclose(cost_total_v1, cost_total_v3, rtol=1e-6, atol=1e2):
            logger.error(
                f'Cannot recreate system cost -- '
                f'manual: {cost_total_v1}, summary_outputs: {cost_total_v3}.'
            )
            passing = False
        if not np.isclose(emissions_total_v1, emissions_total_v3, rtol=1e-6, atol=1e5):
            logger.error(
                f'Cannot recreate emissions -- '
                f'manual: {emissions_total_v1}, sum_outputs: {emissions_total_v3}.'
            )
            passing = False

    # Test that generation levels are all nonnegative
    for tech, region in gen_regions + unmet_regions:
        gen_levels = ts_out[f'gen_{tech}_{region}']
        if (gen_levels < -1e-3).any():
            logger.error(
                f'Generation levels for {tech} in {region} sometimes negative:\n\n{gen_levels}\n.'
            )
            passing = False

    # Test that generation levels add up to demand
    gen_levels_systemwide = ts_out.filter(regex='^gen_.*_region.$', axis=1).sum(axis=1)
    demand_levels_systemwide = ts_out.filter(regex='^demand_region.$', axis=1).sum(axis=1)
    if not np.allclose(gen_levels_systemwide, demand_levels_systemwide, rtol=1e-6, atol=1e0):
        logger.error('Generation does not add up to demand in some time steps.')
        passing = False

    # Test regional power balance: generation equals demand + transmission out of region
    for region in regions:
        generation_total_region = ts_out.filter(regex=f'gen_.*_{region}', axis=1).sum(axis=1)
        demand_total_region = ts_out.filter(regex=f'demand_{region}', axis=1).sum(axis=1)
        transmission_total_from_region = (
            ts_out.filter(regex=f'transmission_{region}_region.*', axis=1).sum(axis=1)
            - ts_out.filter(regex=f'transmission_region.*_{region}', axis=1).sum(axis=1)
        )
        if not np.allclose(
            generation_total_region,
            demand_total_region + transmission_total_from_region,
            rtol=1e-6,
            atol=1e-1
        ):
            balance_info = pd.DataFrame()
            balance_info['generation'] = generation_total_region
            balance_info['demand'] = demand_total_region
            balance_info['transmission_out'] = transmission_total_from_region
            logger.error(f'Power balance not satisfied in {region}:\n\n{balance_info}\n.')
            passing = False

    # Test if storage levels are consistent
    for tech, region in storage_regions:
        gen_storage_np = ts_out[f'gen_storage_{region}'].to_numpy()
        level_storage_np = ts_out[f'level_storage_{region}'].to_numpy()
        key = f'{region}::storage_{region}'
        initial_storage_calliope = float(inp.storage_initial.loc[key] * res.storage_cap.loc[key])
        if not np.isclose(level_storage_np[0], initial_storage_calliope, rtol=1e-6, atol=1e-1):
            logger.error(
                f'Cannot recreate initial storage level in {region} -- '
                f'manual: {level_storage_np[0]}, model: {float(inp.storage_initial)}.'
            )
            passing = False
        self_loss = float(inp.storage_loss.loc[key])
        efficiency = float(inp.energy_eff.loc[key])
        storage_levels_v1 = level_storage_np[1:]
        storage_levels_v2 = (
            (1 - self_loss) * level_storage_np[:-1]  # Self charge loss
            - efficiency * np.clip(gen_storage_np[:-1], a_min=None, a_max=0.)  # Charging
            - (1 / efficiency) * np.clip(gen_storage_np[:-1], a_min=0., a_max=None)  # Discharging
        )
        if not np.allclose(storage_levels_v1, storage_levels_v2, rtol=1e-6, atol=1e-1):
            # pass  # Deactivate for now -- fails because of Calliope implementation of operate mode
            logger.error(f'Cannot recreate storage levels in some time steps in region {region}.')
            passing = False

    # Check consistency between summary and time series outputs
    for tech, region in gen_regions + unmet_regions + demand_regions:
        col = f'demand_{region}' if tech == 'demand' else f'gen_{tech}_{region}'
        total_v1 = sum_out.loc[col, 'output']
        total_v2 = ts_out[col].sum()
        if not np.isclose(total_v1, total_v2, rtol=1e-6, atol=1e1):
            logger.error(
                f'Summary and time series output total do not match for {col} -- '
                f'summary outputs: {total_v1}, time series outputs: {total_v2}.'
            )
            passing = False

    return passing


def has_consistent_outputs(model: calliope.Model) -> bool:  # pragma: no cover
    '''Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of OneRegionModel or SixRegionModel
    '''

    if model.model_name == '1_region':
        return _has_consistent_outputs_1_region(model=model)
    elif model.model_name == '6_region':
        return _has_consistent_outputs_6_region(model=model)
