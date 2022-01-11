import logging
import re
import json
import pandas as pd
import calliope


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


def _get_technology_info(model: calliope.Model) -> pd.DataFrame:
    """Get technology install & generation costs and emissions from model config."""
    
    model_dict = model._model_run
    costs = pd.DataFrame(columns=['install', 'generation', 'emissions'], dtype='float')
    regions = list(model_dict['locations'].keys())

    # Add the technologies in each region
    for region in regions:
        region_dict = model_dict['locations'][region]

        # Add generation technologies
        techs = [i for i in region_dict['techs'].keys() if 'demand' not in i]
        for tech in techs:
            tech_costs_dict = region_dict['techs'][tech]['costs']
            is_variable_renewable = ('wind' in tech) or ('solar' in tech)
            install_cost_name = 'resource_area' if is_variable_renewable else 'energy_cap'
            costs.loc[tech, 'install'] = (
                0. if 'unmet' in tech else float(tech_costs_dict['monetary'][install_cost_name])
            )
            costs.loc[tech, 'generation'] = float(tech_costs_dict['monetary']['om_prod'])
            costs.loc[tech, 'emissions'] = float(tech_costs_dict['emissions']['om_prod'])

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


def _has_consistent_outputs_1_region(model: calliope.Model) -> bool:
    """Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of OneRegionModel
    """

    passing = True  # Changes to False if any outputs are found to be inconsistent
    cost_total_v1 = 0

    costs = _get_technology_info(model=model)
    techs = list(costs.index)

    out = model.get_summary_outputs()
    res = model.results

    # Normalise install costs to same temporal scale as generation costs
    corrfac = model.num_timesteps / 8760

    # Test if generation technology installation costs are consistent
    if model.run_mode == 'plan':
        for tech in techs:
            if tech == 'unmet':
                continue  # Unmet demand doesn't have meaningful install cost
            cost_v1 = corrfac * float(costs.loc[tech, 'install'] * out.loc[f'cap_{tech}_total'])
            cost_v2 = float(res.cost_investment.loc['monetary', f'region1::{tech}'])
            if abs(cost_v1 - cost_v2) > 0.1:
                logger.error(
                    f'Cannot recreate {tech} install costs -- manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

    # Test if generation costs are consistent
    for tech in techs:
        cost_v1 = float(costs.loc[tech, 'generation'] * out.loc[f'gen_{tech}_total'])
        cost_v2 = float(res.cost_var.loc['monetary', f'region1::{tech}'].sum())
        if abs(cost_v1 - cost_v2) > 0.1:
            logger.error(
                f'Cannot recreate {tech} generation costs -- manual: {cost_v1}, model: {cost_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if total costs are consistent
    if model.run_mode == 'plan':
        cost_total_v2 = float(res.cost.loc['monetary'].sum())
        if abs(cost_total_v1 - cost_total_v2) > 0.1:
            logger.error(
                f'Cannot recreate system cost -- manual: {cost_total_v1}, model: {cost_total_v2}.'
            )
            passing = False

    # Test if emissions are consistent
    for tech in techs:
        emission_v1 = float(costs.loc[tech, 'emissions'] * out.loc[f'gen_{tech}_total'])
        emission_v2 = float(res.cost_var.loc['emissions', f'region1::{tech}'].sum())
        if abs(cost_v1 - cost_v2) > 0.1:
            logger.error(
                f'Cannot recreate {tech} emissions -- manual: {emission_v1}, model: {emission_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if supply matches demand
    generation_total = float(out.filter(regex='gen_.*_total', axis=0).sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 0.1:
        logger.error(
            f'Supply-demand mismatch -- generation: {generation_total}, demand: {demand_total}.'
        )
        passing = False

    return passing


def _has_consistent_outputs_6_region(model: calliope.Model) -> bool:
    """Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of SixRegionModel
    """

    passing = True  # Changes to False if any outputs are found to be inconsistent
    cost_total_v1 = 0

    costs = _get_technology_info(model=model)

    out = model.get_summary_outputs()
    res = model.results

    # Normalise install costs to same temporal scale as generation costs
    corrfac = model.num_timesteps / 8760

    # Get list of tech-location pairs
    tech_locations = [i.split('_') for i in costs.index]
    generation_tech_locations = [i for i in tech_locations if i[0] != 'transmission']
    transmission_tech_locations = [i for i in tech_locations if i[0] == 'transmission']

    # Test if generation technology installation costs are consistent
    if model.run_mode == 'plan':
        for tech, region in generation_tech_locations:
            if tech == 'unmet':
                continue  # Unmet demand doesn't have meaningful install cost
            cost_v1 = corrfac * float(
                costs.loc[f'{tech}_{region}', 'install'] * out.loc[f'cap_{tech}_{region}']
            )
            cost_v2 = float(res.cost_investment.loc['monetary', f'{region}::{tech}_{region}'])
            if abs(cost_v1 - cost_v2) > 0.1:
                logger.error(
                    f'Cannot recreate {tech} install costs in {region} -- '
                    f'manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

    # Test if transmission technology installation costs are consistent
    if model.run_mode == 'plan':
        for tech, region, region_to in transmission_tech_locations:
            cost_v1 = corrfac * float(
                costs.loc[f'{tech}_{region}_{region_to}', 'install'] 
                * out.loc[f'cap_transmission_{region}_{region_to}']
            )
            cost_v2 = 2 * float(
                res.cost_investment.loc[
                    'monetary', f'{region}::{tech}_{region}_{region_to}:{region_to}'
                ]
            )
            if abs(cost_v1 - cost_v2) > 0.1:
                logger.error(
                    f'Cannot recreate {tech} install costs from {region} to {region_to} -- '
                    f'manual: {cost_v1}, model: {cost_v2}.'
                )
                passing = False
            cost_total_v1 += cost_v1

    # Test if generation costs are consistent
    for tech, region in generation_tech_locations:
        cost_v1 = float(
            costs.loc[f'{tech}_{region}', 'generation'] * out.loc[f'gen_{tech}_{region}']
        )
        cost_v2 = float(res.cost_var.loc['monetary', f'{region}::{tech}_{region}'].sum())
        if abs(cost_v1 - cost_v2) > 0.1:
            logger.error(
                f'Cannot recreate {tech} generation costs in {region} -- '
                f'manual: {cost_v1}, model: {cost_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if total costs are consistent
    if model.run_mode == 'plan':
        cost_total_v2 = float(res.cost.loc['monetary'].sum())
        if abs(cost_total_v1 - cost_total_v2) > 0.1:
            logger.error(
                f'Cannot recreate system cost -- manual: {cost_total_v1}, model: {cost_total_v2}.'
            )
            passing = False

    # Test if emissions are consistent
    for tech, region in generation_tech_locations:
        emission_v1 = float(
            costs.loc[f'{tech}_{region}', 'emissions'] * out.loc[f'gen_{tech}_{region}']
        )
        emission_v2 = float(res.cost_var.loc['emissions', f'{region}::{tech}_{region}'].sum())
        if abs(cost_v1 - cost_v2) > 0.1:
            logger.error(
                f'Cannot recreate {tech} emissions in {region} -- '
                f'manual: {emission_v1}, model: {emission_v2}.'
            )
            passing = False
        cost_total_v1 += cost_v1

    # Test if supply matches demand
    generation_total = float(out.filter(regex='gen_.*_region.*', axis=0).sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 0.1:
        logger.error(
            f'Supply-demand mismatch -- generation: {generation_total}, demand: {demand_total}.'
        )
        passing = False

    return passing


def has_consistent_outputs(model: calliope.Model) -> bool:
    """Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of OneRegionModel or SixRegionModel
    """
    if model.model_name == '1_region':
        return _has_consistent_outputs_1_region(model=model)
    elif model.model_name == '6_region':
        return _has_consistent_outputs_6_region(model=model)
