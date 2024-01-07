import logging
import numpy as np
import pandas as pd
import calliope
import psm.utils


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


def _has_consistent_outputs_1_region(model: calliope.Model) -> bool:
    '''Check if model outputs (costs, generation levels, emissions) are internally consistent.
    Log errors whenever they are not.

    Parameters:
    -----------
    model: instance of OneRegionModel
    '''

    passing = True  # Changes to False if any outputs are found to be inconsistent
    cost_total_v1 = 0

    costs = psm.utils.get_technology_info(model=model)
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
        inp.storage_initial.loc['region1::storage_'].fillna(0.)
        * res.storage_cap.loc['region1::storage_']
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


def _has_consistent_outputs_6_region(model: calliope.Model) -> bool:
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
    tech_regions = psm.utils.get_tech_regions(model=model)
    gen_regions = [i for i in tech_regions if i[0] not in ['transmission', 'storage', 'unmet']]
    trans_regions = [i for i in tech_regions if i[0] in ['transmission']]
    storage_regions = [i for i in tech_regions if i[0] in ['storage']]
    unmet_regions = [i for i in tech_regions if i[0] in ['unmet']]
    demand_regions = [('demand', i[1]) for i in unmet_regions]  # Demand and unmet have same regions
    costs = psm.utils.get_technology_info(model=model)
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
        initial_storage_calliope = float(
            inp.storage_initial.loc[key].fillna(0.) * res.storage_cap.loc[key]
        )
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
        if not np.allclose(storage_levels_v1, storage_levels_v2, rtol=1e-2, atol=1e-1):
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


def has_consistent_outputs(model: calliope.Model) -> bool:
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
