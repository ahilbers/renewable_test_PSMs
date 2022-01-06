"""The test case power system models."""


import os
import logging
import typing
import shutil
import pandas as pd
import calliope
from psm import utils


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


# Emission intensities of technologies, in ton CO2 equivalent per GWh
# TODO: Sort this out
EMISSION_INTENSITIES = {'baseload': 200, 'peaking': 400, 'wind': 0, 'solar': 0, 'unmet': 0}


def calculate_carbon_emissions(generation_levels):
    """Calculate total carbon emissions.

    Parameters:
    -----------
    generation_levels (pandas DataFrame or dict) : generation levels
        for the technologies (baseload, peaking, wind, solar and unmet)
    """

    emissions_tot = (
        EMISSION_INTENSITIES['baseload'] * generation_levels['baseload']
        + EMISSION_INTENSITIES['peaking'] * generation_levels['peaking']
        + EMISSION_INTENSITIES['wind'] * generation_levels['wind']
        + EMISSION_INTENSITIES['solar'] * generation_levels['solar']
        + EMISSION_INTENSITIES['unmet'] * generation_levels['unmet']
    )

    return emissions_tot


class ModelBase(calliope.Model):
    """Instance of either 1-region or 6-region model."""

    def __init__(self, model_name, ts_data, run_mode,
                 baseload_integer=False, baseload_ramping=False,
                 allow_unmet=False, fixed_caps=None, extra_override=None,
                 run_id=0):
        """
        Create instance of either 1-region or 6-region model.

        Parameters:
        -----------
        model_name (str) : either '1_region' or '6_region'
        ts_data (pandas DataFrame) : time series with time series data.
            It may also contain custom time step weights
        run_mode (str) : 'plan' or 'operate': whether to let the model
            determine the optimal capacities or work with prescribed ones
        baseload_integer (bool) : activate baseload integer capacity
            constraint (built in units of 3GW)
        baseload_ramping (bool) : enforce baseload ramping constraint
        allow_unmet (bool) : allow unmet demand in 'plan' mode (should always be allowed in 
            'operate' mode)
        fixed_caps (dict or Pandas DataFrame) : fixed capacities as override
        extra_override (str) : name of additional override, to customise
            model. The override should be defined in the relevant model.yaml
        run_id (int) : can be changed if multiple models are run in parallel
        """

        if model_name not in ['1_region', '6_region']:
            raise ValueError(f'Invalid model name {model_name} (choose `1_region` or `6_region`).')

        self.model_name = model_name
        self.base_dir = os.path.join('models', model_name)
        self.num_timesteps = ts_data.shape[0]

        # Create scenarios and overrides
        scenario = utils.get_scenario(run_mode, baseload_integer,
                                baseload_ramping, allow_unmet)
        if extra_override is not None:
            scenario = ','.join((scenario, extra_override))
        override_dict = (utils.get_cap_override_dict(model_name, fixed_caps)
                         if fixed_caps is not None else None)

        # Calliope requires a CSV file of the time series data to be present
        # at time of initialisation. This creates a new directory with the
        # model files and data for the model, then deletes it once the model
        # exists in Python
        self._base_dir_iter = self.base_dir + '_' + str(run_id)
        if os.path.exists(self._base_dir_iter):
            shutil.rmtree(self._base_dir_iter)
        shutil.copytree(self.base_dir, self._base_dir_iter)
        ts_data = self._create_init_time_series(ts_data)
        ts_data.to_csv(os.path.join(self._base_dir_iter, 'demand_wind_solar.csv'))
        super(ModelBase, self).__init__(os.path.join(self._base_dir_iter,
                                                     'model.yaml'),
                                        scenario=scenario,
                                        override_dict=override_dict)
        shutil.rmtree(self._base_dir_iter)

        # Adjust weights if these are included in ts_data
        if 'weight' in ts_data.columns:
            self.inputs.timestep_weights.values = ts_data.loc[:, 'weight'].values

        logger.debug('Time series inputs:\n%s', ts_data)
        logger.debug('Override dict:\n%s', override_dict)

        # Some checks when running in operate mode
        if run_mode == 'operate':
            # Throw warning if fixed capacities are not provided here
            if fixed_caps is None:
                logger.info(
                    f'No fixed capacities passed into model call. '
                    f'Reading fixed capacities from {model_name}/model.yaml'
                )
            # Raise error is not allowing unmet demand
            if not allow_unmet:
                raise ValueError('Must allow unmet demand when running in operate mode.')


    def _create_init_time_series(self, ts_data):
        """Create time series data for model initialisation."""

        # Avoid changing ts_data outside function
        ts_data_used = ts_data.copy()

        if self.model_name == '1_region':
            expected_columns = ['demand', 'wind', 'solar']
        elif self.model_name == '6_region':
            expected_columns = [
                'demand_region2', 'demand_region4', 'demand_region5', 
                'wind_region2', 'wind_region5', 'wind_region6',
                'solar_region2', 'solar_region5', 'solar_region6'
            ]
        if not set(expected_columns).issubset(set(ts_data.columns)):
            raise AttributeError(
                f'Incorrect columns in input time series. '
                f'Expected {expected_columns}, got {list(ts_data.columns)}.'
            )

        # Detect missing leap days -- reset index if so
        if utils.has_missing_leap_days(ts_data_used):
            logger.warning('Missing leap days detected in input time series.'
                            'Time series index reset to start in 2020.')
            ts_data_used.index = pd.date_range(start='2020-01-01',
                                               periods=self.num_timesteps,
                                               freq='h')

        # Demand must be negative for Calliope
        ts_data_used.loc[:, ts_data.columns.str.contains('demand')] = (
            -ts_data_used.loc[:, ts_data.columns.str.contains('demand')]
        )

        return ts_data_used


class OneRegionModel(ModelBase):
    """Instance of 1-region power system model."""

    def __init__(self, ts_data, run_mode, baseload_integer=False,
                 baseload_ramping=False, allow_unmet=False,
                 fixed_caps=None, extra_override=None, run_id=0):
        """Initialize model from ModelBase parent."""
        super(OneRegionModel, self).__init__(
            model_name='1_region',
            ts_data=ts_data,
            run_mode=run_mode,
            baseload_integer=baseload_integer,
            baseload_ramping=baseload_ramping,
            allow_unmet=allow_unmet,
            fixed_caps=fixed_caps,
            extra_override=extra_override,
            run_id=run_id
        )

    def get_summary_outputs(self, as_dict: bool = False) -> typing.Union[pd.DataFrame, dict]:
        """Return selection of key model outputs.
        
        Parameters:
        -----------
        as_dict: return dictionary instead of DataFrame
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        outputs = pd.DataFrame(columns=['output'])    # Output DataFrame
        corrfac = (8760/self.num_timesteps)    # For annualisation

        # Insert installed capacities
        outputs.loc['cap_baseload_total'] = (
            float(self.results.energy_cap.loc['region1::baseload'])
        )
        outputs.loc['cap_peaking_total'] = (
            float(self.results.energy_cap.loc['region1::peaking'])
        )
        outputs.loc['cap_wind_total'] = (
            float(self.results.resource_area.loc['region1::wind'])
        )
        outputs.loc['cap_solar_total'] = (
            float(self.results.resource_area.loc['region1::solar'])
        )
        outputs.loc['peak_unmet_total'] = (
            float(self.results.carrier_prod.loc[
                'region1::unmet::power'
            ].max())
        )    # Equal to peak unmet demand

        # Insert generation levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            outputs.loc['gen_{}_total'.format(tech)] = corrfac * float(
                (self.results.carrier_prod.loc['region1::{}::power'.format(tech)]
                 * self.inputs.timestep_weights).sum()
            )

        # Insert annualised demand levels
        outputs.loc['demand_total'] = -corrfac * float(
            (self.results.carrier_con.loc['region1::demand_power::power']
             * self.inputs.timestep_weights).sum()
        )

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'solar': outputs.loc['gen_solar_total'],
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs


class SixRegionModel(ModelBase):
    """Instance of 6-region power system model."""

    def __init__(self, ts_data, run_mode, baseload_integer=False,
                 baseload_ramping=False, allow_unmet=False,
                 fixed_caps=None, extra_override=None, run_id=0):
        """Initialize model from ModelBase parent."""
        super(SixRegionModel, self).__init__(
            model_name='6_region',
            ts_data=ts_data,
            run_mode=run_mode,
            baseload_integer=baseload_integer,
            baseload_ramping=baseload_ramping,
            allow_unmet=allow_unmet,
            fixed_caps=fixed_caps,
            extra_override=extra_override,
            run_id=run_id
        )

    def get_summary_outputs(self, as_dict: bool = False) -> typing.Union[pd.DataFrame, dict]:
        """Return selection of key model outputs.
        
        Parameters:
        -----------
        as_dict: return dictionary instead of DataFrame
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        outputs = pd.DataFrame(columns=['output'])    # Output DataFrame
        corrfac = (8760/self.num_timesteps)    # For annualisation

        # Insert model outputs at regional level
        for region in ['region{}'.format(i+1) for i in range(6)]:

            # Baseload and peaking capacity
            for tech in ['baseload', 'peaking']:
                try:
                    outputs.loc['cap_{}_{}'.format(tech, region)] = (
                        float(self.results.energy_cap.loc[
                            '{}::{}_{}'.format(region, tech, region)
                        ])
                    )
                except KeyError:
                    pass

            # Wind and solar capacity
            for tech in ['wind', 'solar']:
                try:
                    outputs.loc['cap_{}_{}'.format(tech, region)] = (
                        float(self.results.resource_area.loc[
                            '{}::{}_{}'.format(region, tech, region)
                        ])
                    )
                except KeyError:
                    pass

            # Peak unmet demand
            for tech in ['unmet']:
                try:
                    outputs.loc['peak_unmet_{}'.format(region)] = (
                        float(self.results.carrier_prod.loc[
                            '{}::{}_{}::power'.format(region, tech, region)
                        ].max())
                    )
                except KeyError:
                    pass

            # Transmission capacity
            for tech in ['transmission']:
                for region_to in ['region{}'.format(i+1) for i in range(6)]:
                    # No double counting of links -- one way only
                    if int(region[-1]) < int(region_to[-1]):
                        try:
                            outputs.loc['cap_transmission_{}_{}'.format(
                                region, region_to
                            )] = float(self.results.energy_cap.loc[
                                '{}::{}_{}_{}:{}'.format(region,
                                                         tech,
                                                         region,
                                                         region_to,
                                                         region_to)
                            ])
                        except KeyError:
                            pass

            # Baseload, peaking, wind, solar and unmet generation levels
            for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
                try:
                    outputs.loc['gen_{}_{}'.format(tech, region)] = (
                        corrfac * float(
                            (self.results.carrier_prod.loc[
                                '{}::{}_{}::power'.format(region,
                                                          tech,
                                                          region)]
                             *self.inputs.timestep_weights).sum()
                        )
                    )
                except KeyError:
                    pass

            # Demand levels
            try:
                outputs.loc['demand_{}'.format(region)] = -corrfac * float(
                    (self.results.carrier_con.loc[
                        '{}::demand_power::power'.format(region)]
                     * self.inputs.timestep_weights).sum()
                )
            except KeyError:
                pass

        # Insert total capacities
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'transmission']:
            outputs.loc['cap_{}_total'.format(tech)] = outputs.loc[
                outputs.index.str.contains('cap_{}'.format(tech))
            ].sum()

        outputs.loc['peak_unmet_total'] = outputs.loc[
            outputs.index.str.contains('peak_unmet')
        ].sum()

        # Insert total peak unmet demand -- not necessarily equal to
        # peak_unmet_total. Total unmet capacity sums peak unmet demand
        # across regions, whereas this is the systemwide peak unmet demand
        outputs.loc['peak_unmet_systemwide'] = float(self.results.carrier_prod.loc[
            self.results.carrier_prod.loc_tech_carriers_prod.str.contains(
                'unmet')].sum(axis=0).max())

        # Insert total annualised generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            outputs.loc['gen_{}_total'.format(tech)] = outputs.loc[
                outputs.index.str.contains('gen_{}'.format(tech))
            ].sum()

        # Insert total annualised demand levels
        outputs.loc['demand_total'] = (
            outputs.loc[outputs.index.str.contains('demand')].sum()
        )

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'solar': outputs.loc['gen_solar_total'],
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs


if __name__ == '__main__':
    raise NotImplementedError()
