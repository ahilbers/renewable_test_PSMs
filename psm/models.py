"""The test case power system models."""


import os
import logging
import typing
import json
import pandas as pd
import calliope
from psm import utils


logger = logging.getLogger(name=__package__)  # Logger with name 'psm', can be customised elsewhere


class ModelBase(calliope.Model):
    """Base power system model class."""

    def __init__(
        self, 
        model_name: str, 
        ts_data: pd.DataFrame, 
        run_mode: str,
        baseload_integer: bool = False, 
        baseload_ramping: bool = False,
        allow_unmet: bool = False, 
        fixed_caps: dict = None, 
        extra_override: str = None,
        run_id=0
    ):
        """
        Create base power system model instance.

        Parameters:
        -----------
        model_name: '1_region' or '6_region'
        ts_data: time series data, can contain custom time step weights
        run_mode: 'plan' or 'operate': optimise capacities or work with prescribed ones
        baseload_integer: enforce discrete baseload capacity constraint (built in discrete units)
        baseload_ramping: enforce baseload ramping constraint
        allow_unmet: allow unmet demand in 'plan' mode (should always be allowed in 'operate' mode)
        fixed_caps: fixed capacities as override
        extra_override (str) : name of additional override, should be defined in  `.../model.yaml`
        run_id (int) : unique model ID, useful if multiple models are run in parallel
        """

        if model_name not in ['1_region', '6_region']:
            raise ValueError(f'Invalid model name {model_name} (choose `1_region` or `6_region`).')

        self.model_name = model_name
        self.run_id = run_id
        self.base_dir = os.path.join('models', model_name)
        self.num_timesteps = ts_data.shape[0]

        # Create scenarios and overrides
        scenario = utils.get_scenario(run_mode, baseload_integer, baseload_ramping, allow_unmet)
        if extra_override is not None:
            scenario = f'{scenario},{extra_override}'
        if fixed_caps is not None:
            override_dict = utils.get_cap_override_dict(model_name, fixed_caps)
        else:
            override_dict = None

        ts_data = self._create_init_time_series(ts_data)
        super(ModelBase, self).__init__(
            config=os.path.join(self.base_dir, 'model.yaml'),
            timeseries_dataframes={'ts_data': ts_data},
            scenario=scenario,
            override_dict=override_dict
        )

        # Adjust time step weights
        if 'weight' in ts_data.columns:
            self.inputs.timestep_weights.values = ts_data.loc[:, 'weight'].values

        logger.debug(f'Model name: {self.model_name}, base directory: {self.base_dir}.')
        logger.debug(f'Override dict:\n{json.dumps(override_dict, indent=4)}')
        logger.debug(f'Time series inputs:\n{ts_data}\n')

        # Some checks when running in operate mode
        if run_mode == 'operate':
            if fixed_caps is None:
                logger.info(
                    f'No fixed capacities passed into model call. '
                    f'Reading fixed capacities from {model_name}/model.yaml'
                )
            if not allow_unmet:
                raise ValueError('Must allow unmet demand when running in operate mode.')


    def _create_init_time_series(self, ts_data: pd.DataFrame) -> pd.DataFrame:
        """Create time series data for model initialisation.
        
        Parameters:
        -----------
        ts_data: time series data loaded from CSV

        Returns:
        --------
        ts_data_used: same data, but data engineered for a Calliope model
        """

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
            logger.warning('Missing leap days detected. Time series index reset to start in 2020.')
            ts_data_used.index = pd.date_range(
                start='2020-01-01', periods=self.num_timesteps, freq='h'
            )

        # Demand must be negative for Calliope
        demand_columns = ts_data.columns.str.contains('demand')
        ts_data_used.loc[:, demand_columns] = -ts_data_used.loc[:, demand_columns]

        return ts_data_used


class OneRegionModel(ModelBase):
    """Instance of 1-region power system model."""

    def __init__(
        self,
        ts_data: pd.DataFrame, 
        run_mode: str,
        baseload_integer: bool = False, 
        baseload_ramping: bool = False,
        allow_unmet: bool = False, 
        fixed_caps: dict = None, 
        extra_override: str = None,
        run_id=0
    ):
        """Initialize model from ModelBase parent. See parent class for detailed docstring."""
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

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        outputs = pd.DataFrame(columns=['output'])  # Output DataFrame to be populated

        # Insert installed capacities
        outputs.loc['cap_baseload_total'] = float(self.results.energy_cap.loc['region1::baseload'])
        outputs.loc['cap_peaking_total'] = float(self.results.energy_cap.loc['region1::peaking'])
        outputs.loc['cap_wind_total'] = float(self.results.resource_area.loc['region1::wind'])
        outputs.loc['cap_solar_total'] = float(self.results.resource_area.loc['region1::solar'])
        outputs.loc['peak_unmet_total'] = (
            float(self.results.carrier_prod.loc['region1::unmet::power'].max())
        )  # Equal to peak unmet demand

        # Insert generation levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            outputs.loc[f'gen_{tech}_total'] = float(
                (
                    self.results.carrier_prod.loc[f'region1::{tech}::power']
                    * self.inputs.timestep_weights
                ).sum()
            )

        # Insert demand levels
        outputs.loc['demand_total'] = -float(
            (
                self.results.carrier_con.loc['region1::demand_power::power']
                * self.inputs.timestep_weights
            ).sum()
        )

        # Insert total system cost and carbon emissions
        outputs.loc['cost_total'] = float(self.results.cost.loc['monetary'].sum())
        outputs.loc['emissions_total'] = float(self.results.cost.loc['emissions'].sum())

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs


class SixRegionModel(ModelBase):
    """Instance of 6-region power system model."""

    def __init__(
        self, 
        ts_data: pd.DataFrame, 
        run_mode: str,
        baseload_integer: bool = False, 
        baseload_ramping: bool = False,
        allow_unmet: bool = False, 
        fixed_caps: dict = None, 
        extra_override: str = None,
        run_id=0
    ):
        """Initialize model from ModelBase parent. See parent class for detailed docstring."""
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

        if not hasattr(self, 'results'):
            raise AttributeError('Model outputs not yet calculated: call `.run()` first.')

        outputs = pd.DataFrame(columns=['output'])  # Output DataFrame to be populated

        # Insert model outputs at regional level. Loop over all techs and regions -- if a tech
        # doesn't appear in a region, then ignore it -- done by the 'pass' in case of KeyError
        for region in [f'region{i+1}' for i in range(6)]:

            # Baseload and peaking capacity
            for tech in ['baseload', 'peaking']:
                try:
                    outputs.loc[f'cap_{tech}_{region}'] = float(
                        self.results.energy_cap.loc[f'{region}::{tech}_{region}']
                    )
                except KeyError:
                    pass  # If this tech doesn't appear in this region, ignore it

            # Wind and solar capacity
            for tech in ['wind', 'solar']:
                try:
                    outputs.loc[f'cap_{tech}_{region}'] = float(
                        self.results.resource_area.loc[f'{region}::{tech}_{region}']
                    )
                except KeyError:
                    pass  # If this tech doesn't appear in this region, ignore it

            # Peak unmet demand
            for tech in ['unmet']:
                try:
                    outputs.loc[f'peak_unmet_{region}'] = float(
                        self.results.carrier_prod.loc[f'{region}::{tech}_{region}::power'].max()
                    )
                except KeyError:
                    pass  # If this tech doesn't appear in this region, ignore it

            # Transmission capacity
            for tech in ['transmission']:
                for region_to in [f'region{i+1}' for i in range(6)]:
                    # No double counting of links -- one way only
                    if int(region[-1]) < int(region_to[-1]):
                        try:
                            outputs.loc[f'cap_transmission_{region}_{region_to}'] = float(
                                self.results.energy_cap
                                .loc[f'{region}::{tech}_{region}_{region_to}:{region_to}']
                            )
                        except KeyError:
                            pass  # If this tech doesn't appear in this region, ignore it

            # Baseload, peaking, wind, solar and unmet generation levels
            for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
                try:
                    outputs.loc[f'gen_{tech}_{region}'] = float(
                        (
                            self.results.carrier_prod.loc[f'{region}::{tech}_{region}::power'] 
                            * self.inputs.timestep_weights
                        ).sum()
                    )
                except KeyError:
                    pass  # If this tech doesn't appear in this region, ignore it

            # Demand levels
            try:
                outputs.loc['demand_{}'.format(region)] = -float(
                    (
                        self.results.carrier_con.loc[f'{region}::demand_power::power']
                        * self.inputs.timestep_weights
                    ).sum()
                )
            except KeyError:
                pass  # If this tech doesn't appear in this region, ignore it

        # Insert total capacities
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'transmission']:
            outputs.loc[f'cap_{tech}_total'] = (
                outputs.loc[outputs.index.str.contains(f'cap_{tech}')].sum()
            )

        outputs.loc['peak_unmet_total'] = (
            outputs.loc[outputs.index.str.contains('peak_unmet')].sum()
        )

        # Insert total peak unmet demand -- not necessarily equal to peak_unmet_total. Total unmet 
        # capacity sums peak unmet demand across regions, this is systemwide peak unmet demand
        outputs.loc['peak_unmet_systemwide'] = float(
            self.results.carrier_prod
            .loc[self.results.carrier_prod.loc_tech_carriers_prod.str.contains('unmet')]
            .sum(axis=0)
            .max()
        )

        # Insert total generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'solar', 'unmet']:
            outputs.loc[f'gen_{tech}_total'] = (
                outputs.loc[outputs.index.str.contains(f'gen_{tech}')].sum()
            )

        # Insert total demand levels
        outputs.loc['demand_total'] = (outputs.loc[outputs.index.str.contains('demand')].sum())

        # Insert total system cost and carbon emissions
        outputs.loc['cost_total'] = float(self.results.cost.loc['monetary'].sum())
        outputs.loc['emissions_total'] = float(self.results.cost.loc['emissions'].sum())

        if as_dict:
            outputs = outputs['output'].to_dict()

        return outputs


if __name__ == '__main__':
    raise NotImplementedError()
