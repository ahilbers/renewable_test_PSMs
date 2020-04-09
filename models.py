"""The test case power system models."""


import os
import logging
import shutil
import pandas as pd
import calliope


# Emission intensities of technologies, in ton CO2 equivalent per GWh
EMISSION_INTENSITIES = {'baseload': 200,
                        'peaking': 400,
                        'wind': 0,
                        'unmet': 0}


def load_time_series_data(model_name):
    """Load demand and wind time series data for model.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'

    Returns:
    --------
    ts_data (pandas DataFrame) : time series data for use in model
    """

    ####
    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    # If 1_region model, take demand and wind from region 5
    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5']]
        ts_data.columns = ['demand', 'wind']
    ####

    return ts_data


def detect_missing_leap_days(ts_data):
    """Detect if a time series has missing leap days.

    Parameters:
    -----------
    ts_data (pandas DataFrame) : time series
    """

    feb28_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 2)
                                & (ts_data.index.day == 28)]
    feb29_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 2)
                                & (ts_data.index.day == 29)]
    mar01_index = ts_data.index[(ts_data.index.year % 4 == 0)
                                & (ts_data.index.month == 3)
                                & (ts_data.index.day == 1)]
    if len(feb29_index) < min((len(feb28_index), len(mar01_index))):
        return True

    return False


def get_scenario(run_mode, baseload_integer, baseload_ramping, allow_unmet):
    """Get the scenario name for different run settings.

    Parameters:
    -----------
    run_mode (str) : 'plan' or 'operate': whether to let the model
        determine the optimal capacities or work with prescribed ones
    baseload_integer (bool) : activate baseload integer capacity
        constraint (built in units of 3GW)
    baseload_ramping (bool) : enforce baseload ramping constraint
    allow_unmet (bool) : allow unmet demand in planning mode (always
        allowed in operate mode)

    Returns:
    --------
    scenario (str) : name of scenario to pass in Calliope model
    """

    scenario = run_mode
    if run_mode == 'plan' and not baseload_integer:
        scenario = scenario + ',continuous'
    if run_mode == 'plan' and baseload_integer:
        scenario = scenario + ',integer'
    if run_mode == 'plan' and allow_unmet:
        scenario = scenario + ',allow_unmet'
    if baseload_ramping:
        scenario = scenario + ',ramping'

    return scenario


def get_cap_override_dict(model_name, fixed_caps):
    """Create an override dictionary that can be used to set fixed
    fixed capacities in a Calliope model run.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    fixed_caps (pandas DataFrame or dict) : the fixed capacities.
        A DataFrame created via model.get_summary_outputs (at
        regional level) will work.

    Returns:
    --------
    o_dict (dict) : A dict that can be fed as override_dict into Calliope
        model in operate mode
    """

    o_dict = {}

    # Add baseload, peaking and wind capacities
    if model_name == '1_region':
        for tech, attribute in [('baseload', 'energy_cap_equals'),
                                ('peaking', 'energy_cap_equals'),
                                ('wind', 'resource_area_equals')]:
            idx = ('locations.region1.techs.{}.constraints.{}'.
                   format(tech, attribute))
            o_dict[idx] = fixed_caps['cap_{}_total'.format(tech)]

    # Add baseload, peaking, wind and transmission capacities
    if model_name == '6_region':
        for region in ['region{}'.format(i+1) for i in range(6)]:
            for tech, attribute in [('baseload', 'energy_cap_equals'),
                                    ('peaking', 'energy_cap_equals'),
                                    ('wind', 'resource_area_equals')]:
                try:
                    idx = ('locations.{}.techs.{}.constraints.{}'.
                           format(region, tech, attribute))
                    o_dict[idx] = \
                        fixed_caps['cap_{}_{}'.format(tech, region)]
                except KeyError:
                    pass
            for region_to in ['region{}'.format(i+1) for i in range(6)]:
                if (region, region_to) == ('region1', 'region5'):
                    tech = 'transmission_region1to5'
                else:
                    tech = 'transmission_other'
                idx = ('links.{},{}.techs.{}.constraints.energy_cap_equals'.
                       format(region, region_to, tech))
                try:
                    o_dict[idx] = fixed_caps['cap_transmission_{}_{}'.
                                             format(region, region_to)]
                except KeyError:
                    pass

    return o_dict


def calculate_carbon_emissions(generation_levels):
    """Calculate total carbon emissions.

    Parameters:
    -----------
    generation_levels (pandas DataFrame or dict) : generation levels
        for the 4 technologies (baseload, peaking, wind and unmet)
    """

    emissions_tot = \
        EMISSION_INTENSITIES['baseload'] * generation_levels['baseload'] + \
        EMISSION_INTENSITIES['peaking'] * generation_levels['peaking'] + \
        EMISSION_INTENSITIES['wind'] * generation_levels['wind'] + \
        EMISSION_INTENSITIES['unmet'] * generation_levels['unmet']

    return emissions_tot


class ModelBase(calliope.Model):
    """Instance of either 1-region or 6-region model."""

    def __init__(self, model_name, ts_data, run_mode,
                 baseload_integer=False, baseload_ramping=False,
                 allow_unmet=False, fixed_caps=None, run_id=0):
        """
        Create instance of either 1-region or 6-region model.

        Parameters:
        -----------
        model_name (str) : either '1_region' or '6_region'
        ts_data (pandas DataFrame) : time series with demand and wind data.
            It may also contain custom time step weights
        run_mode (str) : 'plan' or 'operate': whether to let the model
            determine the optimal capacities or work with prescribed ones
        baseload_integer (bool) : activate baseload integer capacity
            constraint (built in units of 3GW)
        baseload_ramping (bool) : enforce baseload ramping constraint
        allow_unmet (bool) : allow unmet demand in planning mode (always
            allowed in operate mode)
        fixed_caps (dict or Pandas DataFrame) : fixed capacities as override
        run_id (int) : can be changed if multiple models are run in parallel
        """

        if model_name not in ['1_region', '6_region']:
            raise ValueError('Invalid model name '
                             '(choose 1_region or 6_region)')

        self.model_name = model_name
        self.base_dir = os.path.join('models', model_name)
        self.num_timesteps = ts_data.shape[0]

        scenario = get_scenario(run_mode, baseload_integer,
                                baseload_ramping, allow_unmet)
        override_dict = (get_cap_override_dict(model_name, fixed_caps)
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
        ts_data.to_csv(os.path.join(self._base_dir_iter, 'demand_wind.csv'))
        super(ModelBase, self).__init__(os.path.join(self._base_dir_iter,
                                                     'model.yaml'),
                                        scenario=scenario,
                                        override_dict=override_dict)
        shutil.rmtree(self._base_dir_iter)

        # Adjust weights if these are included in ts_data
        if 'weight' in ts_data.columns:
            self.inputs.timestep_weights.values = \
                ts_data.loc[:, 'weight'].values

        logging.info('Time series inputs:\n%s', ts_data)
        logging.info('Override dict:\n%s', override_dict)
        if run_mode == 'operate' and fixed_caps is None:
            logging.warning('No fixed capacities passed into model call. '
                            'Will read fixed capacities from model.yaml')

    def _create_init_time_series(self, ts_data):
        """Create demand and wind time series data for Calliope model
        initialisation.
        """

        # Avoid changing ts_data outside function
        ts_data_used = ts_data.copy()

        if self.model_name == '1_region':
            expected_columns = {'demand', 'wind'}
        elif self.model_name == '6_region':
            expected_columns = {'demand_region2', 'demand_region4',
                                'demand_region5', 'wind_region2',
                                'wind_region5', 'wind_region6'}
        if not expected_columns.issubset(ts_data.columns):
            raise AttributeError('Input time series: incorrect columns')

        # Detect missing leap days -- reset index if so
        if detect_missing_leap_days(ts_data_used):
            logging.warning('Missing leap days detected in input time series.'
                            'Time series index reset to start in 2020.')
            ts_data_used.index = pd.date_range(start='2020-01-01',
                                               periods=self.num_timesteps,
                                               freq='h')

        # Demand must be negative for Calliope
        ts_data_used.loc[:, ts_data.columns.str.contains('demand')] = (
            -ts_data_used.loc[:, ts_data.columns.str.contains('demand')]
        )

        return ts_data_used


def _dev_test():
    ts_data = load_time_series_data(model_name='6_region')
    ts_data = ts_data.loc['2017-01']
    model = SixRegionModel(ts_data=ts_data, run_mode='operate')
    model.run()
    print(model.get_summary_outputs())


class OneRegionModel(ModelBase):
    """Instance of 1-region power system model."""

    def __init__(self, ts_data, run_mode,
                 baseload_integer=False, baseload_ramping=False,
                 allow_unmet=False, fixed_caps=None, run_id=0):
        """Initialize model from ModelBase parent."""
        super(OneRegionModel, self).__init__(
            '1_region', ts_data, run_mode,
            baseload_integer, baseload_ramping, allow_unmet, fixed_caps
        )

    def get_summary_outputs(self):
        """Create pandas DataFrame of a subset of model outputs.

        Parameters:
        -----------
        save_csv (bool) : save CSV of summary outputs
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
        outputs.loc['cap_unmet_total'] = (
            float(self.results.carrier_prod.loc[
                'region1::unmet::power'
            ].max())
        )    # Equal to peak unmet demand

        # Insert generation levels
        for tech in ['baseload', 'peaking', 'wind', 'unmet']:
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
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        return outputs


class SixRegionModel(ModelBase):
    """Instance of 6-region power system model."""

    def __init__(self, ts_data, run_mode,
                 baseload_integer=False, baseload_ramping=False,
                 allow_unmet=False, fixed_caps=None, run_id=0):
        """Initialize model from ModelBase parent."""
        super(SixRegionModel, self).__init__(
            '6_region', ts_data, run_mode,
            baseload_integer, baseload_ramping, allow_unmet, fixed_caps
        )

    def get_summary_outputs(self, at_regional_level=False):
        """Create a pandas DataFrame of a subset of relevant model outputs

        Parameters:
        -----------
        at_regional_level (bool) : give each model output at
            regional level, otherwise the model totals
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
                            '{}::{}'.format(region, tech)
                        ])
                    )
                except KeyError:
                    pass

            # Wind capacity
            for tech in ['wind']:
                try:
                    outputs.loc['cap_{}_{}'.format(tech, region)] = (
                        float(self.results.resource_area.loc[
                            '{}::{}'.format(region, tech)
                        ])
                    )
                except KeyError:
                    pass

            # Unmet capacity (peak unmet demand)
            for tech in ['unmet']:
                try:
                    outputs.loc['cap_{}_{}'.format(tech, region)] = (
                        float(self.results.carrier_prod.loc[
                            '{}::{}::power'.format(region, tech)
                        ].max())
                    )
                except KeyError:
                    pass

            # Transmission capacity
            for transmission_type in ['transmission_region1to5',
                                      'transmission_other']:
                for region_to in ['region{}'.format(i+1) for i in range(6)]:
                    # No double counting of links -- one way only
                    if int(region[-1]) < int(region_to[-1]):
                        try:
                            outputs.loc['cap_transmission_{}_{}'.format(
                                region, region_to
                            )] = float(self.results.energy_cap.loc[
                                '{}::{}:{}'.format(
                                    region, transmission_type, region_to
                                )
                            ])
                        except KeyError:
                            pass

            # Baseload, peaking, wind and unmet generation levels
            for tech in ['baseload', 'peaking', 'wind', 'unmet']:
                try:
                    outputs.loc['gen_{}_{}'.format(tech, region)] = (
                        corrfac * float(
                            (self.results.carrier_prod.loc[
                                '{}::{}::power'.format(region, tech)]
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

        # Insert total capacities. cap_unmet_total is the sum of peak
        # unmet demand levels across the regions, but not necessarily
        # the systemwide peak unmet demand
        for tech in ['baseload', 'peaking', 'wind', 'unmet',
                     'transmission']:
            outputs.loc['cap_{}_total'.format(tech)] = outputs.loc[
                outputs.index.str.contains('cap_{}'.format(tech))
            ].sum()

        # Insert total annualised generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'unmet']:
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
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        # Aggregate outputs across regions if desired
        if not at_regional_level:
            outputs = outputs.loc[[
                'cap_baseload_total', 'cap_peaking_total', 'cap_wind_total',
                'cap_unmet_total', 'cap_transmission_total', 'gen_baseload_total',
                'gen_peaking_total', 'gen_wind_total', 'gen_unmet_total',
                'demand_total', 'cost_total', 'emissions_total'
                ]]

        return outputs


if __name__ == '__main__':
    _dev_test()
