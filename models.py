"""The test case power system models."""


import os
import numpy as np
import pandas as pd
import calliope
import functions
import tests


def calculate_carbon_emissions(emission_levels, generation_levels):
    """Calculate total carbon emissions.

    Parameters:
    -----------
    generation_levels (pandas DataFrame or dict) : emission levels
        for the 4 technologies (baseload, peaking, wind and unmet)
    generation_levels (pandas DataFrame or dict) : generation levels
        for the 4 technologies (baseload, peaking, wind and unmet)
    """

    emissions_tot = \
        emission_levels['baseload'] * generation_levels['baseload'] + \
        emission_levels['peaking'] * generation_levels['peaking'] + \
        emission_levels['wind'] * generation_levels['wind'] + \
        emission_levels['unmet'] * generation_levels['unmet']

    return emissions_tot


class SixRegionModel(calliope.Model):
    """Instance of 6-region power system model."""

    def __init__(self, model_type, ts_data, preserve_index=False):
        """Create instance of 6-region model in Calliope.

        Parameters:
        -----------
        model_type (str) : either 'LP' (linear program) or 'MILP' (mixed
            integer linear program), depending on the baseload constraints.
        ts_data (pandas DataFrame) : time series with demand and wind data
        preserve_index (bool) : whether to use index from original time
            series. If False, the index is reset to hours starting in 1980.
            If True, the original time series index is used. This may lead
            to problems with leap days.
        """

        assert model_type in ['LP', 'MILP'], \
            'model_type must be either LP or MILP'

        self._base_dir = 'models/6_region'
        self.num_timesteps = ts_data.shape[0]

        # Emission intensity, ton CO2 equivalent per GWh
        self.emission_levels = {'baseload': 200,
                                'peaking': 400,
                                'wind': 0,
                                'unmet': 0}

        # Calliope requires a CSV file of time series data to be present
        # at time of model initialisation. This code creates a CSV with the
        # right format, initialises the model, then deletes the CSV
        ts_data = self._create_init_time_series(ts_data, preserve_index)
        ts_data_init_path = os.path.join(self._base_dir, 'demand_wind.csv')
        ts_data.to_csv(ts_data_init_path)
        super(SixRegionModel, self).__init__(os.path.join(self._base_dir,
                                                          'model.yaml'),
                                             scenario=model_type)
        os.remove(ts_data_init_path)

    def _create_init_time_series(self, ts_data, preserve_index=False):
        """Create demand and wind time series data for Calliope model
        initialisation.

        Parameters:
        -----------
        ts_data (pandas DataFrame) : demand and wind time series data
            to use in model
        preserve_index (bool) : if False, resets the time series index
            to a default starting at 1980-01-01 with hourly resolution. This
            avoids problems with leap days.
        """

        # Test if correct columns are present
        if set(ts_data.columns) != set(['demand_region2', 'demand_region4',
                                        'demand_region5', 'wind_region2',
                                        'wind_region5', 'wind_region6']):
            raise AttributeError('Input time series: incorrect columns')

        # Reset index if applicable
        if not preserve_index:
            ts_data.index = pd.date_range(start='1980-01-01',
                                          periods=self.num_timesteps,
                                          freq='h')
        else:
            print('WARNING: you have selected to maintain the original '
                  'time series index. Be careful with leap days')

        # Create CSV for model intialisation
        ts_data.loc[:, 'demand_region2'] = -ts_data.loc[:, 'demand_region2']
        ts_data.loc[:, 'demand_region4'] = -ts_data.loc[:, 'demand_region4']
        ts_data.loc[:, 'demand_region5'] = -ts_data.loc[:, 'demand_region5']

        return ts_data

    def get_summary_outputs(self, at_regional_level=False, save_csv=False):
        """Create a pandas DataFrame of a subset of relevant model outputs

        Parameters:
        -----------
        at_regional_level (bool) : if True, gives each model output at
            regional level, otherwise the model totals
        save_csv (bool) : whether to save summary outputs as CSV, as file
            called 'outputs_summary.csv'
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        outputs = pd.DataFrame(columns=['output'])    # Output DataFrame
        corrfac = (8760/self.num_timesteps)    # For annualisation

        # These should match the model topology specified in locations.yaml
        baseload_list, peaking_list, wind_list, unmet_list, demand_list = (
            [('baseload', i) for i in ['region1', 'region3', 'region6']],
            [('peaking', i) for i in ['region1', 'region3', 'region6']],
            [('wind', i) for i in ['region2', 'region5', 'region6']],
            [('unmet', i) for i in ['region2', 'region4', 'region5']],
            [('demand', i) for i in ['region2', 'region4', 'region5']]
        )

        transmission_region1to5_list, transmission_other_list, = (
            [('transmission_region1to5', 'region1', 'region5')],
            [('transmission_other', *i) for i in [('region1', 'region2'),
                                                  ('region1', 'region6'),
                                                  ('region2', 'region3'),
                                                  ('region3', 'region4'),
                                                  ('region4', 'region5'),
                                                  ('region5', 'region6')]]
        )

        # Insert baseload & peaking capacities
        for tech, region in baseload_list + peaking_list:
            outputs.loc['_'.join(('cap', tech, region))] = float(
                self.results.energy_cap.loc['::'.join((region, tech))]
            )

        # Insert wind capacities
        for tech, region in wind_list:
            outputs.loc['_'.join(('cap', tech, region))] = float(
                self.results.resource_area.loc['::'.join((region, tech))]
            )

        # Insert transmission capacities
        for transmission_type, regionA, regionB in \
            transmission_region1to5_list + transmission_other_list:
            outputs.loc['_'.join(('cap', 'transmission', regionA, regionB))] = \
                float(self.results.energy_cap.loc[
                    ':'.join((regionA + ':', transmission_type, regionB))
                ])

        # Insert annualised generation and unmet demand levels
        for tech, region in baseload_list+peaking_list+wind_list+unmet_list:
            outputs.loc['_'.join(('gen', tech, region))] = corrfac * float(
                self.results.carrier_prod.loc['::'.join((region,
                                                         tech,
                                                         'power'))].sum()
            )

        # Insert annualised demand levels
        for tech, region in demand_list:
            outputs.loc['_'.join(('demand', region))] = -corrfac * float(
                self.results.carrier_con.loc['::'.join((region,
                                                        'demand_power',
                                                        'power'))].sum()
            )

        # Insert total capacities
        for tech in ['baseload', 'peaking', 'wind', 'transmission']:
            outputs.loc['_'.join(('cap', tech, 'total'))] = \
                outputs.loc[outputs.index.str.contains(
                    '_'.join(('cap', tech)))].sum()

        # Insert total annualised generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'unmet']:
            outputs.loc['_'.join(('gen', tech, 'total'))] = \
                outputs.loc[outputs.index.str.contains(
                    '_'.join(('gen', tech)))].sum()

        # Insert total annualised demand levels
        outputs.loc['demand_total'] = \
            outputs.loc[outputs.index.str.contains('demand')].sum()

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            emission_levels=self.emission_levels,
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        # Run tests to check whether outputs are consistent -- for debugging
        if not tests.test_output_consistency(self, outputs):
            print('WARNING: model outputs are not consistent.\n'
                  'Check model configuration for possible bugs')

        if not at_regional_level:
            outputs = outputs.loc[[
                'cap_baseload_total', 'cap_peaking_total', 'cap_wind_total',
                'cap_transmission_total', 'gen_baseload_total',
                'gen_peaking_total', 'gen_wind_total', 'gen_unmet_total',
                'demand_total', 'cost_total', 'emissions_total'
                ]]

        if save_csv:
            outputs.to_csv('outputs_summary.csv')

        return outputs


class OneRegionModel(calliope.Model):
    """Instance of 1-region power system model."""

    def __init__(self, model_type, ts_data, preserve_index=False):
        """Create instance of 1-region model in Calliope.

        Parameters:
        -----------
        model_type (str) : either 'LP' (linear program) or 'MILP' (mixed
            integer linear program), depending on the baseload constraints.
        ts_data (pandas DataFrame) : time series with demand and wind data
        preserve_index (bool) : whether to use index from original time
            series. If False, the index is reset to hours starting in 1980.
            If True, the original time series index is used. This may lead
            to problems with leap days.
        """

        assert model_type in ['LP', 'MILP'], \
            'model_type must be either LP or MILP'

        self._base_dir = 'models/1_region'
        self.num_timesteps = ts_data.shape[0]

        # Emission intensity, ton CO2 equivalent per GWh
        self.emission_levels = {'baseload': 200,
                                'peaking': 400,
                                'wind': 0,
                                'unmet': 0}

        # Calliope requires a CSV file of time series data to be present
        # at time of model initialisation. This code creates a CSV with the
        # right format, initialises the model, then deletes the CSV
        ts_data = self._create_init_time_series(ts_data, preserve_index)
        ts_data_init_path = os.path.join(self._base_dir, 'demand_wind.csv')
        ts_data.to_csv(ts_data_init_path)
        super(OneRegionModel, self).__init__(os.path.join(self._base_dir,
                                                          'model.yaml'),
                                             scenario=model_type)
        os.remove(ts_data_init_path)

    def _create_init_time_series(self, ts_data, preserve_index=False):
        """Create demand and wind time series data for Calliope model
        initialisation.

        Parameters:
        -----------
        ts_data (pandas DataFrame) : demand and wind time series data
            to use in model
        preserve_index (bool) : if False, resets the time series index
            to a default starting at 1980-01-01 with hourly resolution. This
            avoids problems with leap days.
        """

        # Test if correct columns are present
        if set(ts_data.columns) != set(['demand', 'wind']):
            raise AttributeError('Input time series: incorrect columns')

        # Reset index if applicable
        if not preserve_index:
            ts_data.index = pd.date_range(start='1980-01-01',
                                          periods=self.num_timesteps,
                                          freq='h')
        else:
            print('WARNING: you have selected to maintain the original '
                  'time series index. Be careful with leap days')

        # Create CSV for model intialisation
        ts_data.loc[:, 'demand'] = -ts_data.loc[:, 'demand']

        return ts_data

    def get_summary_outputs(self, save_csv=False):
        """Create a pandas DataFrame of a subset of relevant model outputs

        Parameters:
        -----------
        at_regional_level (bool) : if True, gives each model output at
            regional level, otherwise the model totals
        save_csv (bool) : whether to save summary outputs as CSV, as file
            called 'outputs_summary.csv'
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        outputs = pd.DataFrame(columns=['output'])    # Output DataFrame
        corrfac = (8760/self.num_timesteps)    # For annualisation

        # Insert installed capacities
        outputs.loc['cap_baseload_total'] = \
            float(self.results.energy_cap.loc['region1::baseload'])
        outputs.loc['cap_peaking_total'] = \
            float(self.results.energy_cap.loc['region1::peaking'])
        outputs.loc['cap_wind_total'] = \
            float(self.results.resource_area.loc['region1::wind'])

        # Insert generation levels
        for tech in ['baseload', 'peaking', 'wind', 'unmet']:
            outputs.loc['gen_' + tech + '_total'] = corrfac * float(
                self.results.carrier_prod.loc[
                    'region1::' + tech + '::power'].sum()
        )

        # Insert annualised demand levels
        outputs.loc['demand_total'] = -corrfac * float(
            self.results.carrier_con.loc[
                'region1::demand_power::power'].sum()
        )

        # Insert annualised total system cost
        outputs.loc['cost_total'] = corrfac * float(self.results.cost.sum())

        # Insert annualised carbon emissions
        outputs.loc['emissions_total'] = calculate_carbon_emissions(
            emission_levels=self.emission_levels,
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        # # Run tests to check whether outputs are consistent -- for debugging
        # if not tests.test_output_consistency(self, outputs):
        #     print('WARNING: model outputs are not consistent.\n'
        #           'Check model configuration for possible bugs')

        # if save_csv:
        #     outputs.to_csv('outputs_summary.csv')

        return outputs
