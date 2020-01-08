"""The test case power system models."""


import os
import pandas as pd
import calliope


# Emission intensities of technologies, in ton CO2 equivalent per GWh
EMISSION_INTENSITIES = {'baseload': 200,
                        'peaking': 400,
                        'wind': 0,
                        'unmet': 0}


def load_time_series_data(model_name, demand_region=None, wind_region=None):
    """Load demand and wind time series data for model.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    demand_region (str) : region in data/demand_wind.csv from which
        to take demand data. Use only if model_name='1_region'
    wind_region (str) : region in data/demand_wind.csv from which to
        take wind data. Use only if model_name='1_region'

    Returns:
    --------
    ts_data (pandas DataFrame) : time series data for use in model
    """

    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)

    # If 1_region model, select regions from which to take demand, wind data
    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['_'.join(('demand', demand_region)),
                                  '_'.join(('wind', wind_region))]]
        ts_data.columns = ['demand', 'wind']

    return ts_data


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
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'unmet': outputs.loc['gen_unmet_total']
            }
        )

        if save_csv:
            outputs.to_csv('outputs_summary.csv')

        return outputs


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

        # Insert model outputs at regional level
        for region in ['region{}'.format(i+1) for i in range(6)]:

            # Baseload and peaking capacity
            for tech in ['baseload', 'peaking']:
                try:
                    outputs.loc['_'.join(('cap', tech, region))] = float(
                        self.results.energy_cap.loc[region + '::' + tech]
                    )
                except KeyError:
                    pass

            # Wind capacity
            for tech in ['wind']:
                try:
                    outputs.loc['_'.join(('cap', tech, region))] = float(
                        self.results.resource_area.loc[region + '::' + tech]
                    )
                except KeyError:
                    pass

            # Transmission capacity
            for transmission_type in ['transmission_region1to5',
                                      'transmission_other']:
                for to in ['region{}'.format(i+1) for i in range(6)]:
                    try:
                        outputs.loc['cap_transmission_' + region + '_'+ to] = float(
                            self.results.energy_cap.loc[
                                region + '::' + transmission_type + ':' + to
                            ]
                        )
                    except KeyError:
                        pass

            # Baseload, peaking, wind and unmet generation levels
            for tech in ['baseload', 'peaking', 'wind', 'unmet']:
                try:
                    outputs.loc['gen_' + tech + '_' + region] = corrfac * float(
                        self.results.carrier_prod.loc[
                            region + '::' + tech + '::' + 'power'
                        ].sum()
                    )
                except KeyError:
                    pass

            # Demand levels
            try:
                outputs.loc['demand_' + region] = -corrfac * float(
                    self.results.carrier_con.loc[
                        region+ '::demand_power::power'
                    ].sum()
                )
            except KeyError:
                pass

        # Insert total capacities
        for tech in ['baseload', 'peaking', 'wind', 'transmission']:
            outputs.loc['cap_' + tech + '_total'] = outputs.loc[
                outputs.index.str.contains('cap_' + tech)
            ].sum()

        # Insert total annualised generation and unmet demand levels
        for tech in ['baseload', 'peaking', 'wind', 'unmet']:
            outputs.loc['gen_' + tech + '_total'] = outputs.loc[
                outputs.index.str.contains('gen_' + tech)
            ].sum()

        # Insert total annualised demand levels
        outputs.loc['demand_total'] = \
            outputs.loc[outputs.index.str.contains('demand')].sum()

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
