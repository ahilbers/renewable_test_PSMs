import os
import numpy as np
import pandas as pd
import calliope


class SixRegionModel(calliope.Model):
    """Instance of 6-region power system model."""

    def __init__(self, model_type, ts_data, preserve_index=False):
        """Create instance of 6-region model in Calliope.

        Parameters:
        -----------
        model_type (str) : either 'LP' (linear program) or 'MILP' (mixed
            integer linear program), depending on the baseload constraints.
        ts_data (pandas DataFrame) : time series with demand and wind data
        preserve_index (bool) : whether to use the index on the original time
            series. If False, the index is reset to hours starting in 1980.
            If True, the original time series index is used. This may lead
            to problems with leap days, which is why the index is reset
            be default.
        """

        assert model_type in ['LP', 'MILP'], \
            'model_type must be either LP or MILP'

        self._base_dir = 'models/6_region'
        self.num_timesteps = ts_data.shape[0]
        self.emission_levels = {'baseload': 200,
                                'peaking': 400,
                                'wind': 0,
                                'unmet': 0}   # ton CO2e per GWh

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
            raise AttributeError('Input time series data: incorrect columns')

        # Reset index if applicable
        if not preserve_index:
            ts_data.index = pd.date_range(start='1980-01-01',
                                          periods=self.num_timesteps,
                                          freq='h')
        else:
            print('WARNING: you have selected to maintain the original ' +
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
        save_csv (bool) : whether to save summary outputs as CSV, as file
            called 'outputs_summary.csv'
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        outputs = pd.DataFrame(columns=['output'])    # Output DataFrame
        corrfac = (8760/self.num_timesteps)    # For annualisation

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
                    ':'.join((regionA + ':', transmission_type, regionB))])

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

        outputs.loc['emissions_total'] = self.calculate_carbon_emissions(
            generation_levels={
                'baseload': outputs.loc['gen_baseload_total'],
                'peaking': outputs.loc['gen_peaking_total'],
                'wind': outputs.loc['gen_wind_total'],
                'unmet': outputs.loc['gen_unmet_total']})

        ### CONDUCT TESTS

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

    def calculate_carbon_emissions(self, generation_levels):
        """Calculate total carbon emissions.

        Parameters:
        -----------
        generation_levels (pandas DataFrame or dict) : generation levels
            for the 4 technologies (baseload, peaking, wind and unmet)
        """

        em_lvls = self.emission_levels
        gen_lvls = generation_levels

        emissions_tot = em_lvls['baseload'] * gen_lvls['baseload'] + \
            em_lvls['peaking'] * gen_lvls['peaking'] + \
            em_lvls['wind'] * gen_lvls['wind'] + \
            em_lvls['unmet'] * gen_lvls['unmet']

        return emissions_tot



def test_script():
    """Run some tests."""
    dem_wind_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    dem_wind_data.index = pd.to_datetime(dem_wind_data.index)
    dem_wind_data = dem_wind_data.loc['1980']
    model = SixRegionModel(ts_data=dem_wind_data, model_type='LP')
    pdb.set_trace()
    model.run()
    results = model.get_summary_outputs(save_csv=True)


if __name__ == '__main__':
    test_script()
