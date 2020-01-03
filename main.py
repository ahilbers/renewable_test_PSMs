import os
import numpy as np
import pandas as pd
import calliope
import pdb


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

        if model_type not in ['LP', 'MILP']:
            raise ValueError('model_type must be either LP or MILP')

        self._base_dir = 'models/6_region'
        self._num_timesteps = ts_data.shape[0]
        self.emission_levels = {'baseload': 200, 'peaking': 400,
                                'wind': 0, 'unmet': 0}   # ton CO2e per GWh

        # Create CSV of time series data for model initialisation, create
        # the model, and delete the CSV
        self._init_time_series_csv(ts_data, preserve_index)
        super(SixRegionModel, self).__init__(os.path.join(self._base_dir,
                                                          'model.yaml'),
                                             scenario=model_type)
        os.remove(os.path.join(self._base_dir, 'demand_wind.csv'))

    def _init_time_series_csv(self, ts_data, preserve_index=False):
        """Initialise time series data in model. Calliope requires a CSV file
        to be present at time of initialisation. This function creates the
        relevant CSV file, which can be used to intialise the model and then,
        if desired, deleted again.
        """

        # Test if correct columns are present
        if set(ts_data.columns) != set(['demand_region2', 'demand_region4',
                                        'demand_region5', 'wind_region2',
                                        'wind_region5', 'wind_region6']):
            raise AttributeError('Input time series data: incorrect columns')

        # Reset index if applicable
        if not preserve_index:
            ts_data.index = pd.date_range(start='1980-01-01',
                                          periods=self._num_timesteps,
                                          freq='h')
        else:
            print('WARNING: you have selected to maintain the original ' +
                  'time series index. Be careful with leap days.')

        # Create CSV for model intialisation
        ts_data.loc[:, 'demand_region2'] = -ts_data.loc[:, 'demand_region2']
        ts_data.loc[:, 'demand_region4'] = -ts_data.loc[:, 'demand_region4']
        ts_data.loc[:, 'demand_region5'] = -ts_data.loc[:, 'demand_region5']
        ts_data.to_csv(os.path.join(self._base_dir, 'demand_wind.csv'))

    def get_summary_outputs(self, at_regional_level=False, save_csv=False):
        """Create a pandas DataFrame of a subset of relevant model outputs

        Parameters:
        -----------
        save_csv (bool) : whether to save summary outputs as CSV, as file
            called 'outputs_summary.csv'
        """

        assert hasattr(self, 'results'), \
            'Model outputs have not been calculated: call self.run() first.'

        R = self.results    # For concise code
        out = pd.DataFrame(columns=['output'])    # Output DataFrame

        # Insert optimal installed capacities
        out.loc['cap_r1_bl'] = float(R.energy_cap.loc['region1::baseload'])
        out.loc['cap_r1_pk'] = float(R.energy_cap.loc['region1::peaking'])
        out.loc['cap_r2_wd'] = float(R.resource_area.loc['region2::wind'])
        out.loc['cap_r3_bl'] = float(R.energy_cap.loc['region3::baseload'])
        out.loc['cap_r3_pk'] = float(R.energy_cap.loc['region3::peaking'])
        out.loc['cap_r5_wd'] = float(R.resource_area.loc['region5::wind'])
        out.loc['cap_r6_bl'] = float(R.energy_cap.loc['region6::baseload'])
        out.loc['cap_r6_pk'] = float(R.energy_cap.loc['region6::peaking'])
        out.loc['cap_r6_wd'] = float(R.resource_area.loc['region6::wind'])
        out.loc['cap_tr12'] = float(R.energy_cap.loc['region1::transmission_other:region2'])
        out.loc['cap_tr15'] = float(R.energy_cap.loc['region1::transmission_region1to5:region5'])
        out.loc['cap_tr16'] = float(R.energy_cap.loc['region1::transmission_other:region6'])
        out.loc['cap_tr23'] = float(R.energy_cap.loc['region2::transmission_other:region3'])
        out.loc['cap_tr34'] = float(R.energy_cap.loc['region3::transmission_other:region4'])
        out.loc['cap_tr45'] = float(R.energy_cap.loc['region4::transmission_other:region5'])
        out.loc['cap_tr56'] = float(R.energy_cap.loc['region5::transmission_other:region6'])
        out.loc['cap_tot_bl'] = out.loc[['cap_r1_bl', 'cap_r3_bl', 'cap_r6_bl']].sum()
        out.loc['cap_tot_pk'] = out.loc[['cap_r1_pk', 'cap_r3_pk', 'cap_r6_pk']].sum()
        out.loc['cap_tot_wd'] = out.loc[['cap_r2_wd', 'cap_r5_wd', 'cap_r6_wd']].sum()
        out.loc['cap_tot_tr'] = out.loc[['cap_tr12', 'cap_tr15', 'cap_tr16', 'cap_tr23',
                                         'cap_tr34', 'cap_tr45', 'cap_tr56']].sum()

        # Insert annualised generation levels
        corrfac = (8760/self._num_timesteps)
        out.loc['gen_r1_bl'] = corrfac * float(R.carrier_prod.loc['region1::baseload::power'].sum())
        out.loc['gen_r1_pk'] = corrfac * float(R.carrier_prod.loc['region1::peaking::power'].sum())
        out.loc['gen_r2_wd'] = corrfac * float(R.carrier_prod.loc['region2::wind::power'].sum())
        out.loc['gen_r2_um'] = corrfac * float(R.carrier_prod.loc['region2::unmet::power'].sum())
        out.loc['gen_r3_bl'] = corrfac * float(R.carrier_prod.loc['region3::baseload::power'].sum())
        out.loc['gen_r3_pk'] = corrfac * float(R.carrier_prod.loc['region3::peaking::power'].sum())
        out.loc['gen_r4_um'] = corrfac * float(R.carrier_prod.loc['region4::unmet::power'].sum())
        out.loc['gen_r5_wd'] = corrfac * float(R.carrier_prod.loc['region5::wind::power'].sum())
        out.loc['gen_r5_um'] = corrfac * float(R.carrier_prod.loc['region5::unmet::power'].sum())
        out.loc['gen_r6_bl'] = corrfac * float(R.carrier_prod.loc['region6::baseload::power'].sum())
        out.loc['gen_r6_pk'] = corrfac * float(R.carrier_prod.loc['region6::peaking::power'].sum())
        out.loc['gen_r6_wd'] = corrfac * float(R.carrier_prod.loc['region6::wind::power'].sum())
        out.loc['gen_tot_bl'] = out.loc[['gen_r1_bl', 'gen_r3_bl', 'gen_r6_bl']].sum()
        out.loc['gen_tot_pk'] = out.loc[['gen_r1_pk', 'gen_r3_pk', 'gen_r6_pk']].sum()
        out.loc['gen_tot_wd'] = out.loc[['gen_r2_wd', 'gen_r5_wd', 'gen_r6_wd']].sum()
        out.loc['gen_tot_um'] = out.loc[['gen_r2_um', 'gen_r4_um', 'gen_r5_um']].sum()

        # Insert annualised total system cost
        out.loc['cost_tot'] = corrfac * float(R.cost.sum())

        out.loc['emissions_tot'] = self.calculate_carbon_emissions(
            generation_levels={'baseload': out.loc['gen_tot_bl'],
                               'peaking': out.loc['gen_tot_pk'],
                               'wind': out.loc['gen_tot_wd'],
                               'unmet': out.loc['gen_tot_um']}
            )

        ### CONDUCT TESTS

        if not at_regional_level:
            out = out.loc[[
                'cap_tot_bl', 'cap_tot_pk', 'cap_tot_wd', 'cap_tot_tr',
                'gen_tot_bl', 'gen_tot_pk', 'gen_tot_wd', 'gen_tot_um',
                'cost_tot', 'emissions_tot'
                ]]

        if save_csv:
            out.to_csv('outputs_summary.csv')

        return out

    def calculate_carbon_emissions(self, generation_levels):
        """Calculate total carbon emissions.

        Parameters:
        -----------
        generation_levels (pandas DataFrame or dict) : the generation levels
            for the 4 technologies (baseload, peaking, wind and unmet)
        """

        emissions_tot = \
            self.emission_levels['baseload']*generation_levels['baseload'] + \
            self.emission_levels['peaking']*generation_levels['peaking'] + \
            self.emission_levels['wind']*generation_levels['wind'] + \
            self.emission_levels['unmet']*generation_levels['unmet']

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
