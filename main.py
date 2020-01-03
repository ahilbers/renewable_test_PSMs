import os
import numpy as np
import pandas as pd
import calliope
import pdb


class SixRegionModel(calliope.Model):
    """Instance of 6-region power system model."""

    def __init__(self, ts_data, preserve_index=False):
        """Create instance of 6-region model in Calliope.

        Parameters:
        -----------
        ts_data (pandas DataFrame) : time series with demand and wind data
        preserve_index (bool) : whether to use the index on the original time
            series. If False, the index is reset to hours starting in 1980.
            If True, the original time series index is used. This may lead
            to problems with leap days, which is why the index is reset
            be default.
        """

        self._base_dir = 'models/6_region'
        self._num_timesteps = ts_data.shape[0]
        self._init_time_series_data(ts_data, preserve_index)

    def _init_time_series_data(self, ts_data, preserve_index=False):
        """Initialise time series data in model. Calliope requires a CSV file
        to be present at time of initialisation. This function creates the
        relevant CSV file, uses it to create an instance of the model, and
        then deletes it again.
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

        # Create CSV, initialise model, and delete the CSV. Calliope requires
        # demand to be entered as negative values.
        ts_data.loc[:, 'demand_region2'] = -ts_data.loc[:, 'demand_region2']
        ts_data.loc[:, 'demand_region4'] = -ts_data.loc[:, 'demand_region4']
        ts_data.loc[:, 'demand_region5'] = -ts_data.loc[:, 'demand_region5']
        ts_data.to_csv(os.path.join(self._base_dir, 'demand_wind.csv'))
        super(SixRegionModel, self).__init__(os.path.join(self._base_dir,
                                                          'model.yaml'))
        os.remove(os.path.join(self._base_dir, 'demand_wind.csv'))


def test_script():
    """Run some tests."""
    dem_wind_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    dem_wind_data.index = pd.to_datetime(dem_wind_data.index)
    dem_wind_data = dem_wind_data.loc['1980']
    model = SixRegionModel(ts_data=dem_wind_data)


if __name__ == '__main__':
    test_script()
