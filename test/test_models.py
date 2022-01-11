import pytest
import typing
import numpy as np
import pandas as pd
import psm


def test_import():
    assert hasattr(psm, 'models')
    assert hasattr(psm, 'utils')


@pytest.mark.parametrize('model_name', ['1_region', '6_region'])
class TestModels:
    """Test core model functionality and some options."""

    @pytest.fixture(autouse=True)
    def _set_model_and_ts_data(self, model_name: str, ts_data_dict: typing.Dict[str, pd.DataFrame]):
        """Set model and ts_data used across this class."""
        self.model_name = model_name
        model_dict = {'1_region': psm.models.OneRegionModel, '6_region': psm.models.SixRegionModel}
        self.Model = model_dict[model_name]
        self.ts_data = ts_data_dict[model_name]


    @pytest.mark.parametrize('run_mode', ['plan', 'operate'])
    def test_model_basic(self, run_mode: str):
        """Test basic model initialisation and running."""

        # Check model is created correctly
        allow_unmet = True if run_mode == 'operate' else False
        model = self.Model(ts_data=self.ts_data, run_mode=run_mode, allow_unmet=allow_unmet)
        assert model.model_name == self.model_name
        assert model.num_timesteps == self.ts_data.shape[0]

        # Check time series data is added correctly
        ts_in = self.ts_data.copy()
        if self.model_name == '1_region':
            def get_ts_model_column_name(ts_in_column_name: str) -> str:
                if ts_in_column_name == 'demand':
                    ts_in_column_name = 'demand_power'
                return f'region1::{ts_in_column_name}'
            ts_in.loc[:, 'demand'] = -ts_in.loc[:, 'demand']  # Demand = negative
        elif self.model_name == '6_region':
            def get_ts_model_column_name(ts_in_column_name: str) -> str:
                tech, region = ts_in_column_name.split('_')
                if tech == 'demand':
                    return f'{region}::demand_power'
                return f'{region}::{tech}_{region}'
            ts_in_demand_cols = [i for i in ts_in.columns if 'demand' in i]
            ts_in.loc[:, ts_in_demand_cols] = -ts_in.loc[:, ts_in_demand_cols]  # Demand = negative
        ts_model_idx = [get_ts_model_column_name(i) for i in ts_in.columns]
        ts_model = model.inputs.resource.loc[ts_model_idx]
        assert np.allclose(ts_model.values.T, ts_in.values)

        # Check that model runs and gives sensible outputs
        model.run()
        summary_outputs = model.get_summary_outputs()
        summary_outputs_dict = model.get_summary_outputs(as_dict=True)
        assert summary_outputs.loc['cost_total', 'output'] > 0.
        assert np.isclose(
            summary_outputs.loc['cost_total', 'output'], summary_outputs_dict['cost_total']
        )
        if not allow_unmet:
            assert np.isclose(summary_outputs.loc['gen_unmet_total', 'output'], 0.)
        assert psm.utils.has_consistent_outputs(model)


    @pytest.mark.parametrize('run_mode', ['plan'])
    def test_model_set_fixed_caps(self, run_mode: str):
        """Test functionality to set fixed generation and transmission capacities."""

        # Set some fixed capacities
        fixed_caps_dict = {
            '1_region': {
                'cap_baseload_total': 20., 
                'cap_peaking_total': 20., 
                'cap_wind_total': 20.,
                'cap_solar_total': 15.
            },
            '6_region': {
                'cap_baseload_region1': 20.,
                'cap_peaking_region1': 25.,
                'cap_transmission_region1_region2': 30.,
                'cap_transmission_region1_region5': 20.,
                'cap_transmission_region1_region6': 10.,
                'cap_wind_region2': 40.,
                'cap_solar_region2': 20.,
                'cap_transmission_region2_region3': 40.,
                'cap_baseload_region3': 50.,
                'cap_peaking_region3': 20.,
                'cap_transmission_region3_region4': 30.,
                'cap_transmission_region4_region5': 30.,
                'cap_wind_region5': 40.,
                'cap_solar_region5': 30.,
                'cap_transmission_region5_region6': 10.,
                'cap_baseload_region6': 20.,
                'cap_peaking_region6': 20.,
                'cap_wind_region6': 30.,
                'cap_solar_region6': 20.,
            }
        }
        fixed_caps = fixed_caps_dict[self.model_name]

        # Create and run a model with these capacities and get summary outputs dictionary
        model = self.Model(ts_data=self.ts_data, run_mode=run_mode, fixed_caps=fixed_caps)
        model.run()
        summary_outputs = model.get_summary_outputs(as_dict=True)

        # Check that model capacities are the ones we set
        summary_outputs_caps = {key: summary_outputs[key] for key in fixed_caps.keys()}
        assert summary_outputs_caps == fixed_caps
