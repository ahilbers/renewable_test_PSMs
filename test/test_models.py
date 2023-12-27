import pytest
import typing
import numpy as np
import pandas as pd
import psm


def test_import():
    assert hasattr(psm, 'models')
    assert hasattr(psm, 'utils')


def test_invalid_model_name():
    '''Should get ValueError when calling different name than '1_region' or '6_region.'''
    with pytest.raises(ValueError, match='Invalid model name'):
        model = psm.models.ModelBase(model_name='invalid_name', ts_data=None, run_mode='plan')


@pytest.mark.parametrize('model_name', ['1_region', '6_region'])
class TestModels:
    '''Test core model functionality and some options.'''

    @pytest.fixture(autouse=True)
    def _set_model_and_ts_data(self, model_name: str, ts_data_dict: typing.Dict[str, pd.DataFrame]):
        '''Set model and ts_data used across this class.'''
        self.model_name = model_name
        model_dict = {'1_region': psm.models.OneRegionModel, '6_region': psm.models.SixRegionModel}
        self.Model = model_dict[model_name]
        self.ts_data = ts_data_dict[model_name]

    def test_check_ts_column_names(self):
        '''Should get AttributeError when using timeseries with wrong column names.'''
        with pytest.raises(AttributeError, match='Incorrect columns in input time series.'):
            ts_data_wrong_columns = self.ts_data.rename(
                columns={"demand": "wrong_name", "demand_region2": "wrong_name"}
            )
            _ = self.Model(ts_data=ts_data_wrong_columns, run_mode='plan')

    def test_must_allow_unmet_in_operate_mode(self):
        '''Should get ValueError for model in 'operate' mode without allowing unmet demand.'''
        with pytest.raises(ValueError, match='Must allow unmet demand'):
            _ = self.Model(ts_data=self.ts_data, run_mode='operate', allow_unmet=False)

    @pytest.mark.parametrize('run_mode', ['plan', 'operate'])
    def test_model_basic(self, run_mode: str):
        '''Test basic model initialisation and running.'''

        # Check model is created correctly
        allow_unmet = True if run_mode == 'operate' else False
        model = self.Model(ts_data=self.ts_data, run_mode=run_mode, allow_unmet=allow_unmet)
        assert model.model_name == self.model_name
        assert model.num_timesteps == self.ts_data.shape[0]

        def get_ts_model_column_name(ts_in_column_name: str) -> str:
            '''Map column name in ts_in to key in Calliope model.'''
            if self.model_name == '1_region':
                return f'region1::{ts_in_column_name.replace("demand", "demand_power")}'
            elif self.model_name == '6_region':
                tech, region = ts_in_column_name.split('_')
                if tech == 'demand':
                    return f'{region}::demand_power'
                return f'{region}::{tech}_{region}'

        # Check time series data is added correctly
        ts_in = self.ts_data.copy()
        ts_in_demand_cols = [i for i in ts_in.columns if 'demand' in i]
        ts_in.loc[:, ts_in_demand_cols] = -ts_in.loc[:, ts_in_demand_cols]  # Make demand negative
        ts_model_idx = [get_ts_model_column_name(i) for i in ts_in.columns]
        ts_model = model.inputs.resource.loc[ts_model_idx]
        assert np.allclose(ts_model.values.T, ts_in.values)

        # Should throw error if calling .get_summary_outputs() before solving model
        with pytest.raises(AttributeError, match='Model outputs not yet calculated'):
            model.get_summary_outputs()

        # Check that model runs
        model.run()

        # Checks on summary model outputs
        summary_outputs = model.get_summary_outputs()
        summary_outputs_dict = model.get_summary_outputs(as_dict=True)
        assert summary_outputs.loc['cost_total', 'output'] > 0.
        assert np.isclose(
            summary_outputs.loc['cost_total', 'output'], summary_outputs_dict['cost_total']
        )
        if not allow_unmet:
            assert np.isclose(summary_outputs.loc['gen_unmet_total', 'output'], 0.)
        assert psm.utils.has_consistent_outputs(model)

        # Check that model can produce timeseries outputs and that they make sense
        ts_outputs = model.get_timeseries_outputs()
        assert isinstance(ts_outputs, pd.DataFrame)
        assert (ts_outputs.index == model.inputs.timesteps.values).all()
        assert np.allclose(
            ts_outputs.filter(like='gen', axis=1).sum(axis=1),
            ts_outputs.filter(like='demand', axis=1).sum(axis=1)
        )  # Check generation matches demand in each time step

    @pytest.mark.parametrize('run_mode', ['plan', 'operate'])
    def test_model_set_fixed_caps(self, run_mode: str):
        '''Test functionality to set fixed generation and transmission capacities.'''

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
        model = self.Model(
            ts_data=self.ts_data, run_mode=run_mode, fixed_caps=fixed_caps, allow_unmet=True
        )
        model.run()
        summary_outputs = model.get_summary_outputs(as_dict=True)

        # Check that model capacities are the ones we set
        summary_outputs_caps = {key: summary_outputs[key] for key in fixed_caps.keys()}
        assert summary_outputs_caps == fixed_caps

    def test_ts_weights(self):
        '''Test the ability to set weights for each time step.'''

        # Create weighted version of ts_data
        ts_data_weighted = self.ts_data.copy()
        weights = np.ones(ts_data_weighted.shape[0], dtype='float')
        weights[:round(weights.shape[0]/2)] = 0.5
        weights[round(weights.shape[0]/2):] = 1.5
        ts_data_weighted.loc[:, 'weight'] = weights

        model_unweighted = self.Model(ts_data=self.ts_data, run_mode='plan')
        model_weighted = self.Model(ts_data=ts_data_weighted, run_mode='plan')
        model_unweighted.run()
        model_weighted.run()

        assert np.allclose(model_unweighted.inputs.timestep_weights.values, 1.)
        assert np.allclose(model_weighted.inputs.timestep_weights.values, weights)
        assert not np.allclose(
            model_unweighted.get_summary_outputs().filter(like='cap', axis=0),
            model_weighted.get_summary_outputs().filter(like='cap', axis=0)
        )  # Installed capacities should be different

    def test_baseload_integer_and_ramping(self):
        '''Test baseload integer and ramping constraints.'''

        # Set renewable caps to 0 to encourage model to install baseload
        fixed_caps_dict = {
            '1_region': {'cap_wind_total': 0., 'cap_solar_total': 0.},
            '6_region': {
                'cap_wind_region2': 0.,
                'cap_solar_region2': 0.,
                'cap_wind_region5': 0.,
                'cap_solar_region5': 0.,
                'cap_wind_region6': 0.,
                'cap_solar_region6': 0.,
            }
        }
        fixed_caps = fixed_caps_dict[self.model_name]

        # Create and solve model and check baseload capacities
        model = self.Model(
            ts_data=self.ts_data,
            run_mode='plan',
            baseload_integer=True,
            baseload_ramping=True,
            fixed_caps=fixed_caps
        )
        model.run()

        # Check baseload capacities are in the unit size
        baseload_discrete_size = 3  # Hard coded for now
        baseload_caps = model.get_summary_outputs().filter(like='cap_baseload', axis=0)
        assert np.allclose(baseload_caps % baseload_discrete_size, 0.)

    def test_extra_override(self):
        '''Test the ability to set an extra override.'''
        # Use the extra override 'gurobi' which changes the solver
        model = self.Model(ts_data=self.ts_data, run_mode='plan', extra_override='gurobi')
        assert model.run_config['solver'] == 'gurobi'
