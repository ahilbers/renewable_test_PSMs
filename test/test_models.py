import pytest
import typing
import numpy as np
import pandas as pd
import test_utils
import psm


def test_import():
    assert hasattr(psm, 'models')
    assert hasattr(psm, 'utils')


@pytest.mark.parametrize('model_name', ['1_region'])
class TestModels:
    """Test core model functionality and some options."""

    @pytest.fixture(autouse=True)
    def _set_model_and_ts_data(self, model_name, ts_data_dict: typing.Dict[str, pd.DataFrame]):
        """Set model and ts_data used across this class."""
        self.model_name = model_name
        model_dict = {'1_region': psm.models.OneRegionModel, '6_region': psm.models.SixRegionModel}
        self.Model = model_dict[model_name]
        self.ts_data = ts_data_dict[model_name]


    @pytest.mark.parametrize('run_mode', ['plan'])
    def test_core_model(self, run_mode):
        """Test model initialisation with default settings."""

        # Check model is created correctly
        model = self.Model(ts_data=self.ts_data, run_mode=run_mode)
        assert model.model_name == self.model_name
        assert model.num_timesteps == self.ts_data.shape[0]

        # Check time series data is added correctly
        if self.model_name == '1_region':
            ts_in = self.ts_data.loc[:, ['demand', 'wind', 'solar']]
            ts_in.loc[:, 'demand'] = -ts_in.loc[:, 'demand']  # Demand is negative in Calliope
            ts_model = model.inputs.resource.loc[
                ['region1::demand_power', 'region1::wind', 'region1::solar']
            ].values.T
            assert np.allclose(ts_model, ts_in)
        elif self.model_name == '6_region':
            assert False

        # Check that model runs and gives sensible outputs
        model.run()
        summary_outputs = model.get_summary_outputs()
        summary_outputs_dict = model.get_summary_outputs(as_dict=True)
        assert summary_outputs.loc['cost_total', 'output'] > 0.
        assert np.isclose(
            summary_outputs.loc['demand_total', 'output'],  self.ts_data.loc[:, 'demand'].sum()
        )
        assert np.isclose(
            summary_outputs.loc['cost_total', 'output'], summary_outputs_dict['cost_total']
        )
