"""Example model runs."""


import argparse
import models


def run_example(model_name,
                ts_data,
                run_mode,
                baseload_integer,
                baseload_ramping):
    """Conduct an example model run."""

    # Choose correct model and time series data properties
    if model_name == '1_region':
        Model = models.OneRegionModel
    if model_name == '6_region':
        Model = models.SixRegionModel

    # Create and run model
    model = Model(ts_data=ts_data,
                  run_mode=run_mode,
                  baseload_integer=baseload_integer,
                  baseload_ramping=baseload_ramping)
    model.run()

    # Get DataFrame of key outputs and save to csv
    summary_outputs = model.get_summary_outputs()
    summary_outputs.to_csv('summary_outputs.csv')

    # Save full set of model outputs in new directory called 'outputs'
    model.to_csv('outputs')


def run_example_from_command_line():
    """Read in a command line call and run the relevant example script."""

    # Read in the example name from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of example to run')
    args = parser.parse_args()

    # Create the correct run dictionary. The following 3 serve only
    # as examples -- they can be customised as desired.
    if args.run_name == '1_region_plan_LP':
        ts_data = models.load_time_series_data(model_name='1_region')
        run_dict = {'model_name': '1_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'plan',
                    'baseload_integer': False,
                    'baseload_ramping': False}
    elif args.run_name == '6_region_plan_MILP':
        ts_data = models.load_time_series_data(model_name='6_region')
        run_dict = {'model_name': '6_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'plan',
                    'baseload_integer': True,
                    'baseload_ramping': True}
    elif args.run_name == '6_region_operate':
        ts_data = models.load_time_series_data(model_name='6_region')
        run_dict = {'model_name': '6_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'operate',
                    'baseload_integer': False,
                    'baseload_ramping': False}
    else:
        raise ValueError('No run dictionary associated with this run_name.')

    # Run the model
    run_example(**run_dict)


if __name__ == '__main__':
    run_example_from_command_line()
