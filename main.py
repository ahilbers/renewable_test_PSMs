"""Example model runs."""


import argparse
import logging
import models
import tests


def run_example(model_name, ts_data, run_mode,
                baseload_integer, baseload_ramping,
                allow_unmet):
    """Conduct an example model run.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    ts_data (pandas DataFrame) : time series with demand and wind data
    run_mode (str) : 'plan' or 'operate': whether to let the model
        determine the optimal capacities or work with prescribed ones
    baseload_integer (bool) : activate baseload integer capacity
        constraint (built in units of 3GW)
    baseload_ramping (bool) : enforce baseload ramping constraint
    allow_unmet (bool) : allow unmet demand in planning mode (always
        allowed in operate mode)
    """

    # Choose correct model and time series data properties
    if model_name == '1_region':
        Model = models.OneRegionModel
        test_output_consistency = tests.test_output_consistency_1_region
    if model_name == '6_region':
        Model = models.SixRegionModel
        test_output_consistency = tests.test_output_consistency_6_region

    # Create and run model
    model = Model(ts_data=ts_data,
                  run_mode=run_mode,
                  baseload_integer=baseload_integer,
                  baseload_ramping=baseload_ramping,
                  allow_unmet=allow_unmet)
    model.run()

    # Get DataFrame of key outputs and save to csv
    summary_outputs = model.get_summary_outputs()
    summary_outputs.to_csv('summary_outputs.csv')

    # Test consistency of summary outputs
    consistent_outputs = test_output_consistency(model, run_mode)
    if consistent_outputs:
        logging.info('Summary outputs are consistent.')
    else:
        logging.error('FAIL: Summary outputs are not consistent.')

    # Save full set of model outputs in new directory
    model.to_csv('full_outputs')


def run_example_from_command_line():
    """Read in a command line call and run the relevant example script."""

    # Read in the example name from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of example to run')
    parser.add_argument('--logging_level', required=False, type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'], default='INFO',
                        help='Python logging module verbosity level')
    args = parser.parse_args()

    # Read in command line arguments
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=getattr(logging, args.logging_level),
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Log the run characteristics
    logging.info('Run dictionary: \n%s\n', args)

    # Create the correct run dictionary. The following 3 serve only
    # as examples -- they can be customised as desired.
    if args.run_name == '1_region_plan_LP':
        ts_data = models.load_time_series_data(model_name='1_region')
        run_dict = {'model_name': '1_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'plan',
                    'baseload_integer': False,
                    'baseload_ramping': False,
                    'allow_unmet': False}
    elif args.run_name == '6_region_plan_MILP_unmet':
        ts_data = models.load_time_series_data(model_name='6_region')
        run_dict = {'model_name': '6_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'plan',
                    'baseload_integer': True,
                    'baseload_ramping': True,
                    'allow_unmet': True}
    elif args.run_name == '6_region_operate':
        ts_data = models.load_time_series_data(model_name='6_region')
        run_dict = {'model_name': '6_region',
                    'ts_data': ts_data.loc['2017-01'],
                    'run_mode': 'operate',
                    'baseload_integer': False,
                    'baseload_ramping': False,
                    'allow_unmet': True}
    else:
        raise ValueError('No run dictionary associated with this run_name.')

    # Run the model
    run_example(**run_dict)


if __name__ == '__main__':
    run_example_from_command_line()
