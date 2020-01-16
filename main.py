"""Example model runs."""


import argparse
import models


def run_example_1_region_plan():
    """Run the 1_region model in plan mode, with continuous baseload
    capacity without ramping constraints."""

    # Load and subset time series data
    ts_data = models.load_time_series_data(model_name='1_region',
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc['2017-01']

    # Create and run model
    model = models.OneRegionModel(ts_data=ts_data,
                                  run_mode='plan',
                                  baseload_integer=False,
                                  baseload_ramping=False)
    model.run()

    # Get DataFrame of the key outputs and save to csv
    summary_outputs = model.get_summary_outputs()
    summary_outputs.to_csv('summary_outputs.csv')


def run_example_6_region_plan():
    """Run the 6_region model in plan mode, with integer baseload
    capacity with ramping constraints."""

    # Load and subset time series data
    ts_data = models.load_time_series_data(model_name='6_region')
    ts_data = ts_data.loc['2017-01']

    # Create and run model
    model = models.SixRegionModel(ts_data=ts_data,
                                  run_mode='plan',
                                  baseload_integer=True,
                                  baseload_ramping=True)
    model.run()

    # Save full set of model outputs in new directory called 'outputs'
    model.to_csv('outputs')

    # Get DataFrame of the key outputs and save to csv
    summary_outputs = model.get_summary_outputs(at_regional_level=False)
    summary_outputs.to_csv('summary_outputs.csv')


def run_example_6_region_operate():
    """Run the 6_region model in operate mode, with baseload ramping
    constraints."""

    # Load and subset time series data
    ts_data = models.load_time_series_data(model_name='6_region')
    ts_data = ts_data.loc['2017-01']

    # Create dictionary for fixed capacities. These override the values
    # found in 'model.yaml'. If the model is run using fixed_caps=None,
    # the values in 'model.yaml' are used instead.
    fixed_caps = {'cap_baseload_region1': 10,
                  'cap_baseload_region3': 20,
                  'cap_baseload_region6': 30,
                  'cap_peaking_region1': 40,
                  'cap_peaking_region3': 50,
                  'cap_peaking_region6': 60,
                  'cap_wind_region2': 70,
                  'cap_wind_region5': 80,
                  'cap_wind_region6': 90,
                  'cap_transmission_region1_region2': 12,
                  'cap_transmission_region1_region5': 15,
                  'cap_transmission_region1_region6': 16,
                  'cap_transmission_region2_region3': 23,
                  'cap_transmission_region3_region4': 34,
                  'cap_transmission_region4_region5': 45,
                  'cap_transmission_region5_region6': 56}

    # Create and run model
    model = models.SixRegionModel(ts_data=ts_data,
                                  run_mode='operate',
                                  baseload_integer=False,
                                  baseload_ramping=True,
                                  fixed_caps=fixed_caps)
    model.run()

    # Get DataFrame of the key outputs and save to csv
    summary_outputs = model.get_summary_outputs()
    summary_outputs.to_csv('summary_outputs.csv')


def run_example_from_command_line():
    """Read in a command line call and run the relevant example script."""

    # Read in the example name from command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='Name of example to run')
    args = parser.parse_args()

    # Run the correct example
    if args.run_name == '1_region_plan':
        run_example_1_region_plan()
    elif args.run_name == '6_region_plan':
        run_example_6_region_plan()
    elif args.run_name == '6_region_operate':
        run_example_6_region_operate()
    else:
        raise ValueError('Invalid run name.')


if __name__ == '__main__':
    run_example_from_command_line()
