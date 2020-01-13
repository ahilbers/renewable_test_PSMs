"""Tests to check whether the models behave as required."""


import os
import pandas as pd
import time
import models
import pdb


# Install costs and generation costs. These should match the information
# provided in the model.yaml and techs.yaml files in the model definition
COSTS = pd.DataFrame(columns=['install', 'generation'])
COSTS.loc['baseload']                = [300., 0.006]
COSTS.loc['peaking']                 = [100., 0.035]
COSTS.loc['wind']                    = [100., 0.   ]
COSTS.loc['unmet']                   = [  0., 6.   ]
COSTS.loc['transmission_other']      = [100., 0.   ]
COSTS.loc['transmission_region1to5'] = [150., 0.   ]


# Topology of 6 region model. These should match the information provided
# in the locations.yaml files in the model definition
BASELOAD_TOP, PEAKING_TOP, WIND_TOP, UNMET_TOP, DEMAND_TOP = (
    [('baseload', i) for i in ['region1', 'region3', 'region6']],
    [('peaking', i) for i in ['region1', 'region3', 'region6']],
    [('wind', i) for i in ['region2', 'region5', 'region6']],
    [('unmet', i) for i in ['region2', 'region4', 'region5']],
    [('demand', i) for i in ['region2', 'region4', 'region5']]
)
TRANSMISSION_REGION1TO5_TOP, TRANSMISSION_OTHER_TOP = (
    [('transmission_region1to5', 'region1', 'region5')],
    [('transmission_other', *i) for i in [('region1', 'region2'),
                                          ('region1', 'region6'),
                                          ('region2', 'region3'),
                                          ('region3', 'region4'),
                                          ('region4', 'region5'),
                                          ('region5', 'region6')]]
)


def test_output_consistency_1_region(model, run_mode):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = model.get_summary_outputs()
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    if run_mode == 'plan':
        for tech in ['baseload', 'peaking', 'wind']:
            cost_method1 = float(COSTS.loc[tech, 'install']
                                 * out.loc['cap_{}_total'.format(tech)])
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['region1::{}'.format(tech)]
            )
            if abs(cost_method1 - cost_method2) > 1:
                print('FAIL: {} install costs do not match!\n'
                      '    manual: {}, model: {}'.format(
                          tech, cost_method1, cost_method2))
                passing = False
            cost_total_method1 += cost_method1

    # Test if generation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        cost_method1 = float(COSTS.loc[tech, 'generation']
                             * out.loc['gen_{}_total'.format(tech)])
        cost_method2 = corrfac * float(res.cost_var[0].loc[
            'region1::{}'.format(tech)
        ].sum())
        if abs(cost_method1 - cost_method2) > 1:
            print('FAIL: {} generation costs do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 1:
            print('FAIL: total system costs do not match!\n'
                  '    manual: {}, model: {}'.format(cost_total_method1,
                                                     cost_total_method2))
            passing = False

    # Test if supply matches demand
    generation_total = float(out.loc[['gen_baseload_total',
                                      'gen_peaking_total',
                                      'gen_wind_total',
                                      'gen_unmet_total']].sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 1:
        print('FAIL: generation does not match demand!\n'
              '    generation: {}, demand: {}'.format(generation_total,
                                                      demand_total))
        passing = False

    return passing


def test_output_consistency_6_region(model, run_mode):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = model.get_summary_outputs(at_regional_level=True)
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    if run_mode == 'plan':
        for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP:
            cost_method1 = float(
                COSTS.loc[tech, 'install'] *
                out.loc['cap_{}_{}'.format(tech, region)]
            )
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['{}::{}'.format(region, tech)]
            )
            if abs(cost_method1 - cost_method2) > 1:
                print('FAIL: {} install costs in {} do not match!\n'
                      '    manual: {}, model: {}'.format(tech,
                                                         region,
                                                         cost_method1,
                                                         cost_method2))
                passing = False
            cost_total_method1 += cost_method1

    # Test if transmission installation costs are consistent
    if run_mode == 'plan':
        for tech, region_a, region_b in \
            TRANSMISSION_REGION1TO5_TOP + TRANSMISSION_OTHER_TOP:
            cost_method1 = float(
                COSTS.loc[tech, 'install'] * out.loc[
                    'cap_transmission_{}_{}'.format(region_a, region_b)
                ]
            )
            cost_method2 = 2 * corrfac * \
                float(res.cost_investment[0].loc[
                    '{}::{}:{}'.format(region_a, tech, region_b)
                ])
            if abs(cost_method1 - cost_method2) > 1:
                print('FAIL: {} install costs from {} to {} do not match!\n'
                      '    manual: {}, model: {}'.format(tech,
                                                         region_a,
                                                         region_b,
                                                         cost_method1,
                                                         cost_method2))
                passing = False
            cost_total_method1 += cost_method1

    # Test if generation costs are consistent
    for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP + UNMET_TOP:
        cost_method1 = float(
            COSTS.loc[tech, 'generation']
            * out.loc['gen_{}_{}'.format(tech, region)]
        )
        cost_method2 = corrfac * float(
            res.cost_var[0].loc[region + '::' + tech].sum()
        )
        if abs(cost_method1 - cost_method2) > 1:
            print('FAIL: {} generation costs in {} do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, region, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 1:
            print('FAIL: total system costs do not match!\n'
                  '    manual: {}, model: {}'.format(cost_total_method1,
                                                     cost_total_method2))
            passing = False

    # Test if supply matches demand
    generation_total = float(out.loc[['gen_baseload_total',
                                      'gen_peaking_total',
                                      'gen_wind_total',
                                      'gen_unmet_total']].sum())
    demand_total = float(out.loc['demand_total'])
    if abs(generation_total - demand_total) > 1:
        print('FAIL: generation does not match demand!\n'
              '    generation: {}, demand: {}'.format(generation_total,
                                                      demand_total))
        passing = False

    return passing


def test_output_consistency(model, model_name, run_mode):
    """Check if model outputs are internally consistent.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    model_name (str) : '1_region' or '6_region'
    run_mode (str) : 'plan' or 'operate'

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    # Run the consistency tests
    if model_name == '1_region':
        passing = test_output_consistency_1_region(model, run_mode)
    if model_name == '6_region':
        passing = test_output_consistency_6_region(model, run_mode)

    if passing:
        print('PASS: model outputs are consistent.')

    return passing


def test_outputs_against_benchmark(model, model_name, run_mode,
                                   baseload_integer, baseload_ramping):
    """Test model outputs against benchmark."""

    if not baseload_integer and not baseload_ramping:
        baseload_name = 'continuous'
    elif baseload_integer and baseload_ramping:
        baseload_name = 'integer_ramping'
    else:
        raise ValueError('Benchmark outputs not available for '
                         'this model setup.')

    passing = True
    summary_outputs = model.get_summary_outputs()

    # Load benchmarks
    benchmark_outputs = pd.read_csv(
        os.path.join('benchmarks', '{}_{}_{}_2017-01.csv'.format(
            model_name, run_mode, baseload_name)),
        index_col=0
    )

    if float(abs(summary_outputs - benchmark_outputs).max()) > 1:
        print('FAIL: Model outputs do not match benchmark outputs!')
        print('Model outputs:\n', summary_outputs)
        print('')
        print('Benchmark outputs:\n', benchmark_outputs)
        passing = False

    if passing:
        print('PASS: model outputs match benchmark outputs.')

    return passing


def run_tests(model_name, run_mode, baseload_integer, baseload_ramping,
              fixed_capacities):
    """ Run the tests."""

    # Load time series data
    ts_data = models.load_time_series_data(model_name=model_name,
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc['2017-01']    # Should match benchmarks

    # Run a simulation on which to run tests
    print('TESTS: Running test simulation...')
    if model_name == '1_region':
        model = models.OneRegionModel
    elif model_name == '6_region':
        model = models.SixRegionModel
    else:
        raise ValueError('Valid model names: 1_region, 6_region')
    model = model(ts_data, run_mode, baseload_integer, baseload_ramping,
                  fixed_capacities)
    model.run()
    print('TESTS: Done running test simulation \n')

    # Run the tests
    print('TESTS: Starting tests\n---------------------')
    test_output_consistency(model, model_name, run_mode)
    # test_outputs_against_benchmark(model, model_name, run_mode,
    #                                baseload_integer, baseload_ramping)


def get_summary_outputs(model_name, run_mode, baseload_integer,
                        baseload_ramping, fixed_capacities,
                        time_start, time_end,
                        at_regional_level=False, save_csv=False):

    # Load time series data
    ts_data = models.load_time_series_data(model_name=model_name,
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc[time_start:time_end]

    start = time.time()
    if model_name == '1_region':
        model = models.OneRegionModel(ts_data, run_mode,
                                      baseload_integer, baseload_ramping,
                                      fixed_capacities,
                                      preserve_index=False)
        model.run()
        summary_outputs = model.get_summary_outputs()
    else:
        model = models.SixRegionModel(ts_data, run_mode,
                                      baseload_integer, baseload_ramping,
                                      fixed_capacities,
                                      preserve_index=False)
        model.run()
        summary_outputs = model.get_summary_outputs(at_regional_level)
    end = time.time()
    summary_outputs.loc['time'] = end - start

    if save_csv:
        raise NotImplementedError()

    return summary_outputs


def compare_summary_outputs():

    fixed_capacities = {}
    fixed_capacities['cap_baseload_total'] = 23.2123
    fixed_capacities['cap_peaking_total'] = 26.5267
    fixed_capacities['cap_wind_total'] = 23.2782

    start = '2015'
    end = '2017'

    run_dict_1 = {
        'model_name': '1_region',
        'run_mode': 'plan',
        'baseload_integer': True,
        'baseload_ramping': True,
        'fixed_capacities': None,
        'time_start': start,
        'time_end': end
    }

    run_dict_2 = {
        'model_name': '1_region',
        'run_mode': 'operate',
        'baseload_integer': True,
        'baseload_ramping': True,
        'fixed_capacities': fixed_capacities,
        'time_start': start,
        'time_end': end
    }

    summary_outputs_1 = get_summary_outputs(**run_dict_1)
    summary_outputs_2 = get_summary_outputs(**run_dict_2)
    summary_outputs = pd.merge(summary_outputs_1, summary_outputs_2,
                               left_index=True, right_index=True)
    summary_outputs.columns = ['run_1', 'run_2']
    summary_outputs['diff'] = summary_outputs['run_1'] \
                              - summary_outputs['run_2']
    print(summary_outputs)


def dev_test():

    fixed_capacities = {}

    # fixed_capacities['cap_baseload_total'] = 23.2124
    # fixed_capacities['cap_peaking_total'] = 26.5268
    # fixed_capacities['cap_wind_total'] = 23.2783

    fixed_capacities['cap_baseload_region1']             = 3.112262e+01
    fixed_capacities['cap_peaking_region1']              = 1.581766e+01
    fixed_capacities['cap_transmission_region1_region5'] = 5.534785e+00
    fixed_capacities['cap_transmission_region1_region2'] = 4.199222e+01
    fixed_capacities['cap_transmission_region1_region6'] = 0.000000e+00
    fixed_capacities['cap_wind_region2']                 = 2.241848e+00
    fixed_capacities['cap_transmission_region2_region3'] = 3.048031e+01
    fixed_capacities['cap_baseload_region3']             = 9.071644e+01
    fixed_capacities['cap_peaking_region3']              = 2.742114e+01
    fixed_capacities['cap_transmission_region3_region4'] = 8.815252e+01
    fixed_capacities['cap_transmission_region4_region5'] = 2.028268e+00
    fixed_capacities['cap_wind_region5']                 = 6.233972e+01
    fixed_capacities['cap_transmission_region5_region6'] = 4.129871e+01
    fixed_capacities['cap_baseload_region6']             = 0.000000e+00
    fixed_capacities['cap_peaking_region6']              = 4.129871e+01
    fixed_capacities['cap_wind_region6']                 = 0.000000e+00
    fixed_capacities['cap_baseload_total']               = 1.218391e+02
    fixed_capacities['cap_peaking_total']                = 8.453750e+01
    fixed_capacities['cap_wind_total']                   = 6.458156e+01
    fixed_capacities['cap_transmission_total']           = 4.189736e+02



    run_dict = {
        'model_name': '6_region',
        'run_mode': 'operate',
        'baseload_integer': False,
        'baseload_ramping': False,
        'fixed_capacities': fixed_capacities,
    }

    time_start, time_end = '2017-01', '2017-01'

    # summary_outputs = get_summary_outputs(**run_dict,
    #                                       time_start=time_start,
    #                                       time_end=time_end)
    # print(summary_outputs)

    run_tests(**run_dict)


if __name__ == '__main__':
    dev_test()
