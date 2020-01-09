"""Tests to check whether the models behave as required."""


import os
import pandas as pd
import time
import models
import pdb


# Install costs and generation costs. These should match the information
# provided in the model.yaml and techs.yaml files in the model definition
COSTS = pd.DataFrame(columns=['install', 'generation'])
COSTS.loc['baseload']                = [300., 0.005]
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


def test_output_consistency_1_region(model):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    costs (pandas DataFrame) : costs of model technologies

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
    for tech in ['baseload', 'peaking', 'wind']:
        cost_method1 = float(COSTS.loc[tech, 'install']
                             * out.loc['_'.join(('cap', tech, 'total'))])
        cost_method2 = corrfac * float(
            res.cost_investment[0].loc['::'.join(('region1', tech))]
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
                             * out.loc['_'.join(('gen', tech, 'total'))])
        cost_method2 = corrfac * float(res.cost_var[0].loc[
            '::'.join(('region1', tech))
        ].sum())
        if abs(cost_method1 - cost_method2) > 1:
            print('FAIL: {} generation costs do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
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


def test_output_consistency_6_region(model):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel

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
    for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP:
        cost_method1 = float(
            COSTS.loc[tech, 'install'] *
            out.loc['_'.join(('cap', tech, region))]
        )
        cost_method2 = corrfac * float(
            res.cost_investment[0].loc['::'.join((region, tech))]
        )
        if abs(cost_method1 - cost_method2) > 1:
            print('FAIL: {} install costs in {} do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, region, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if transmission installation costs are consistent
    for tech, region_a, region_b in \
        TRANSMISSION_REGION1TO5_TOP + TRANSMISSION_OTHER_TOP:
        cost_method1 = float(
            COSTS.loc[tech, 'install'] * out.loc[
                '_'.join(('cap_transmission', region_a, region_b))
            ]
        )
        cost_method2 = 2 * corrfac * \
            float(res.cost_investment[0].loc[
                ':'.join((region_a + ':', tech, region_b))
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
            * out.loc['_'.join(('gen', tech, region))]
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


def test_output_consistency(model, model_name):
    """Check if model outputs are internally consistent.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    model_name (str) : '1_region' or '6_region'
    summary_outputs (pandas DataFrame) : summary outputs, from
        model.get_summary_outputs(...)

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    # Run the consistency tests
    if model_name == '1_region':
        passing = test_output_consistency_1_region(model)
    if model_name == '6_region':
        passing = test_output_consistency_6_region(model)

    if passing:
        print('PASS: model outputs are consistent.')

    return passing


def test_outputs_against_benchmark(model,
                                   model_name,
                                   run_mode,
                                   baseload_type):
    """Test model outputs against benchmark.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    model_name (str) : '1_region' or '6_region'
    model_type (str) : 'plan' or 'operate'
    baseload_type (str) : 'continuous' or 'integer_ramp'
    """

    passing = True
    summary_outputs = model.get_summary_outputs()

    # Load benchmarks
    benchmark_outputs = pd.read_csv(
        os.path.join('benchmarks',
                     '_'.join((model_name, run_mode, baseload_type, '2017.csv'))),
        index_col=0
    )

    if float(abs(summary_outputs - benchmark_outputs).max()) > 1:
        print('FAIL: Model outputs do not match benchmark outputs!')
        print('Model outputs:')
        print(summary_outputs)
        print('')
        print('Benchmark outputs:')
        print(benchmark_outputs)
        passing = False

    if passing:
        print('PASS: model outputs match benchmark outputs.')

    return passing


def run_tests(model_name, run_mode, baseload_type):
    """ Run the tests.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    model_type (str) : 'plan' or 'operate'
    baseload_type (str) : 'continuous' or 'integer_ramp'
    """

    assert model_name in ['1_region', '6_region']
    assert run_mode in ['plan', 'operate']
    assert baseload_type in ['continuous', 'integer_ramp']

    ts_data = models.load_time_series_data(model_name=model_name,
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc['2017']    # Should match benchmarks

    # Run a simulation on which to run tests
    print('TESTS: Running test simulation...')
    if model_name == '1_region':
        model = models.OneRegionModel(run_mode, baseload_type, ts_data)
    # elif model_name == '6_region':
    #     model = models.SixRegionModel(model_type, ts_data)
    model.run()
    print('TESTS: Done running test simulation \n')

    # Run the tests
    print('TESTS: Starting tests\n---------------------')
    test_output_consistency(model, model_name)
    test_outputs_against_benchmark(model,
                                   model_name,
                                   run_mode,
                                   baseload_type)


def get_summary_outputs(model_name, run_mode, baseload_type, time_subset,
                        save_csv=False):

    # Load time series data
    ts_data = models.load_time_series_data(model_name=model_name,
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc[time_subset]

    # Set up and run model
    start = time.time()
    if model_name == '1_region':
        model = models.OneRegionModel(run_mode, baseload_type, ts_data)
    if model_name == '6_region':
        model = models.SixRegionModel(run_mode, baseload_type, ts_data)
    model.run()
    summary_outputs = model.get_summary_outputs()
    end = time.time()
    summary_outputs.loc['time'] = end - start

    if save_csv:
        summary_outputs.to_csv(
            '_'.join((model_name, run_mode, baseload_type, time_subset + '.csv'))
        )

    return summary_outputs


def compare_continuous_with_integer_ramping():
    summary_outputs_cont = get_summary_outputs(model_name='1_region',
                                               run_mode='operate',
                                               baseload_type='continuous',
                                               time_subset='2017')
    summary_outputs_disc = get_summary_outputs(model_name='1_region',
                                               run_mode='operate',
                                               baseload_type='integer_ramp',
                                               time_subset='2017')

    summary_outputs = pd.merge(summary_outputs_cont, summary_outputs_disc,
                               left_index=True, right_index=True)
    summary_outputs.columns = ['cont', 'disc']
    summary_outputs['diff'] = summary_outputs['cont'] - summary_outputs['disc']
    print(summary_outputs)

    # expected_cost = 0.005 * summary_outputs.loc['gen_baseload_total', 'output'] \
    #                 + 0.035 * summary_outputs.loc['gen_peaking_total', 'output'] \
    #                 + 6.0 * summary_outputs.loc['gen_unmet_total', 'output']

    # print(summary_outputs)
    # print('')
    # print(float(summary_outputs.loc['cost_total']), expected_cost)
    # print('')
    # print(float(summary_outputs.loc['cost_total']) /  expected_cost)


def dev_test():
    # Load time series data
    ts_data = models.load_time_series_data(model_name='1_region',
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc['2017-01-01']

        # locations:
        #     region1:
        #         techs:
        #             baseload:
        #                 constraints:
        #                     energy_cap_equals: 9  # GW
        #                     # units_equals: 3
        #                     # energy_cap_per_unit: 3
        #             peaking:
        #                 constraints:
        #                     energy_cap_equals: 0  # GW
        #             wind:
        #                 constraints:
        #                     resource_area_equals: 30  # GW
        #             unmet:
        #                 constraints:
        #                     energy_cap_equals: 1e10  # GW


    model = models.OneRegionModel(run_mode='operate',
                                  baseload_type='continuous',
                                  ts_data=ts_data)
    model.run()
    model.plot.timeseries(array='results')

    
if __name__ == '__main__':
    summary_outputs = get_summary_outputs(model_name='1_region',
                                          run_mode='operate',
                                          baseload_type='continuous',
                                          time_subset='2017-01-01')
    print(summary_outputs)
