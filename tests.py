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
            if abs(cost_method1 - cost_method2) > 0.1:
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
        if abs(cost_method1 - cost_method2) > 0.1:
            print('FAIL: {} generation costs do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 0.1:
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
    if abs(generation_total - demand_total) > 0.1:
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
            if abs(cost_method1 - cost_method2) > 0.1:
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
            if abs(cost_method1 - cost_method2) > 0.1:
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
        if abs(cost_method1 - cost_method2) > 0.1:
            print('FAIL: {} generation costs in {} do not match!\n'
                  '    manual: {}, model: {}'.format(
                      tech, region, cost_method1, cost_method2))
            passing = False
        cost_total_method1 += cost_method1

    # Test if total costs are consistent
    if run_mode == 'plan':
        cost_total_method2 = corrfac * float(res.cost.sum())
        if abs(cost_total_method1 - cost_total_method2) > 0.1:
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
    if abs(generation_total - demand_total) > 0.1:
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


def get_test_fixed_caps_override_dict(model_name):
    """Create test fixed capacities and override dict."""

    attributes = {'baseload': 'energy_cap_equals',
                  'peaking': 'energy_cap_equals',
                  'wind': 'resource_area_equals'}

    # Create test versions of an override_dict and fixed capacities
    fixed_caps = {}
    o_dict = {}
    i = 1
    if model_name == '1_region':
        for tech in ['baseload', 'peaking', 'wind']:
            attribute = attributes[tech]
            o_dict['locations.region1.techs.{}.constraints.{}'.
                   format(tech, attribute)] = 10*i
            fixed_caps['cap_{}_total'.format(tech)] = 10*i
            i += 1
    if model_name == '6_region':
        for tech, region in BASELOAD_TOP + PEAKING_TOP + WIND_TOP:
            attribute = attributes[tech]
            o_dict['locations.{}.techs.{}.constraints.{}'.
                   format(region, tech, attribute)] = 10*i
            fixed_caps['cap_{}_{}'.format(tech, region)] = 10*i
            i += 1
        for transmission_type, region_a, region_b in \
            TRANSMISSION_REGION1TO5_TOP + TRANSMISSION_OTHER_TOP:
            i = int(region_a[-1] + region_b[-1])
            o_dict['links.{},{}.techs.{}.constraints.energy_cap_equals'.
                   format(region_a, region_b, transmission_type)] = i
            fixed_caps['cap_transmission_{}_{}'.
                       format(region_a, region_b)] = i

    return fixed_caps, o_dict


def test_override_dict(model_name):
    """Test if the override dictionary is working properly"""

    passing = True
    fixed_caps, o_dict_1 = get_test_fixed_caps_override_dict(model_name)

    # Test if override dictionary created by function is correct
    o_dict_2 = models.get_cap_override_dict(model_name, fixed_caps)
    if o_dict_1 != o_dict_2:
        print('FAIL: Override dictionary does not match!\n'
              '    Problem keys:')
        for key in o_dict_1:
            try:
                if o_dict_1[key] != o_dict_2[key]:
                    print(key)
            except KeyError:
                print(key)
        passing = False

    if passing:
        print('PASS: override_dictionary is created properly.')

    return passing


def run_model_tests(model_name, run_mode, baseload_integer, baseload_ramping):
    """ Run tests to see if models give the expected outputs."""

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

    if run_mode == 'operate':
        fixed_caps, _ = get_test_fixed_caps_override_dict(model_name)
    else:
        fixed_caps = None

    model = model(ts_data, run_mode, baseload_integer, baseload_ramping,
                  fixed_caps)
    model.run()
    print('TESTS: Done running test simulation \n')

    # Run the tests
    print('TESTS: Starting tests\n---------------------')
    test_output_consistency(model, model_name, run_mode)
    test_outputs_against_benchmark(model, model_name, run_mode,
                                   baseload_integer, baseload_ramping)


def get_summary_outputs(model_name, run_mode, baseload_integer,
                        baseload_ramping, fixed_caps,
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
                                      fixed_caps, preserve_index=False)
        model.run()
        summary_outputs = model.get_summary_outputs()
    else:
        model = models.SixRegionModel(ts_data, run_mode,
                                      baseload_integer, baseload_ramping,
                                      fixed_caps, preserve_index=False)
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


def get_quick_outputs():

    run_dict = {'model_name': '6_region',
                'run_mode': 'operate',
                'baseload_integer': False,
                'baseload_ramping': False,
                'fixed_caps': None,
                'time_start': '2017-01',
                'time_end': '2017-01'}

    if run_dict['run_mode'] == 'operate':
        fixed_caps, _ = get_test_fixed_caps_override_dict(
            model_name=run_dict['model_name'])
        run_dict['fixed_caps'] = fixed_caps

    summary_outputs = get_summary_outputs(**run_dict)
    print(summary_outputs)


def dev_test():
    ts_data = models.load_time_series_data(model_name='6_region',
                                           demand_region='region5',
                                           wind_region='region5')
    ts_data = ts_data.loc['2017-01']    # Should match benchmarks

    model = models.SixRegionModel(ts_data,
                                  run_mode='operate',
                                  baseload_integer=False,
                                  baseload_ramping=False,
                                  fixed_capacities=None)

    model.run()
    pdb.set_trace()

if __name__ == '__main__':
    dev_test()
