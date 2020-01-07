"""Tests to check whether the models behave as required."""


import os
import numpy as np
import pandas as pd
import models
import pdb


def test_output_consistency(model_name, model_type):
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

    # Install costs, generation costs, and carbon emissions. These
    # should match the information provided in the model.yaml and
    # techs.yaml files in the model definition
    costs = pd.DataFrame(columns=['install', 'generation'])
    costs.loc['baseload']                = [300., 0.005]
    costs.loc['peaking']                 = [100., 0.035]
    costs.loc['wind']                    = [100., 0.   ]
    costs.loc['unmet']                   = [  0., 6.   ]
    costs.loc['transmission_other']      = [100., 0.   ]
    costs.loc['transmission_region1to5'] = [150., 0.   ]

    assert model_name in ['1_region', '6_region'], \
        'Admissible model names: 1_region and 6_region'
    assert model_type in ['LP', 'MILP'], \
        'Admissible model types: LP and MILP'

    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.loc['2017']

    if model_name == '1_region':
        passing = test_output_consistency_1_region(model,
                                                   summary_outputs,
                                                   costs)
    if model_name == '6_region':
        passing = test_output_consistency_6_region(model,
                                                   summary_outputs,
                                                   costs)

    if passing:
        print('PASS: model outputs are consistent.')

    return passing


def test_output_consistency_1_region(model, summary_outputs, costs):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    summary_outputs (pandas DataFrame) : summary outputs, from
        model.get_summary_outputs(...)
    costs (pandas DataFrame) : costs of model technologies

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = summary_outputs
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        try:
            cost_method1 = float(
                costs.loc[tech, 'install'] *
                out.loc['_'.join(('cap', tech, 'total'))]
            )
            cost_method2 = corrfac * float(
                res.cost_investment[0].loc['::'.join(('region1', tech))]
            )
            if abs(cost_method1 - cost_method2) > 1:
                print('FAIL: {} install costs do not match!\n'
                      '    manual: {}, model: {}'.format(
                          tech, cost_method1, cost_method2))
                passing = False
            cost_total_method1 += cost_method1
        except KeyError:
            pass

    # Test if generation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        try:
            cost_method1 = float(
                costs.loc[tech, 'generation'] *
                out.loc['_'.join(('gen', tech, 'total'))]
            )
            cost_method2 = corrfac * float(res.cost_var[0].loc[
                '::'.join(('region1', tech))
            ].sum())
            if abs(cost_method1 - cost_method2) > 1:
                print('FAIL: {} generation costs do not match!\n'
                      '    manual: {}, model: {}'.format(
                          tech, cost_method1, cost_method2))
                passing = False
            cost_total_method1 += cost_method1
        except KeyError:
            pass

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


def test_output_consistency_6_region(model, summary_outputs, costs):
    """Check if model outputs are internally consistent for 6 region model.

    Parameters:
    -----------
    model (calliope.Model) : instance of OneRegionModel or SixRegionModel
    summary_outputs (pandas DataFrame) : summary outputs, from
        model.get_summary_outputs(...)
    costs (pandas DataFrame) : costs of model technologies

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True
    cost_total_method1 = 0

    out = summary_outputs
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

    # Test if generation technology installation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        for region in ['region{}'.format(i+1) for i in range(6)]:
            try:
                cost_method1 = float(
                    costs.loc[tech, 'install'] *
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
            except KeyError:
                pass

    # Test if transmission installation costs are consistent
    for tech in ['transmission_other', 'transmission_region1to5']:
        for regionA in ['region{}'.format(i+1) for i in range(6)]:
            for regionB in ['region{}'.format(i+1) for i in range(6)]:
                try:
                    cost_method1 = float(
                        costs.loc[tech, 'install'] * out.loc[
                            '_'.join(('cap_transmission', regionA, regionB))
                        ]
                    )
                    cost_method2 = 2 * corrfac * \
                        float(res.cost_investment[0].loc[
                            ':'.join((regionA + ':', tech, regionB))
                        ])
                    if abs(cost_method1 - cost_method2) > 1:
                        print('FAIL: {} install costs from {} to {} do '
                              'not match!\n'
                              '    manual: {}, model: {}'.format(
                                  tech, regionA, regionB,
                                  cost_method1, cost_method2))
                        passing = False
                    cost_total_method1 += cost_method1
                except KeyError:
                    pass

    # Test if generation costs are consistent
    for tech in ['baseload', 'peaking', 'wind', 'unmet']:
        for region in ['region{}'.format(i+1) for i in range(6)]:
            try:
                cost_method1 = float(
                    costs.loc[tech, 'generation'] *
                    out.loc['_'.join(('gen', tech, region))]
                )
                cost_method2 = corrfac * float(res.cost_var[0].loc[
                    '::'.join((region, tech))
                ].sum())
                if abs(cost_method1 - cost_method2) > 1:
                    print('FAIL: {} generation costs in {} do not match!\n'
                          '    manual: {}, model: {}'.format(
                              tech, region, cost_method1, cost_method2))
                    passing = False
                cost_total_method1 += cost_method1
            except KeyError:
                pass

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


def test_output_against_benchmark(model_name, model_type, summary_outputs):
    """Test whether model outputs match benchmark values.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    model_type (str) : 'LP' or 'MILP'
    summary_outputs (pandas DataFrame) : summary outputs, from
        model.get_summary_outputs(...)

    Returns:
    --------
    passing: True if test is passed, False otherwise
    """

    passing = True

    assert model_name in ['1_region', '6_region'], \
        'Admissible model names: 1_region and 6_region'
    assert model_type in ['LP', 'MILP'], \
        'Admissible model types: LP and MILP'

    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.loc['2017']

    benchmark_outputs = pd.read_csv(
        os.path.join('benchmarks',
                     '_'.join((model_name, model_type, '2017.csv'))),
        index_col=0
    )

    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5']]
        ts_data.columns = ['demand', 'wind']
        model = models.OneRegionModel(model_type, ts_data)
    elif model_name == '6_region':
        model = models.SixRegionModel(model_type, ts_data)
    model.run()
    summary_outputs = model.get_summary_outputs()

    if float(abs(summary_outputs - benchmark_outputs).max()) > 1:
        print('FAIL: Model outputs do not match benchmark outputs!')
        passing = False

    if passing:
        print('PASS: model outputs match benchmark outputs.')

    return passing


def run_tests(model_name, model_type):
    """ Run the tests.

    Parameters:
    -----------
    model_name (str) : '1_region' or '6_region'
    model_type (str) : 'LP' or 'MILP'
    """

    assert model_name in ['1_region', '6_region'], \
        'Admissible model names: 1_region and 6_region'
    assert model_type in ['LP', 'MILP'], \
        'Admissible model types: LP and MILP'

    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data = ts_data.loc['2017']

    benchmark_outputs = pd.read_csv(
        os.path.join('benchmarks',
                     '_'.join((model_name, model_type, '2017.csv'))),
        index_col=0
    )

    if model_name == '1_region':
        ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5']]
        ts_data.columns = ['demand', 'wind']
        model = models.OneRegionModel(model_type, ts_data)
    elif model_name == '6_region':
        model = models.SixRegionModel(model_type, ts_data)
    model.run()
    summary_outputs = model.get_summary_outputs()

    if float(abs(summary_outputs - benchmark_outputs).max()) > 1:
        print('FAIL: Model outputs do not match benchmark outputs!')
        passing = False

    if passing:
        print('PASS: model outputs match benchmark outputs.')

    return passing    


if __name__ == '__main__':
    run_tests(model_name='1_region', model_type='LP')
