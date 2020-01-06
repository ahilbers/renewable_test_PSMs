"""Tests to check whether the models behave as required."""


import numpy as np
import pandas as pd
import models
import pdb


def test_output_consistency(model, summary_outputs):
    """Check if model outputs are internally consistent."""

    passing = True

    out = summary_outputs
    res = model.results
    corrfac = 8760/model.num_timesteps    # For annualisation

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

    cost_total_method1 = 0

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
              '    manual: {}, model: {}'.format(
                  cost_total_method1, cost_total_method2))
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


def dev_test():
    """The development test function."""

    ts_data = pd.read_csv('data/demand_wind.csv', index_col=0)
    ts_data = ts_data.loc[:, ['demand_region5', 'wind_region5']]
    ts_data.index = pd.to_datetime(ts_data.index)
    ts_data.columns = ['demand', 'wind']
    
    model = models.OneRegionModel(model_type='LP',
                                  ts_data=ts_data.loc['2017'])
    model.run()
    summary_outputs = model.get_summary_outputs()
    print(summary_outputs)
    summary_outputs.to_csv('benchmarks/1_region_LP.csv')



if __name__ == '__main__':
    dev_test()
