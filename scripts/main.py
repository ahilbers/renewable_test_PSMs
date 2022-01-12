import os
import warnings
import logging
import json
import psm


def run_model(config: dict, logger: logging.Logger = None):
    '''Create and solve a power system model across a time period.
    
    Parameters:
    -----------
    config: dictionary with model, run and save properties
    logger: the logger to use in creating the model. If named 'psm', then it logs internal messages
        from the 'psm' package
    '''

    # If no logger is specified, include log messages from 'psm' package
    if logger is None:
        logger = logging.getLogger(name='psm')

    logger.info(f'Conducting psm simulation. Run config:\n{json.dumps(config, indent=4)}.')

    # Load time series data and slice to desired time period
    ts_data = psm.utils.load_time_series_data(model_name=config['model_name'])
    ts_data = ts_data.loc[config['ts_first_period']:config['ts_last_period']]
    logger.info(f'Loaded time series data, shape {ts_data.shape}.')
    logger.debug(f'Time series data:\n\n{ts_data}\n')

    # Get correct model class
    model_name = config['model_name']
    if model_name == '1_region':
        Model = psm.models.OneRegionModel
    elif model_name == '6_region':
        Model = psm.models.SixRegionModel
    else:
        raise ValueError(f'Invalid model name {model_name}. Options: `1_region`, `6_region`.')

    # Create and run the model
    logger.info('Creating model.')
    model = Model(
        ts_data=ts_data,
        run_mode=config['run_mode'],
        baseload_integer=config['baseload_integer'],
        baseload_ramping=config['baseload_ramping'],
        allow_unmet=config['allow_unmet'],
        fixed_caps=config['fixed_caps'],
        extra_override=config['extra_override']
    )
    logger.info('Done creating model.')

    logger.info('Running model to determine optimal solution.')
    model.run()
    logger.info('Done running model.')
    logger.info(f'Summary model outputs:\n\n{model.get_summary_outputs()}\n')

    # Save outputs to file
    output_save_dir = config['output_save_dir']
    model.get_summary_outputs().to_csv(f'{output_save_dir}/summary_outputs.csv')
    logger.info(f'Saved summary model results to `{output_save_dir}`.')
    if config['save_full_model']:
        full_model_results_save_dir = f'{output_save_dir}/full_model_results'
        model.to_csv(full_model_results_save_dir)
        logger.info(f'Saved full model results to `{full_model_results_save_dir}`.')


def main():

    '''
    Info on run_config keys:
    ------------------------
    model_name (str) : '1_region' or '6_region'
    ts_first_period (str) : first period of time series, slice, e.g. '2017-06-08'
    ts_last_period (str) : last period of time series slice, e.g. '2017-06-15'
    run_mode (str) : 'plan' or 'operate': whether model determines optimal generation and
        transmission capacities or optimises operation with a fixed setup
    baseload_integer (bool) : baseload integer capacity constraint (units of 3GW)
    baseload_ramping (bool) : baseload ramping constraint
    allow_unmet (bool) : allow unmet demand in 'plan' mode (always allowed in operate mode)
    fixed_caps (dict[str, float]) : fixed generation and transmission capacities. See
        `tutorial.ipynb` for an example.
    extra_override (str) : name of additional override, should be defined in relevant `model.yaml`
    output_save_dir (str) : name of directory where outputs are saved
    save_full_model (bool) : save all model properies and results in addition to summary outputs
    '''

    run_config = {
        'model_name': '1_region',
        'ts_first_period': '2017-06-01',
        'ts_last_period': '2017-06-07',
        'run_mode': 'plan',
        'baseload_integer': False,
        'baseload_ramping': False,
        'allow_unmet': False,
        'fixed_caps': None,
        'extra_override': None,
        'output_save_dir': 'outputs',
        'save_full_model': True,
        'logging_level': 'INFO'
    }

    # Create directory where the logs and outputs are saved
    output_save_dir = run_config['output_save_dir']
    os.mkdir(output_save_dir)

    # Log from 'psm' package, ignore warnings like 'setting depreciation rate as 1/lifetime'
    logger = psm.utils.get_logger(name='psm', run_config=run_config)
    warning_message_to_ignore = '.*\n.*setting depreciation rate as 1/lifetime.*'
    warnings.filterwarnings(action='ignore', message=warning_message_to_ignore)

    run_model(config=run_config, logger=logger)


if __name__ == '__main__':
    main()
