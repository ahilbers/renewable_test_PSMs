import os
import warnings
import logging
import psm


def get_logger(run_config: dict):
    """Get a logger to use throughout the codebase.
    
    Parameters:
    -----------
        config: dictionary with model, run and save properties
    """

    output_save_dir = run_config['output_save_dir']

    # Create the master logger and formatter
    logger = logging.getLogger(name='model_run')
    logger.setLevel(level=getattr(logging, run_config['logging_level']))
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-8s - %(name)s - %(filename)s - %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )

    # Create two handlers: one writes to a log file, the other to stdout
    logger_file = logging.FileHandler(f'{output_save_dir}/model_run.log')
    logger_file.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=logger_file)
    logger_stdout = logging.StreamHandler()
    logger_stdout.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=logger_stdout)

    return logger


def run_model(config: dict):
    '''Create and solve a power system model across a time period.
    
    Parameters:
    -----------
        config: dictionary with model, run and save properties
    '''

    # Load time series data and slice to desired time period
    ts_data = psm.load_time_series_data(model_name=config['model_name'])
    ts_data = ts_data.loc[config['ts_first_period']:config['ts_last_period']]

    # Get correct model class
    model_name = config['model_name']
    if model_name == '1_region':
        Model = psm.OneRegionModel
    elif model_name == '6_region':
        Model = psm.SixRegionModel
    else:
        raise ValueError(f'Invalid model name {model_name}. Options: `1_region`, `6_region`.')

    # Create and run the model
    model = Model(
        ts_data=ts_data,
        run_mode=config['run_mode'],
        baseload_integer=config['baseload_integer'],
        baseload_ramping=config['baseload_ramping'],
        allow_unmet=config['allow_unmet'],
        fixed_caps=config['fixed_caps'],
        extra_override=config['extra_override']
    )
    # model.run()

    # # Save outputs to file
    # output_save_dir = config['output_save_dir']
    # model.get_summary_outputs().to_csv(f'{output_save_dir}/summary_outputs.csv')
    # if config['save_full_model']:
    #     model.to_csv(f'{output_save_dir}/full_model_results')


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
        'ts_first_period': '2017-06-08',
        'ts_last_period': '2017-06-15',
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

    logger = get_logger(run_config=run_config)

    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.critical('critical')
    

    

    # Ignore warnings of the form `setting depreciation rate as 1/lifetime.`
    warnings.filterwarnings(action='ignore', message='.*\n.*setting depreciation rate.*')

    # run_model(config=run_config)


if __name__ == '__main__':

    # TODO: Delete!!
    if os.path.exists('outputs'):
        os.rmdir('outputs')

    main()
