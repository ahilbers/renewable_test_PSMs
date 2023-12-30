import os
import shutil
import warnings
import logging
import json
import psm
import example_configs


def run_model(config: dict, logger: logging.Logger = None):
    '''Create and solve a power system model across a time period.

    Parameters:
    -----------
    config: dictionary with model, run and save properties
    logger: the logger to use in creating the model. If named 'psm', then it logs internal messages
        from the 'psm' package
    '''

    output_save_dir = config['output_save_dir']

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

    # Make plot and save to file
    logger.info(f'Saving plot to file `{output_save_dir}/plot.html`.')
    model.plot_timeseries()
    shutil.copy('temp_plot.html', f'{output_save_dir}/plot.html')  # Copy file to output directory

    # Save outputs to file
    model.get_summary_outputs().to_csv(f'{output_save_dir}/summary_outputs.csv')
    logger.info(f'Saved summary model results to `{output_save_dir}`.')
    model.get_timeseries_outputs().to_csv(f'{output_save_dir}/timeseries_outputs.csv')
    logger.info(f'Saved time series model results to `{output_save_dir}`.')
    if config['save_full_model']:
        full_model_results_save_dir = f'{output_save_dir}/full_model_results'
        model.to_csv(full_model_results_save_dir)
        logger.info(f'Saved full model results to `{full_model_results_save_dir}`.')


def main():
    '''Create, solve and analyse model.'''

    # config_*, with * = one_region_operate, one_region_plan, six_region_operate, six_region_plan
    # are example configurations
    run_config = example_configs.config_one_region_operate

    # Create directory where the logs and outputs are saved
    output_save_dir = run_config['output_save_dir']
    if os.path.exists(output_save_dir):
        print(f'Output directory `{output_save_dir}` already exists. Deleting old version.')
        shutil.rmtree(output_save_dir)
        os.mkdir(output_save_dir)
    else:
        os.mkdir(output_save_dir)

    # Log from 'psm' package, ignore warnings like 'setting depreciation rate as 1/lifetime'
    logger = psm.utils.get_logger(name='psm', run_config=run_config)
    warning_message_to_ignore = '.*\n.*setting depreciation rate as 1/lifetime.*'
    warnings.filterwarnings(action='ignore', message=warning_message_to_ignore)

    run_model(config=run_config, logger=logger)


if __name__ == '__main__':
    main()
