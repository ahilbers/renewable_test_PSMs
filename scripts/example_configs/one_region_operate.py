config_one_region_operate = {
    'model_name': '1_region',
    'ts_first_period': '2017-06-01',
    'ts_last_period': '2017-06-07',
    'run_mode': 'operate',
    'baseload_integer': False,
    'baseload_ramping': False,
    'allow_unmet': True,
    'fixed_caps': {
        'cap_baseload_total': 10.,
        'cap_peaking_total': 20.,
        'cap_wind_total': 50.,
        'cap_solar_total': 50.,
        'cap_storage_power_total': 10.,
        'cap_storage_energy_total': 50.
    },
    'extra_override': None,
    'output_save_dir': 'outputs',
    'save_full_model': True,
    'log_level_file': 'DEBUG',  # logging level for log file
    'log_level_stdout': 'INFO',  # logging level for terminal
}