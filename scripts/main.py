import psm


def main():
    ts_data = psm.load_time_series_data('1_region')
    ts_data = ts_data.loc['2017-06-08':'2017-06-15']

    model = psm.OneRegionModel(ts_data=ts_data, run_mode='plan')
    model.run()
    print(model.get_summary_outputs())


if __name__ == '__main__':
    main()
