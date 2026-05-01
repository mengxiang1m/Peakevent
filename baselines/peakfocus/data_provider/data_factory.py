from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Custom_Mixed, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'load_data': Dataset_Custom,  # Use Dataset_Custom for load_data
    'load_data_mixed': Dataset_Custom_Mixed,  # Use Dataset_Custom_Mixed for mixed input/target
    'ETTh2_mixed': Dataset_Custom_Mixed,
    'electricity_mixed': Dataset_Custom_Mixed,  # Electricity dataset
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        
        # Prepare common arguments
        dataset_args = {
            'args': args,
            'root_path': args.root_path,
            'data_path': args.data_path,
            'flag': flag,
            'size': [args.seq_len, args.label_len, args.pred_len],
            'features': args.features,
            'target': args.target,
            'timeenc': timeenc,
            'freq': freq,
            'seasonal_patterns': args.seasonal_patterns,
            'cycle': getattr(args, 'cycle', None)
        }
        
        # Add mixed dataset specific parameters if using load_data_mixed or ETTh2_mixed or electricity_mixed
        if args.data == 'load_data_mixed' or args.data == 'ETTh2_mixed' or args.data == 'electricity_mixed':
            dataset_args['input_col'] = getattr(args, 'input_col', 'value_60min')
            dataset_args['target_col'] = getattr(args, 'target_col', 'value_max')
            
            # External features (weather covariates) support
            if getattr(args, 'use_external_features', 0) and getattr(args, 'external_feature_cols', ''):
                ext_cols = [c.strip() for c in args.external_feature_cols.split(',') if c.strip()]
                dataset_args['external_feature_cols'] = ext_cols
                print(f"External features enabled: {ext_cols}")
            
            # Auto-detect date ranges for different datasets
            # Check if data_path contains ETT (ETTh1, ETTh2, etc.)
            if 'ETTh' in args.data_path or 'ETTm' in args.data_path:
                # ETT datasets use 2016-2018 date ranges
                # Use standard split: 12 months train, 4 months val, 4 months test (from ETTh benchmark)
                dataset_args['split_by_date_range'] = True
                dataset_args['train_date_range'] = ('2016-07', '2017-07')  # 12 months
                dataset_args['val_date_range'] = ('2017-08', '2017-11')     # 4 months
                dataset_args['test_date_range'] = ('2017-12', '2018-06')    # 7 months (to cover all data)
                print(f"Auto-detected ETT dataset, using date ranges: train=2016-07 to 2017-07, val=2017-08 to 2017-11, test=2017-12 to 2018-06")
            elif 'electricity' in args.data_path:
                # Electricity dataset: 2016-07 to 2019-07 (37 months total)
                # Use 6:2:2 split: 22 months train, 7 months val, 8 months test
                dataset_args['split_by_date_range'] = True
                dataset_args['train_date_range'] = ('2016-07', '2018-04')  # 22 months (60%)
                dataset_args['val_date_range'] = ('2018-05', '2018-11')     # 7 months (20%)
                dataset_args['test_date_range'] = ('2018-12', '2019-07')    # 8 months (20%)
                print(f"Auto-detected Electricity dataset, using date ranges: train=2016-07 to 2018-04, val=2018-05 to 2018-11, test=2018-12 to 2019-07")
            elif 'hf_load_data' in args.data_path:
                # HF load data uses 2021-2025 date ranges (original default)
                dataset_args['split_by_date_range'] = True
                dataset_args['train_date_range'] = ('2021-01', '2023-08')
                dataset_args['val_date_range'] = ('2023-09', '2024-08')
                dataset_args['test_date_range'] = ('2024-09', None)
                print(f"Auto-detected HF load dataset, using date ranges: train=2021-01 to 2023-08, val=2023-09 to 2024-08, test=2024-09 to end")
            else:
                # Use ratio-based split for unknown datasets
                dataset_args['split_by_date_range'] = False
                print(f"Unknown dataset pattern, using ratio-based split (0.7/0.1/0.2)")
        
        data_set = Data(**dataset_args)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
