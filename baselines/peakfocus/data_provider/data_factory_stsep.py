from torch.utils.data import DataLoader

try:
    from baselines.peakfocus.data_provider.data_loader_stsep import Dataset_STSEP_Mixed
except ImportError:
    from data_provider.data_loader_stsep import Dataset_STSEP_Mixed


def data_provider_stsep(args, flag):
    """PeakFocus data provider using STSEP split alignment."""
    Data = Dataset_STSEP_Mixed

    shuffle_flag = flag == 'train'
    drop_last = flag == 'train'
    batch_size = args.batch_size

    data_set = Data(args=args, flag=flag)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 4),
        drop_last=drop_last,
    )
    return data_set, data_loader
