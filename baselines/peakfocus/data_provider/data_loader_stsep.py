import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


def _parse_split_range(node) -> Tuple[int, int]:
    """Parse split range to inclusive [start, end]."""
    if isinstance(node, (list, tuple)) and len(node) >= 2:
        return int(node[0]), int(node[1])
    if isinstance(node, dict):
        if 'start' in node and 'end' in node:
            return int(node['start']), int(node['end'])
        if 'range' in node and isinstance(node['range'], (list, tuple)) and len(node['range']) >= 2:
            return int(node['range'][0]), int(node['range'][1])
    raise ValueError(f'Unsupported split range format: {node}')


def _extract_split_from_dataset_config(config_path: str, dataset_name: str) -> Dict[str, Tuple[int, int]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    ds_cfg = None
    if isinstance(cfg, dict):
        if 'datasets' in cfg and isinstance(cfg['datasets'], dict):
            ds_cfg = cfg['datasets'].get(dataset_name)
        if ds_cfg is None:
            ds_cfg = cfg.get(dataset_name)

    if ds_cfg is None:
        raise KeyError(f'dataset_name={dataset_name} not found in {config_path}')

    split_cfg = None
    if isinstance(ds_cfg, dict):
        split_cfg = ds_cfg.get('splits') or ds_cfg.get('split')
        if split_cfg is None and all(k in ds_cfg for k in ['train', 'val', 'test']):
            split_cfg = ds_cfg

    if split_cfg is None:
        raise KeyError(f'No split definitions found for dataset={dataset_name} in {config_path}')

    out = {}
    for split_name in ['train', 'val', 'test']:
        if split_name not in split_cfg:
            raise KeyError(f'Missing split `{split_name}` for dataset={dataset_name} in {config_path}')
        out[split_name] = _parse_split_range(split_cfg[split_name])
    return out


def _extract_split_from_csv(df_raw: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    if 'split' not in df_raw.columns:
        raise KeyError('`split` column not found in CSV for fallback split inference')

    split_series = df_raw['split'].astype(str).str.lower()

    aliases = {
        'train': {'train', 'tr'},
        'val': {'val', 'valid', 'validation', 'dev'},
        'test': {'test', 'te'},
    }

    out = {}
    for canonical, names in aliases.items():
        idx = np.where(split_series.isin(names))[0]
        if idx.size == 0:
            raise ValueError(f'Cannot infer split `{canonical}` from CSV `split` column')
        out[canonical] = (int(idx.min()), int(idx.max()))

    return out


class Dataset_STSEP_Mixed(Dataset):
    """
    PeakFocus-compatible dataset using STSEP split alignment.

    Output tuple per sample:
    - seq_x: [seq_len, 1]
    - seq_y: [label_len + pred_len, 1]
    - seq_x_mark: [seq_len, time_feat_dim + 1] (last dim is is_peak)
    - seq_y_mark: [label_len + pred_len, time_feat_dim + 1] (last dim is is_peak)
    """

    def __init__(
        self,
        args,
        flag: str = 'train',
        size=None,
        data_path: str = None,
        input_col: str = 'value_60min',
        target_col: str = 'value_max',
        input_date_col: str = 'date_60min',
        target_date_col: str = 'date_max',
        scale: bool = True,
        freq: str = 'h',
    ):
        assert flag in ['train', 'val', 'test']

        self.args = args
        self.flag = flag
        self.input_col = input_col
        self.target_col = target_col
        self.input_date_col = input_date_col
        self.target_date_col = target_date_col
        self.scale = scale
        self.freq = getattr(args, 'freq', freq)

        self.seq_len = int(getattr(args, 'seq_len', 96 if size is None else size[0]))
        self.label_len = int(getattr(args, 'label_len', 48 if size is None else size[1]))
        self.pred_len = int(getattr(args, 'pred_len', 96 if size is None else size[2]))
        self.window_stride = max(1, int(getattr(args, 'window_stride', 4)))

        self.root_path = getattr(args, 'root_path', '.')
        self.data_path = data_path or getattr(args, 'data_path', '')
        if not self.data_path:
            raise ValueError('data_path is required for Dataset_STSEP_Mixed')

        self.dataset_config = getattr(args, 'dataset_config', '')
        self.dataset_name = getattr(args, 'dataset_name', '')

        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()

        self.__read_data__()

    def _resolve_csv_path(self) -> str:
        if os.path.isabs(self.data_path):
            return self.data_path
        return os.path.join(self.root_path, self.data_path)

    def _load_split_boundaries(self, df_raw: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
        if self.dataset_config:
            try:
                return _extract_split_from_dataset_config(self.dataset_config, self.dataset_name)
            except Exception as e:
                print(f'[Dataset_STSEP_Mixed] Failed to read dataset_config, fallback to CSV split: {e}')

        return _extract_split_from_csv(df_raw)

    def _build_time_features(self, date_values: np.ndarray) -> np.ndarray:
        feats = time_features(pd.to_datetime(date_values), freq=self.freq)
        return feats.transpose(1, 0).astype(np.float32)

    def __read_data__(self):
        csv_path = self._resolve_csv_path()
        df_raw = pd.read_csv(csv_path)

        required_cols = [self.input_col, self.target_col]
        for col in required_cols:
            if col not in df_raw.columns:
                raise KeyError(f'Required column `{col}` not found in {csv_path}')

        if self.input_date_col not in df_raw.columns:
            if 'timestamp' in df_raw.columns:
                df_raw[self.input_date_col] = df_raw['timestamp']
            elif 'date' in df_raw.columns:
                df_raw[self.input_date_col] = df_raw['date']
            else:
                raise KeyError(f'Required column `{self.input_date_col}` not found in {csv_path}')

        if self.target_date_col not in df_raw.columns:
            if 'timestamp' in df_raw.columns:
                df_raw[self.target_date_col] = df_raw['timestamp']
            elif 'date' in df_raw.columns:
                df_raw[self.target_date_col] = df_raw['date']
            else:
                raise KeyError(f'Required column `{self.target_date_col}` not found in {csv_path}')

        if 'is_peak' not in df_raw.columns:
            df_raw['is_peak'] = 0.0

        splits = self._load_split_boundaries(df_raw)
        split_start, split_end = splits[self.flag]
        train_start, train_end = splits['train']

        border1 = int(split_start)
        border2 = int(split_end) + 1
        train_b1 = int(train_start)
        train_b2 = int(train_end) + 1

        input_values = df_raw[[self.input_col]].values.astype(np.float32)
        target_values = df_raw[[self.target_col]].values.astype(np.float32)

        if self.scale:
            self.scaler_input.fit(input_values[train_b1:train_b2])
            self.scaler_target.fit(target_values[train_b1:train_b2])
            input_scaled = self.scaler_input.transform(input_values).astype(np.float32)
            target_scaled = self.scaler_target.transform(target_values).astype(np.float32)
        else:
            input_scaled = input_values
            target_scaled = target_values

        is_peak_slice = df_raw['is_peak'].iloc[border1:border2].to_numpy(dtype=np.float32).reshape(-1, 1)

        input_time_feats = self._build_time_features(df_raw[self.input_date_col].iloc[border1:border2].values)
        target_time_feats = self._build_time_features(df_raw[self.target_date_col].iloc[border1:border2].values)

        self.data_x = input_scaled[border1:border2]
        self.data_y = target_scaled[border1:border2]
        self.data_stamp_input = np.concatenate([input_time_feats, is_peak_slice], axis=1).astype(np.float32)
        self.data_stamp_target = np.concatenate([target_time_feats, is_peak_slice], axis=1).astype(np.float32)

        self.border1 = border1
        self.border2 = border2

        max_start = self.border2 - self.seq_len - self.pred_len
        if max_start < self.border1:
            self.starts_global = np.array([], dtype=np.int64)
            self.starts_local = np.array([], dtype=np.int64)
        else:
            offset = (self.window_stride - (self.border1 % self.window_stride)) % self.window_stride
            first_aligned = self.border1 + offset
            self.starts_global = np.arange(first_aligned, max_start + 1, self.window_stride, dtype=np.int64)
            self.starts_local = self.starts_global - self.border1

    def __getitem__(self, index: int):
        s_begin = int(self.starts_local[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp_input[s_begin:s_end]
        seq_y_mark = self.data_stamp_target[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.starts_local)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if not self.scale:
            return data

        arr = np.asarray(data)
        original_shape = arr.shape

        if arr.ndim == 1:
            arr2d = arr.reshape(-1, 1)
        else:
            arr2d = arr.reshape(-1, original_shape[-1])

        inv = self.scaler_target.inverse_transform(arr2d)
        return inv.reshape(original_shape)
