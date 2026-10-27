from array import array
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .pcap_feats import FEAT_COLS, iter_pkt_rows, read_labels

VAL_RATIO        = 0.1
FLOW_KEY_COLS    = ['ip_src', 'ip_dst', 'ip_proto', 'port_src', 'port_dst']

def default_labels_path(pcap_path):
    pcap_path = Path(pcap_path)

    return pcap_path.with_name(f'{pcap_path.stem}-labels.csv')

def _load_pcap(pcap_path, labels_path):
    labels_path = Path(labels_path) if labels_path else default_labels_path(pcap_path)

    if not labels_path.is_file():
        raise FileNotFoundError(f'Error: {labels_path} does not exist ')

    feats       = array('q')
    labels      = array('q')
    flow_keys   = []

    for row in iter_pkt_rows(pcap_path, read_labels(labels_path)):
        feats.extend(row[column] for column in FEAT_COLS)
        labels.append(row['label'])
        flow_keys.append(tuple(row[column] for column in FLOW_KEY_COLS))

    return (np.frombuffer(feats, dtype=np.int64).reshape(-1, len(FEAT_COLS)),
            np.frombuffer(labels, dtype=np.int64),
            flow_keys)

def load_pkts(path, labels_path=None, with_flow_keys=False):
    if Path(path).suffix.lower() in {'.pcap'}:
        feats, labels, flow_keys = _load_pcap(path, labels_path)
    else:
        columns     = FEAT_COLS + ['label'] + (FLOW_KEY_COLS if with_flow_keys else [])
        data        = pd.read_csv(path, usecols=list(dict.fromkeys(columns)))
        feats    = data[FEAT_COLS].to_numpy(dtype=np.int64)
        labels      = data['label'].to_numpy(dtype=np.int64)
        flow_keys   = (list(data[FLOW_KEY_COLS].itertuples(index=False, name=None))
                       if with_flow_keys else None)

    if (labels < 0).any():
        raise ValueError(f'Error: {path} has unlabeled packets (label < 0)')

    if with_flow_keys:
        return feats, labels, flow_keys

    return feats, labels

class PerPacketDataset(Dataset):
    def __init__(self, path, labels_path=None):
        feats, labels   = load_pkts(path, labels_path)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.feats      = torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]

def build_train_loaders(path, batch_size=32, shuffle=True, labels_path=None):
    dataset     = PerPacketDataset(path, labels_path)
    full_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_indices, val_indices = train_test_split(range(len(dataset)),
                                                  test_size=VAL_RATIO,
                                                  stratify=dataset.labels.numpy())

    train_loader    = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                                 shuffle=shuffle)
    val_loader      = DataLoader(Subset(dataset, val_indices), batch_size=batch_size,
                                 shuffle=shuffle)

    return train_loader, val_loader, full_loader

def build_eval_loader(path, batch_size=32, shuffle=False, labels_path=None):
    return DataLoader(PerPacketDataset(path, labels_path), batch_size=batch_size, shuffle=shuffle)
