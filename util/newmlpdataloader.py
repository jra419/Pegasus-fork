import csv
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from util.data_loader import CICIOT2022_rule, ISCXVPN_rule, PeerRush_rule


# Added to len_seq for total_len / min/max packet length (Ethernet + FCS overhead).
MLP_FRAME_OVERHEAD = 14
# Same IPD scaling as FlowDataset in util/data_loader.py.
MLP_IPD_SCALE = 10000

MLP_JSON_FEATURE_NAMES = (
    "ip_len",
    "total_len",
    "proto",
    "tos",
    "tcp_offset",
    "max_packet_len",
    "min_packet_len",
    "max_ipd",
    "min_ipd",
    "last ipd",
)

class mlpdataset(Dataset):
    def __init__(self, csv_file, modelname,args, istrained = True):
        self.data       = pd.read_csv(csv_file)
        labels          = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.labels     = labels
        features        = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        self.features   = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label       = self.labels[idx]
        features    = self.features[idx]

        return features, label

def binarize_and_expand(input_tensor,args):
    if args.dataset == "UNSWNB15":
        if args.onlypacket:
            bit_lengths = [4, 16, 8, 8]
        else:
            bit_lengths = [4, 16, 8, 8, 16, 16, 16, 16, 16]
    else:
        if args.onlypacket:
            bit_lengths = [4, 16, 8, 8, 8, 4]
        else:
            bit_lengths = [4, 16, 8, 8, 8, 4, 16, 16, 16, 16, 16]

    expanded_features = []

    for i, bit_len in enumerate(bit_lengths):
        # Extract the feature column and convert to integer type
        # Convert to int to apply bitwise operations
        feature = input_tensor[:, i].unsqueeze(1).int()

        # Create a mask to extract each bit using bitwise operations
        binary_feature = ((feature.unsqueeze(-1) & (1 << torch.arange(bit_len)))) > 0

        # Convert binary feature to float and map 0 to -1.0, and 1 to 1.0
        bits_tensor = binary_feature.float() * 2 - 1.0

        # Reshape bits_tensor from [N, 1, X] to [N, X]
        bits_tensor = bits_tensor.squeeze(1)

        # Add to the list of expanded features
        expanded_features.append(bits_tensor)

    # Concatenate all expanded binary features along the last dimension
    output_tensor = torch.cat(expanded_features, dim=1)

    return output_tensor

def new_mlp_dataloader(csv_file, model_name, args, batch_size=32, istrained=True, shuffle=True):
    dataset         = mlpdataset(csv_file, model_name,args,istrained = istrained)
    val_ratio       = 0.1
    known_loader    = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if istrained:
        labels = [dataset[i][1] for i in range(len(dataset))]  

        train_indices, val_indices = \
                train_test_split(range(len(dataset)),
                                 test_size=val_ratio,
                                 stratify=np.array(labels).squeeze())

        train_dataset   = Subset(dataset, train_indices)
        val_dataset     = Subset(dataset, val_indices)

        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader, known_loader
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader

# Map a JSON flow label to the integer class id used by MLP-B CSV training.
def mlp_map_flow_label(ins, dataset):
    label = ins["label"]

    if dataset == "CICIOT2022":
        if label not in CICIOT2022_rule:
            return None
        return CICIOT2022_rule[label]

    if dataset == "PeerRush":
        if label not in PeerRush_rule:
            return None
        return PeerRush_rule[label]

    if dataset == "ISCXVPN":
        if label not in ISCXVPN_rule:
            return None
        mapped = ISCXVPN_rule[label]
        if mapped == 0:
            return None
        return mapped - 1

    if isinstance(label, int):
        return label

    try:
        return int(label)
    except (TypeError, ValueError):
        return None

def mlp_compute_ipd_seq(ts_seq):
    ipd = [0.0]
    ipd.extend(ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq)))

    return ipd

def mlp_scaled_ipd_seq(ts_seq, ipd_scale=MLP_IPD_SCALE):
    return [round(delta * ipd_scale) for delta in mlp_compute_ipd_seq(ts_seq)]

# Build one 10-feature MLP-B vector for a single packet within a flow.
def mlp_feature_vector_from_flow(ins, packet_idx, frame_overhead=MLP_FRAME_OVERHEAD,
                                 ipd_scale=MLP_IPD_SCALE, cumulative_ipd=True):
    len_seq = ins["len_seq"]
    ipd_seq = mlp_scaled_ipd_seq(ins["ts_seq"], ipd_scale=ipd_scale)

    max_packet_len = max(len_seq) + frame_overhead
    min_packet_len = min(len_seq) + frame_overhead

    if cumulative_ipd and packet_idx >= 1:
        ipd_window = ipd_seq[1: packet_idx + 1]
    else:
        ipd_window = ipd_seq[1:]

    max_ipd = max(ipd_window) if ipd_window else 0
    min_ipd = min(ipd_window) if ipd_window else 0

    return [
        float(ins["ip_hl_seq"][packet_idx]),
        float(len_seq[packet_idx] + frame_overhead),
        float(ins["proto_seq"][packet_idx]),
        float(ins["ip_tos_seq"][packet_idx]),
        float(ins["tcp_off_seq"][packet_idx]),
        float(max_packet_len),
        float(min_packet_len),
        float(max_ipd),
        float(min_ipd),
        float(ipd_seq[packet_idx]),
    ]

# Return feature vectors (without labels) for every packet in one JSON flow.
def mlp_json_samples_from_flow(ins, args):
    required = (
        "len_seq",
        "ts_seq",
        "ip_hl_seq",
        "ip_tos_seq",
        "tcp_off_seq",
        "proto_seq",
    )

    for key in required:
        if key not in ins:
            raise KeyError(f"JSON flow missing required field {key!r}")

    n_packets = len(ins["len_seq"])

    if n_packets == 0:
        return []

    for key in required:
        if len(ins[key]) != n_packets:
            raise ValueError(f"JSON flow field {key!r} length mismatch")

    frame_overhead  = getattr(args, "mlp_frame_overhead", MLP_FRAME_OVERHEAD)
    ipd_scale       = getattr(args, "mlp_ipd_scale", MLP_IPD_SCALE)
    cumulative_ipd  = getattr(args, "mlp_cumulative_ipd", True)

    return [
        mlp_feature_vector_from_flow(ins, packet_idx, frame_overhead=frame_overhead,
                                     ipd_scale=ipd_scale, cumulative_ipd=cumulative_ipd)
        for packet_idx in range(n_packets)
    ]

class mlp_json_dataset(Dataset):
    def __init__(self, json_file, modelname, args, istrained=True):
        del modelname, istrained

        features    = []
        labels      = []

        with open(json_file) as fp:
            instances = json.load(fp)

        for ins in instances:
            label = mlp_map_flow_label(ins, args.dataset)

            if label is None:
                continue

            for feat in mlp_json_samples_from_flow(ins, args):
                features.append(feat)
                labels.append(label)

        if not features:
            raise ValueError(f"No MLP-B samples loaded from {json_file}")

        self.features   = torch.tensor(features, dtype=torch.float32)
        self.labels     = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def new_mlp_json_dataloader(json_file, model_name, args, batch_size=32, istrained=True,
                            shuffle=True):
    dataset         = mlp_json_dataset(json_file, model_name, args, istrained=istrained)
    val_ratio       = 0.1
    known_loader    = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    if istrained:
        labels = [dataset[i][1] for i in range(len(dataset))]

        train_indices, val_indices = \
                train_test_split(range(len(dataset)),
                                 test_size=val_ratio,
                                 stratify=np.array(labels).squeeze())

        train_dataset   = Subset(dataset, train_indices)
        val_dataset     = Subset(dataset, val_indices)

        train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader, known_loader

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader
