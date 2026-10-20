import json
import copy
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from .dataset_labels_conf import CICIOT2022_rule, ISCXVPN_rule, PeerRush_rule


class FlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args):
        super().__init__()

        self.flows          = []
        self.window_size    = window_size

        with open(filename) as fp:
            instances = json.load(fp)

        for ins in instances:
            if args.dataset == "CICIOT2022":
                ins['label'] = CICIOT2022_rule[ins['label']]
            elif args.dataset == "PeerRush":
                ins['label'] = PeerRush_rule[ins['label']]
            elif args.dataset == "ISCXVPN":
                ins['label'] = ISCXVPN_rule[ins['label']]
            if args.dataset == "ISCXVPN" and ins['label'] == 0:
                continue

            label = ins['label'] - 1 if args.dataset == "ISCXVPN" else ins['label']

            len_seq         = ins['len_seq']
            real_len_seq    = copy.deepcopy(len_seq)

            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)

            ts_seq  = ins['ts_seq']
            ipd_seq = [0]

            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])

            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]

            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(round(ipd_seq[i] * 10000), ipd_vocab - 1)
                assert ipd_seq[i] >= 0

            x = 4096

            if len(len_seq) > x:
                len_seq         = len_seq[:x]
                ipd_seq         = ipd_seq[:x]
                real_len_seq    = real_len_seq[:x]
                real_ipd_seq_us = real_ipd_seq_us[:x]

            combined_seq = torch.tensor([len_seq, ipd_seq])
            flow_packets = len(len_seq)

            if flow_packets >= self.window_size:
                segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]

                for idx in segs_idx:
                    seg_len         = torch.LongTensor(len_seq[idx: idx + self.window_size])
                    seg_ipd         = torch.LongTensor(ipd_seq[idx: idx + self.window_size])
                    combined_seg    = torch.stack([seg_len, seg_ipd], dim=-1)

                    self.flows.append({'x': combined_seg,
                                       'label': label})

            else:
                raise Exception('Flow packets < window size!!!')

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]

        return flow['x'], flow['label']

def _ae_is_benign_flow(ins, dataset):
    if dataset == "CICIOT2022":
        return CICIOT2022_rule.get(ins['label'], -1) == 0
    if dataset == "ISCXVPN":
        return ISCXVPN_rule.get(ins['label'], -1) != 0
    if dataset == "PeerRush":
        return PeerRush_rule.get(ins['label'], -1) == 0

    return True

def _ae_preprocess_flow(ins, len_vocab, ipd_vocab, max_packets=4096):
    len_seq = ins['len_seq']

    for i in range(len(len_seq)):
        len_seq[i] = min(len_seq[i], len_vocab - 1)

    ts_seq  = ins['ts_seq']
    ipd_seq = [0]
    ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])

    for i in range(len(ipd_seq)):
        ipd_seq[i] = min(round(ipd_seq[i] * 10000), ipd_vocab - 1)
        assert ipd_seq[i] >= 0

    if len(len_seq) > max_packets:
        len_seq = len_seq[:max_packets]
        ipd_seq = ipd_seq[:max_packets]

    return len_seq, ipd_seq

def _ae_append_windows(flows, len_seq, ipd_seq, window_size, label):
    flow_packets = len(len_seq)

    if flow_packets < window_size:
        raise Exception('Error: Flow packets < window size')

    for idx in range(0, flow_packets - window_size + 1):
        seg_len         = torch.LongTensor(len_seq[idx: idx + window_size])
        seg_ipd         = torch.LongTensor(ipd_seq[idx: idx + window_size])
        combined_seg    = torch.stack([seg_len, seg_ipd], dim=-1)

        flows.append({'x': combined_seg, 'label': label})

class aeTrainDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args):
        super().__init__()

        self.flows          = []
        self.window_size    = window_size

        with open(filename) as fp:
            instances = json.load(fp)

        for ins in instances:
            if not _ae_is_benign_flow(ins, args.dataset):
                continue

            len_seq, ipd_seq = _ae_preprocess_flow(ins, len_vocab, ipd_vocab)

            _ae_append_windows(self.flows, len_seq, ipd_seq, self.window_size, label=0)

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]

        return flow['x'], flow['label']

class aeEvalDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, window_size, dataset, filenames):
        super().__init__()

        self.flows          = []
        self.window_size    = window_size

        for source_name, filename in filenames.items():
            with open(filename) as fp:
                instances = json.load(fp)

            is_attack_source = source_name in ("FLOOD", "DDOS", "FUZZ", "test")

            for ins in instances:
                if is_attack_source:
                    binary_label = 1
                else:
                    if not _ae_is_benign_flow(ins, dataset):
                        continue
                    binary_label = 0

                len_seq, ipd_seq = _ae_preprocess_flow(ins, len_vocab, ipd_vocab)
                _ae_append_windows(self.flows, len_seq, ipd_seq, self.window_size, binary_label)

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]

        return flow['x'], flow['label']

class newencoderDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, window_size, filenames):
        super().__init__()

        self.flows          = []
        self.window_size    = window_size

        # 加载所有数据集并统一
        for dataset_name, filename in filenames.items():
            with open(filename) as fp:
                instances = json.load(fp)

            for ins in instances:
                if dataset_name == "FLOOD" \
                        or dataset_name == "DDOS" \
                        or dataset_name == "FUZZ" \
                        or dataset_name == "test":
                    ins['label'] = 1
                else:
                    ins['label'] = 0

                len_seq = ins['len_seq']

                # Truncate packet length
                len_seq = [min(i, len_vocab - 1) for i in len_seq]
                ts_seq  = ins['ts_seq']
                ipd_seq = [0] + [ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))]

                # Truncate ipd (malware uses 1e3, benign uses 1e4 — matches pegasus-repo)
                if ins['label'] == 1:
                    ipd_seq = [min(round(i * 1e3), ipd_vocab - 1) for i in ipd_seq]
                else:
                    ipd_seq = [min(round(i * 1e4), ipd_vocab - 1) for i in ipd_seq]

                # Truncate the flow
                if len(len_seq) > 4096:
                    len_seq = len_seq[:4096]
                    ipd_seq = ipd_seq[:4096]

                # Combine sequences
                combined_seq = torch.tensor([len_seq, ipd_seq])

                # Slice into segments
                flow_packets = len(len_seq)

                if flow_packets >= self.window_size:
                    segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]

                    for idx in segs_idx:
                        seg_len         = torch.LongTensor(len_seq[idx: idx + self.window_size])
                        seg_ipd         = torch.LongTensor(ipd_seq[idx: idx + self.window_size])
                        combined_seg    = torch.stack([seg_len, seg_ipd], dim=-1)

                        self.flows.append({'x': combined_seg,
                                           'label': ins['label']})

                else:
                    raise Exception('Flow packets < window size!!!')

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]

        return flow['x'], flow['label']

class bigcnnFlowDataset(Dataset):
    def __init__(self, len_vocab, ipd_vocab, filename, window_size, args):
        super().__init__()
        self.flows = []
        self.window_size = window_size

        with open(filename) as fp:
            instances = json.load(fp)
        for ins in instances:
            if args.dataset == "CICIOT2022":
                ins['label'] = CICIOT2022_rule[ins['label']]
            elif args.dataset == "PeerRush":
                ins['label'] = PeerRush_rule[ins['label']]
            elif args.dataset == "ISCXVPN":
                ins['label'] = ISCXVPN_rule[ins['label']]

            if args.dataset == "ISCXVPN" and ins['label'] == 0:
                continue
            # 如果是 ISCXVPN 数据集，调整标签值
            label = ins['label'] - 1 if args.dataset == "ISCXVPN" else ins['label']

            len_seq = ins['len_seq']
            tos_seq = ins['ip_tos_seq']
            ip_hl_seq = ins['ip_hl_seq']
            #ip_ttl_seq =  ins['ip_ttl_seq']
            #ip_tos_seq =  ins['ip_tos_seq']
            tcp_off_seq = ins['tcp_off_seq']
            #tcp_win_seq =  ins['tcp_win_seq']
            real_len_seq = copy.deepcopy(len_seq)
            payload_sequences = ins['payload_sequences']
            #print(len(len_seq))
            # Truncate the packet length
            for i in range(len(len_seq)):
                len_seq[i] = min(len_seq[i], len_vocab - 1)

            ts_seq = ins['ts_seq']
            ipd_seq = [0]
            ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])
            real_ipd_seq_us = [i * 1e6 for i in ipd_seq]
            # Truncate the ipd, unit: 16384 ns
            for i in range(len(ipd_seq)):
                ipd_seq[i] = min(round(ipd_seq[i] * 10000), ipd_vocab - 1)
                #ipd_seq[i] = min(ipd_seq[i] * 1e9 // 16384, ipd_vocab - 1)
                assert ipd_seq[i] >= 0

            x = 4096

            # Truncate the flow
            if len(len_seq) > x:
                len_seq = len_seq[:x]
                ipd_seq = ipd_seq[:x]
                real_len_seq = real_len_seq[:x]
                real_ipd_seq_us = real_ipd_seq_us[:x]
            """
            payload_sequences.append(len_seq)
            payload_sequences.append(ipd_seq)
            payload_sequences.append(tos_seq)
            payload_sequences.append(ip_hl_seq)
            payload_sequences.append(tcp_off_seq)
            """
            seg_extradata = torch.tensor(payload_sequences)
            flow_packets = len(len_seq)
            if flow_packets >= self.window_size:
                segs_idx = [idx for idx in range(0, flow_packets - self.window_size + 1)]
                for idx in segs_idx:
                    bit=3000
                    seg_len = torch.tensor(len_seq[idx: idx + self.window_size])
                    seg_ipd = torch.tensor(ipd_seq[idx: idx + self.window_size])
                    seg_len = torch.clamp(torch.tensor(len_seq[idx: idx + self.window_size]), max=bit)
                    seg_ipd = torch.clamp(torch.tensor(ipd_seq[idx: idx + self.window_size]), max=bit)
                    seg_tos = torch.tensor(tos_seq[idx: idx + self.window_size])
                    seg_iphl = torch.tensor(ip_hl_seq[idx: idx + self.window_size])
                    seg_tcpoff = torch.tensor(tcp_off_seq[idx: idx + self.window_size])
                    extradata = seg_extradata[:,idx:idx + self.window_size]
                    if torch.max(seg_len)>bit or torch.max(seg_ipd)>bit or torch.max(seg_tos)>bit or torch.max(seg_iphl)>bit or torch.max(seg_tcpoff)>bit:
                        print(torch.max(seg_len))
                        print(torch.max(seg_ipd))
                        print(torch.max(seg_tos))
                        print(torch.max(seg_iphl))
                        print(torch.max(seg_tcpoff))
                        print(torch.max(extradata))
                    packetfeatures = torch.stack([seg_len, seg_ipd, seg_tos, seg_iphl, seg_tcpoff], dim=1)
                    combined_seg = torch.cat((extradata.T, packetfeatures), dim=1)
                    self.flows.append({
                        'x': combined_seg,
                        'label': label
                    })
            else:
                raise Exception('Flow packets < window size!!!')

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        flow = self.flows[index]
        return flow['x'], flow['label']
