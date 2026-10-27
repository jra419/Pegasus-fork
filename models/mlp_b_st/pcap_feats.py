import argparse
import csv
import socket
import struct
import sys
import zlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

FEAT_COLS   = ['ip_proto', 'tcp_flags', 'ts_rel', 'packet_index', 'flow_len']

CSV_COLS    = ['frame_index', 'ts_abs', 'ip_src', 'ip_dst', 'ip_proto', 'port_src', 'port_dst',
               'tcp_flags', 'flow_id', 'flow_slot', 'packet_index', 'pkt_len', 'flow_len',
               'ts_rel', 'label']

# Timestamps are kept in units of 2**DEFAULT_TIME_SHIFT ns (256 ns): the switch reads them as
# ingress_mac_tstamp[39:8], a byte-aligned slice. A 32-bit register wraps every ~18 minutes.
DEFAULT_TIME_SHIFT      = 8
# Idle gap that ends a flow: 2**20 units of 2**8 ns (~268ms)
DEFAULT_IDLE_TIMEOUT    = 1 << 20
DEFAULT_FLOW_TABLE_SIZE = 65536

PKT_IDX_MAX   = 0xFFFF
FLOW_LEN_MAX  = 0xFFFFFFFF
TS_MASK       = 0xFFFFFFFF

NS_PER_S        = Decimal(1_000_000_000)
PURGE_INTERVAL  = 100_000


def flow_key_bytes(ip_src, ip_dst, ip_proto, port_src, port_dst):
    return (socket.inet_aton(ip_src) + socket.inet_aton(ip_dst)
            + struct.pack('!BHH', ip_proto, port_src, port_dst))

class HashedFlowTable:
    def __init__(self, size, timeout_units):
        self.size           = size
        self.timeout_units  = timeout_units
        self.sign           = [0] * size
        self.last_ts        = [0] * size
        self.pkts           = [0] * size
        self.bytes          = [0] * size
        self.flow_id        = [-1] * size
        self.next_flow_id   = 0

    # (flow_id, slot, ts_rel, pkt_idx, flow_len) for a packet, updating the slot.
    def update(self, key, now_units, pkt_len):
        crc    = zlib.crc32(flow_key_bytes(*key))
        slot   = crc & (self.size - 1)
        sign   = crc >> 16
        now    = now_units & TS_MASK
        gap    = (now - self.last_ts[slot]) & TS_MASK

        new_flow = self.sign[slot] != sign or gap >= self.timeout_units

        self.sign[slot]    = sign
        self.last_ts[slot] = now

        if new_flow:
            pkt_idx            = 0
            self.pkts[slot]    = 1
            self.bytes[slot]   = min(pkt_len, FLOW_LEN_MAX)
            self.flow_id[slot] = self.next_flow_id
            self.next_flow_id  += 1
            gap                = 0
        else:
            pkt_idx            = self.pkts[slot]
            self.pkts[slot]    = min(pkt_idx + 1, PKT_IDX_MAX)
            self.bytes[slot]   = min(self.bytes[slot] + pkt_len, FLOW_LEN_MAX)

        return self.flow_id[slot], slot, gap, pkt_idx, self.bytes[slot]

@dataclass
class FlowState:
    flow_id:    int
    pkts:       int = 0
    bytes:      int = 0
    last_ts:    int = 0

# Reference flow state without switch limits: one entry per 5-tuple, no ts wrap.
class UnlimitedFlowTable:
    def __init__(self, timeout_units):
        self.timeout_units  = timeout_units
        self.flows          = {}
        self.next_flow_id   = 0
        self.updates        = 0

    def update(self, key, now_units, pkt_len):
        self.updates += 1

        # Drop timed-out flows to bound memory.
        if self.updates % PURGE_INTERVAL == 0:
            self.flows = {k: f for k, f in self.flows.items()
                          if now_units - f.last_ts < self.timeout_units}

        flow = self.flows.get(key)

        if flow is None or now_units - flow.last_ts >= self.timeout_units:
            flow                = self.flows[key] = FlowState(self.next_flow_id)
            self.next_flow_id  += 1
            gap                 = 0
        else:
            gap                 = now_units - flow.last_ts

        pkt_idx         = flow.pkts
        flow.pkts       = min(flow.pkts + 1, PKT_IDX_MAX)
        flow.bytes      = min(flow.bytes + pkt_len, FLOW_LEN_MAX)
        flow.last_ts    = now_units

        return flow.flow_id, -1, gap, pkt_idx, flow.bytes

# One integer label per line; a non-numeric first line is treated as a header.
def read_labels(path):
    labels = []

    with open(path) as f:
        for line_number, line in enumerate(f):
            line = line.strip()

            if not line:
                continue

            try:
                labels.append(int(float(line)))
            except ValueError:
                if line_number == 0:
                    continue
                raise ValueError(f'{path}:{line_number + 1}: not an int label: {line!r}')

    return labels

def iter_pkt_rows(pcap_path, labels=None, idle_timeout=DEFAULT_IDLE_TIMEOUT,
                  max_frames=None, flow_table_size=DEFAULT_FLOW_TABLE_SIZE,
                  time_shift=DEFAULT_TIME_SHIFT):
    from scapy.all import IP, TCP, UDP, Dot1AD, Dot1Q, PcapReader

    table           = (HashedFlowTable(flow_table_size, idle_timeout) \
                            if flow_table_size else UnlimitedFlowTable(idle_timeout))
    frame_count     = 0

    with PcapReader(str(pcap_path)) as reader:
        for frame_idx, pkt in enumerate(reader):
            if max_frames is not None and frame_idx >= max_frames:
                break

            frame_count = frame_idx + 1

            # Skip VLAN tags.
            if not pkt.haslayer(IP) or pkt.haslayer(Dot1Q) or pkt.haslayer(Dot1AD):
                continue

            if labels is not None and frame_idx >= len(labels):
                raise ValueError('Length mismatch with PCAP. '
                                 f'Labels file length has {len(labels)} labels.')

            ts_ns       = int(Decimal(str(pkt.time)) * NS_PER_S)
            ip          = pkt[IP]
            ip_proto    = int(ip.proto)
            port_src    = port_dst = tcp_flags = 0

            has_l4 = int(ip.ihl) == 5 and int(ip.frag) == 0

            if has_l4 and ip_proto == 6 and pkt.haslayer(TCP):
                port_src, port_dst, tcp_flags = (int(pkt[TCP].sport), int(pkt[TCP].dport),
                                                 int(pkt[TCP].flags) & 0xFF)
            elif has_l4 and ip_proto == 17 and pkt.haslayer(UDP):
                port_src, port_dst = int(pkt[UDP].sport), int(pkt[UDP].dport)

            pkt_len = int(ip.len)

            flow_id, slot, ts_rel, pkt_idx, flow_len = \
                    table.update((ip.src, ip.dst, ip_proto, port_src, port_dst),
                                 ts_ns >> time_shift, pkt_len)

            yield {'frame_index':  frame_idx,
                   'ts_abs':       ts_ns,
                   'ip_src':       ip.src,
                   'ip_dst':       ip.dst,
                   'ip_proto':     ip_proto,
                   'port_src':     port_src,
                   'port_dst':     port_dst,
                   'tcp_flags':    tcp_flags,
                   'flow_id':      flow_id,
                   'flow_slot':    slot,
                   'packet_index': pkt_idx,
                   'pkt_len':      pkt_len,
                   'flow_len':     flow_len,
                   'ts_rel':       ts_rel,
                   'label':        labels[frame_idx] if labels is not None else -1}

    if labels is not None and max_frames is None and len(labels) != frame_count:
        print(f'WARNING: {len(labels)} labels for {frame_count} pcap frames', file=sys.stderr)

def write_csv(pcap_path, csv_path, labels_path=None, idle_timeout=DEFAULT_IDLE_TIMEOUT,
              max_frames=None, flow_table_size=DEFAULT_FLOW_TABLE_SIZE,
              time_shift=DEFAULT_TIME_SHIFT):
    labels  = read_labels(labels_path) if labels_path else None
    cnt     = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()

        for row in iter_pkt_rows(pcap_path, labels, idle_timeout, max_frames,
                                    flow_table_size, time_shift):
            writer.writerow(row)
            cnt += 1

            if cnt % 500_000 == 0:
                print(f'  {cnt} packets', file=sys.stderr)

    return cnt

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('pcap', type=Path)
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--idle_timeout', type=int, default=DEFAULT_IDLE_TIMEOUT,
                        help='Idle gap that ends a flow, in 2**time_shift ns units')
    parser.add_argument('--flow_table_size', type=int, default=DEFAULT_FLOW_TABLE_SIZE,
                        help='Hashed flow slots as on the switch (power of two, <= 65536); '
                             '0 for an unlimited table')
    parser.add_argument('--time_shift', type=int, default=DEFAULT_TIME_SHIFT,
                        help='Timestamps are kept in units of 2**time_shift ns')
    parser.add_argument('--max_frames', type=int,
                        help='Only read the first N pcap frames (for quick tests)')

    args = parser.parse_args()

    count = write_csv(args.pcap, args.output_csv, args.labels, args.idle_timeout,
                      args.max_frames, args.flow_table_size, args.time_shift)

    print(f'Wrote {count} IPv4 packets to {args.output_csv}')


if __name__ == "__main__":
    main()
