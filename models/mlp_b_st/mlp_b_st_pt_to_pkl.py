import argparse
import csv
import pickle
import sys
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .mlp_b_st import (
    HIDDEN_SIZE_1,
    HIDDEN_SIZE_2,
    NUM_LEAVES,
    NUM_SPLITS,
    TREE_DEPTH,
    MlpLut,
)
from .mlp_b_st_dataloader import FEAT_COLS, FLOW_KEY_COLS, load_pkts
from .mlp_b_st_metrics import write_metrics

# Bit width of each MM1 input feature (its header field on the switch), in FEAT_COLS order.
TOFINO_CODEBLOCK_BIT_LEN = [8, 8, 32, 16, 32]

# MM1 has one 1-value codeblock per feature. On the switch they share three tables: a pair
# is one table with two keys, a single feature one table with one key.
# (ip_proto, ts_rel), (tcp_flags, packet_index), (flow_len).
MM1_GROUPS = [(0, 2), (1, 3), (4,)]

# Values per codeblock in MM1, MM2, MM3 (training's --dlist); the table layout assumes these.
CODEBLOCK_DIMS = (1, 2, 1)

# Hidden values are packed two per 32-bit word: each in a signed 13-bit lane, the upper one
# 19 bits up. The 6 free bits between the lanes take the lower lane's carries, so packed
# words can be added directly: each lane, read as 13 bits, is the sum of its values.
LANE_BITS   = 13
LANE_SHIFT  = 19
LANE_MASK   = (1 << LANE_BITS) - 1
LANE_MIN    = -(1 << (LANE_BITS - 1))
LANE_MAX    = (1 << (LANE_BITS - 1)) - 1

# Each MM2/MM3 codeblock value is one hidden value, so one lane wide.
MM2_CODEBLOCK_BIT_LEN = LANE_BITS
MM3_CODEBLOCK_BIT_LEN = LANE_BITS

# MM2 has 16 codeblocks. Codeblock 0's table gives its LUT row directly (mm2_direct).
# Codeblocks 1..15 are done in two steps: one table per codeblock gives the leaf it reaches
# (mm2_leaf_*), then one exact-match table per three codeblocks maps their three leaves to
# the sum of their three LUT rows (mm2_group_*).
MM2_NUM_CODEBLOCKS  = HIDDEN_SIZE_1 // CODEBLOCK_DIMS[1]
MM2_GROUP_SIZE      = 3
MM2_NUM_GROUPS      = (MM2_NUM_CODEBLOCKS - 1) // MM2_GROUP_SIZE

INTERVAL_MIN = -10_000_000
INTERVAL_MAX = 100_000_000

TOFINO_TABLE_NAMES = {
    'table1': [
        'SwitchIngress.mm1_proto_ts_rel',
        'SwitchIngress.mm1_flags_pkt_idx',
        'SwitchIngress.mm1_flow_len',
    ],
    'table2_0_1'    : ['SwitchIngress.mm2_direct'],
    'table2_1_16'   : [f'SwitchIngress.mm2_leaf_{c}' for c in range(1, MM2_NUM_CODEBLOCKS)],
    'table2_3'      : [f'SwitchIngress.mm2_group_{c}_{c + MM2_GROUP_SIZE - 1}'
                        for c in range(1, MM2_NUM_CODEBLOCKS, MM2_GROUP_SIZE)],
    'table3'        : [f'SwitchIngress.mm3_word_{k}' for k in range(8)],
}

TABLE_SIZES = {
    'SwitchIngress.mm1_proto_ts_rel'            : 2048,
    'SwitchIngress.mm1_flags_pkt_idx'           : 2048,
    'SwitchIngress.mm1_flow_len'                : 2048,
    'SwitchIngress.mm2_direct'                  : 2048,
    'SwitchIngress.mm2_leaf_1'                  : 4096,
    **{f'SwitchIngress.mm2_leaf_{c}'            : 2048 for c in range(2, MM2_NUM_CODEBLOCKS)},
    'SwitchIngress.mm2_group_1_3'               : 8192,
    **{f'SwitchIngress.mm2_group_{c}_{c + 2}'   : 4096 for c in (4, 7, 10, 13)},
    **{f'SwitchIngress.mm3_word_{k}'            : 4096 for k in range(8)},
    'SwitchIngress.mm3_word_3'                  : 8192,
}

# ---------------------------------------------------------------------------
# Tree -> intervals
# ---------------------------------------------------------------------------

# The values of the codeblock's value $dim that can reach $leaf: follows the path to $leaf
# from the root, and each split on $dim on the way narrows the interval (going left keeps
# values <= threshold, going right values > threshold). Splits on the other value are
# skipped. S is None for 1-D codeblocks (every split tests the one value).
# Returns (left, right), meaning left < value <= right, or None if no value reaches $leaf.
def _leaf_interval_on_dim(S, T, H, codeblock, dim, leaf):
    valid   = 1
    node    = 0
    left    = INTERVAL_MIN
    right   = INTERVAL_MAX

    for _level in range(TREE_DEPTH):
        threshold = int(T[codeblock][node] // 1)
        tests_dim = S is None or int(S[codeblock][dim][node]) == 1

        if int(H[node][leaf]) == -1:
            if tests_dim:
                if left > threshold:
                    valid = 0
                elif right > threshold:
                    right = threshold

            node = node * 2 + 1

        elif int(H[node][leaf]) == 1:
            if tests_dim:
                if right < threshold:
                    valid = 0
                elif left < threshold:
                    left = threshold

            node = node * 2 + 2

        else:
            print(f'Error: {int(H[node][leaf])} in H is invalid!')

    if valid == 1 and left < right:
        return left, right

    return None

# 1-D codeblocks (MM1, MM3): the leaves split the value axis into consecutive intervals,
# left to right. Returns, per codeblock, [upper bound, leaf] for each reachable leaf in that
# order: values above the previous bound and up to this one reach this leaf. The last leaf
# (everything above the last bound) is left out; intervals_to_ternary_1d adds it.
def leaf_intervals_1d(T, H):
    intervals = []

    for codeblock in range(T.shape[0]):
        per_leaf = []

        for leaf in range(NUM_SPLITS):
            interval = _leaf_interval_on_dim(None, T, H, codeblock, 0, leaf)

            if interval is not None:
                per_leaf.append([interval[1], leaf])

        intervals.append(per_leaf)

    return intervals

# 2-D codeblocks (MM2): which leaf a pair of values reaches depends on both, so each value
# alone only narrows it down to a set of leaves. This cuts the axis of value $dim at every
# leaf's upper bound and returns, per piece in increasing order, [upper bound, leaf set]:
# bit j of the leaf set is 1 if leaf j can be reached with value $dim in that piece.
def _boundary_leaf_masks(S, T, H, codeblock, dim):
    leaf_intervals  = []
    boundaries      = []

    for leaf in range(NUM_LEAVES):
        interval = _leaf_interval_on_dim(S, T, H, codeblock, dim, leaf)

        if interval is not None:
            left, right = interval
            leaf_intervals.append([left, right, leaf])

            if right not in boundaries:
                boundaries.append(right)

    boundaries.sort()

    masks = [[bound, 0] for bound in boundaries]

    for left, right, leaf in leaf_intervals:
        lo = 0
        hi = len(boundaries) - 1

        while boundaries[lo] <= left:
            lo += 1
        while boundaries[hi] > right:
            hi -= 1
        while lo <= hi:
            masks[lo][1] = masks[lo][1] | (1 << leaf)
            lo += 1

    return masks

# _boundary_leaf_masks for both values of every 2-D codeblock: (per codeblock for value 0,
# per codeblock for value 1). The leaf a pair reaches is the one in both values' sets.
def leaf_intervals_2d(S, T, H):
    masks_dim0 = []
    masks_dim1 = []

    for codeblock in range(T.shape[0]):
        masks_dim0.append(_boundary_leaf_masks(S, T, H, codeblock, 0))
        masks_dim1.append(_boundary_leaf_masks(S, T, H, codeblock, 1))

    return masks_dim0, masks_dim1

# Fits the intervals to the key's value range: drops the ones entirely below it, cuts the
# upper bounds at $value_max, and merges intervals left with the same bound.
# starting bound [value_min, 0] in front, as _intervals_to_prefix_entries expects.
def _clamp_intervals(intervals, value_min, value_max):
    clamped = [[value_min, 0]]

    for interval in intervals:
        if value_min < interval[0]:
            interval[0] = min(interval[0], value_max)

            if clamped[-1][0] != interval[0]:
                clamped.append([interval[0], interval[1]])

    return clamped

# ---------------------------------------------------------------------------
# Intervals -> ternary entries
# ---------------------------------------------------------------------------

# One ternary entry, [priority, value, mask, action], that matches a block of 2^$level
# consecutive keys: the keys whose top ($bit_len - $level) bits are $value (a key already
# shifted right by $level).
# The action is $category itself (a leaf or leaf set), or the LUT row of leaf $category.
def prefix_entry(value, level, bit_len, category, codeblock, LUT, return_category):
    mask    = ((1 << (bit_len - level)) - 1) << level
    action  = (category if return_category
               else [int(num) for num in torch.floor(LUT[codeblock][category]).tolist()])

    return [level, (value % (1 << (bit_len - level))) << level, mask, action]

# Range-to-prefix expansion: turns consecutive intervals into ternary entries.
# $bounds is [upper bound, category] in increasing order; the first one is only the
# starting bound (one below the smallest key). Keys above one bound and up to the next
# get the next one's category.
# A ternary entry can only match an aligned block of 2^level keys, so an interval is
# covered by blocks of different sizes. The loop works one level at a time, from single
# keys (level 0) up: where a bound falls inside a block of the next size, that block
# would mix two categories, so the keys on each side of the bound get entries at this
# level. Then every bound is divided by 2 (one level coarser) and the loop repeats. The
# last two entries cover what is left with the two widest blocks.
def _intervals_to_prefix_entries(bounds, bit_len, codeblock=0, LUT=None,
                                 return_category=True):
    entries = []
    level   = 0

    def entry(value, category):
        return prefix_entry(value, level, bit_len, category, codeblock, LUT, return_category)

    while level < bit_len - 1:
        coarser = []
        j       = 1

        while j < len(bounds) - 1:
            if bounds[j][0] % 2 == 0:
                parent = bounds[j][0] // 2

                if parent % 2 == 0:
                    if bounds[j][0] + 1 != bounds[j + 1][0]:
                        entries.append(entry(bounds[j][0], bounds[j][1]))

                        if bounds[j][0] - 1 != bounds[j - 1][0]:
                            coarser.append([parent - 1, bounds[j][1]])
                            bounds[j][0] -= 1
                    else:
                        coarser.append([parent, bounds[j][1]])
                        bounds[j][0] += 1
                        entries.append(entry(bounds[j][0], bounds[j + 1][1]))
                        j += 1
                else:
                    if bounds[j][0] - 1 != bounds[j - 1][0]:
                        entries.append(entry(bounds[j][0] + 1, bounds[j + 1][1]))
                        coarser.append([parent, bounds[j][1]])
                        bounds[j][0] += 1

                        if bounds[j][0] == bounds[j + 1][0]:
                            bounds[j + 1][1] = bounds[j][1]
                            j += 1
                    else:
                        entries.append(entry(bounds[j][0], bounds[j][1]))
            else:
                coarser.append([bounds[j][0] // 2, bounds[j][1]])

            j += 1

        if j == len(bounds) - 1:
            coarser.append([bounds[-1][0] // 2, bounds[-1][1]])

        bounds = [[bounds[0][0] // 2, bounds[0][1]]]
        bounds.extend(coarser)
        level += 1

    entries.append(entry(bounds[-1][0], bounds[-1][1]))

    if bounds[-1][0] == bounds[0][0]:
        print(f'Warning: {bounds}')
    elif bounds[-2][0] != bounds[0][0]:
        entries.append(entry(bounds[-2][0], bounds[-2][1]))
    else:
        entries.append(entry(bounds[-2][0] + 1, bounds[-1][1]))

    return entries

# The first and last bound for a key of $bit_len bits: one below its smallest value, and
# its largest value.
def _value_range(bit_len, signed):
    if signed == 1:
        return -(1 << (bit_len - 1)) - 1, (1 << (bit_len - 1)) - 1

    return -1, (1 << bit_len) - 1

# One ternary table per 1-D codeblock, from leaf_intervals_1d: each entry's action is the
# LUT row of the leaf its keys reach. $bit_len is the key width, one for all codeblocks or a
# list with one per codeblock. The last leaf gets the keys above every interval, and
# _clamp_intervals fits the bounds to the key's range: a leaf only reached by values the key
# cannot hold (e.g. negative values of an unsigned key) is dropped, and when a threshold is at
# or above the largest key, the leaf on its 'no' side gets every key up to the largest.
def intervals_to_ternary_1d(intervals, bit_len, LUT, signed=1):
    ternary = []

    for codeblock in range(len(intervals)):
        key_bit_len = bit_len[codeblock] if isinstance(bit_len, list) else bit_len

        value_min, value_max = _value_range(key_bit_len, signed)

        bounds = _clamp_intervals(intervals[codeblock] + [[value_max, NUM_LEAVES - 1]],
                                  value_min, value_max)

        ternary.append(_intervals_to_prefix_entries(bounds, key_bit_len, codeblock, LUT,
                                                    return_category=False))

    return ternary

# One ternary table per 2-D codeblock, from leaf_intervals_2d. The key is the codeblock's
# two values packed in one word (value 0 in the upper lane). Each value's pieces are
# expanded into ternary entries separately, with the leaf set as the action; then every
# pair of entries (one per value) whose leaf sets share a leaf becomes one entry on the
# packed key. The action is that leaf ($return_category) or its LUT row.
# With $return_category, also returns which leaves each codeblock can reach.
def intervals_to_ternary_2d(masks_dim0, masks_dim1, bit_len_0, bit_len_1, LUT,
                            signed=1, return_category=True):
    min0, max0 = _value_range(bit_len_0, signed)
    min1, max1 = _value_range(bit_len_1, signed)

    ternary         = []
    used_categories = []

    for codeblock in range(len(masks_dim0)):
        entries_dim0 = _intervals_to_prefix_entries(
                _clamp_intervals(masks_dim0[codeblock], min0, max0), bit_len_0)
        entries_dim1 = _intervals_to_prefix_entries(
                _clamp_intervals(masks_dim1[codeblock], min1, max1), bit_len_1)

        entries     = []
        used_leaves = [0] * NUM_LEAVES

        for prio0, value0, mask0, leaves0 in entries_dim0:
            for prio1, value1, mask1, leaves1 in entries_dim1:
                shared = leaves0 & leaves1

                if shared == 0:
                    continue

                for leaf in range(NUM_LEAVES):
                    if shared & (1 << leaf) == 0:
                        continue

                    used_leaves[leaf] = 1

                    action = (leaf if return_category
                              else [int(num) for num in torch.floor(LUT[codeblock][leaf]).tolist()])

                    entries.append([prio0 + prio1,
                                    ((value0 & LANE_MASK) << LANE_SHIFT) | (value1 & LANE_MASK),
                                    ((mask0 & LANE_MASK) << LANE_SHIFT) | (mask1 & LANE_MASK),
                                    action])

        ternary.append(entries)
        used_categories.append(used_leaves)

    return (ternary, used_categories) if return_category else ternary

# ---------------------------------------------------------------------------
# Combining tables
# ---------------------------------------------------------------------------

# Merges two tables into one, keyed on both keys packed in one word ($table1's in the
# upper lane): one entry per pair of entries, with the two priorities and actions added.
# One lookup then gives the sum of both tables' outputs.
def fuse_tables(table1, lane_shift, table2):
    table = []

    for prio1, value1, mask1, action1 in table1:
        for prio2, value2, mask2, action2 in table2:
            table.append([prio1 + prio2,
                          ((value1 & LANE_MASK) << lane_shift) | (value2 & LANE_MASK),
                          ((mask1 & LANE_MASK) << lane_shift) | (mask2 & LANE_MASK),
                          [a1 + a2 for a1, a2 in zip(action1, action2)]])

    return table

# Like fuse_tables, but the result has two separate keys (for keys in different fields).
def link_tables(table1, table2):
    table = []

    for prio1, value1, mask1, action1 in table1:
        for prio2, value2, mask2, action2 in table2:
            table.append([prio1 + prio2, value1, value2, mask1, mask2,
                          [a1 + a2 for a1, a2 in zip(action1, action2)]])

    return table

# Packs each entry's action (a LUT row) two values per 32-bit word, as the switch adds them).
def pack_outputs(tables, lane_shift):
    for table in tables:
        for entry in table:
            action      = entry[-1]
            entry[-1]   = [((action[2 * j] & LANE_MASK) << lane_shift)
                           | (action[2 * j + 1] & LANE_MASK)
                           for j in range(len(action) // 2)]

    return tables

# One exact-match table per codeblock: [leaf, LUT row] for each leaf the codeblock can reach.
def lut_to_exact_tables(LUT, used_categories):
    tables = []

    for codeblock in range(len(LUT)):
        table = []

        for leaf in range(NUM_LEAVES):
            if used_categories[codeblock][leaf] == 1:
                table.append([leaf,
                              [int(num) for num in torch.floor(LUT[codeblock][leaf]).tolist()]])

        tables.append(table)

    return tables

# Merges three exact tables into one with three keys: one entry per combination of
# leaves, whose action is the sum of the three LUT rows.
def _link_exact_triple(table0, table1, table2):
    table = []

    for key0, action0 in table0:
        for key1, action1 in table1:
            for key2, action2 in table2:
                action = [action0[k] + action1[k] + action2[k] for k in range(len(action0))]
                table.append([key0, key1, key2, action])

    return table

# _link_exact_triple on consecutive tables, three at a time (MM2 codeblocks 1-3, 4-6, ...).
# Tables left over at the end are dropped.
def link_exact_triples(tables):
    return [_link_exact_triple(tables[i], tables[i + 1], tables[i + 2])
            for i in range(0, len(tables) - 1, 3)]

# fuse_tables on consecutive tables, two at a time. For MM3, whose codeblocks are single
# hidden values: the two codeblocks in one packed hidden2 word become one table keyed on
# that word (mm3_word_*). An odd last table is kept as is.
def fuse_pairs(tables, lane_shift):
    fused = [fuse_tables(tables[i], lane_shift, tables[i + 1])
             for i in range(0, len(tables) - 1, 2)]

    if len(tables) % 2 == 1:
        fused.append(tables[-1])

    return fused

# The three MM1 tables (MM1_GROUPS): each pair of feature tables merged into one with two
# keys, the single one kept; actions packed into hidden1 words.
def build_mm1_tables(ternary_mm1):
    tables = [link_tables(ternary_mm1[g[0]], ternary_mm1[g[1]]) if len(g) == 2
              else ternary_mm1[g[0]]
              for g in MM1_GROUPS]

    return pack_outputs(tables, LANE_SHIFT)

# ---------------------------------------------------------------------------
# Table lookups (simulating the switch)
# ---------------------------------------------------------------------------

# Action of the first entry of table $index matching $key: the entry the switch picks, as
# the controller ranks entries by list position.
# If none matches (bug in the conversion), return the last entry's action.
def lookup_ternary(tables, index, key):
    for _prio, value, mask, action in tables[index]:
        if key & mask == value:
            return action

    print(f'Error: No ternary match in table {index} for key {key}.')

    return tables[index][-1][-1]

# lookup_ternary for a table with two keys.
def lookup_dual_ternary(tables, index, key1, key2):
    for _prio, value1, value2, mask1, mask2, action in tables[index]:
        if key1 & mask1 == value1 and key2 & mask2 == value2:
            return action

    print(f'Error: No dual-key match in table {index} for keys ({key1}, {key2}).')

    return tables[index][-1][-1]

# Element-wise sum of equal-length vectors.
def sum_vectors(vectors):
    return [sum(column) for column in zip(*vectors)]

# Action of the entry of exact table $index with keys ($key0, $key1, $key2);
# if there is none, return the last entry's action.
def lookup_exact_3key(tables, index, key0, key1, key2):
    for *keys, action in tables[index]:
        if keys == [key0, key1, key2]:
            return action

    print(f'No exact match in MM2 group table {index} for keys ({key0}, {key1}, {key2}).')

    return tables[index][-1][-1]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require_state_dict(model):
    if isinstance(model, dict) and 'MM1.T' in model:
        return model

    if hasattr(model, 'state_dict'):
        sd = model.state_dict()
        if 'MM1.T' in sd:
            return sd

    keys = list(model.keys())[:12] if isinstance(model, dict) else type(model)

    raise ValueError('Error: Expected a LUT state dict with keys MM1.T, MM2.T, MM3.T '
                     f'(from mlp_b_st_lut_convert), got: {keys}.')

def _validate_mm1_codeblocks(model):
    num_codeblocks = int(model['MM1.T'].shape[0])

    if num_codeblocks != len(TOFINO_CODEBLOCK_BIT_LEN):
        raise ValueError(f'Error: MM1.T has {num_codeblocks} codeblocks but expects '
                         f'{len(TOFINO_CODEBLOCK_BIT_LEN)}.')

# MM2/MM3 table layout is fixed to HIDDEN_SIZE_1/HIDDEN_SIZE_2 with CODEBLOCK_DIMS.
def _validate_architecture(model):
    expected_mm2 = (MM2_NUM_CODEBLOCKS, CODEBLOCK_DIMS[1])
    expected_mm3 = (HIDDEN_SIZE_2 // CODEBLOCK_DIMS[2], CODEBLOCK_DIMS[2])

    if tuple(model['MM2.S'].shape[:2]) != expected_mm2:
        raise ValueError(f"Error: MM2.S has shape {tuple(model['MM2.S'].shape)}; "
                         f"expected hidden1={HIDDEN_SIZE_1} with D2={CODEBLOCK_DIMS[1]}.")

    if tuple(model['MM3.S'].shape[:2]) != expected_mm3:
        raise ValueError(f"Error: MM3.S has shape {tuple(model['MM3.S'].shape)}; "
                         f"expected hidden2={HIDDEN_SIZE_2} with D3={CODEBLOCK_DIMS[2]}.")

# Warn when the worst-case sum of a layer's LUT rows can overflow a signed lane.
def _check_lane_bound(LUT, layer):
    bound = float(LUT.abs().amax(dim=1).sum(dim=0).max())

    if bound > LANE_MAX:
        print(f'Warning: {layer} worst-case output {bound:.0f} exceeds {LANE_BITS}-bit lane '
              f'[{LANE_MIN}, {LANE_MAX}]; confirm actual outputs with --check.')

# Warn when a converted table has more entries than its P4 declaration holds.
def _check_table_sizes(tables, names):
    for table, name in zip(tables, names, strict=True):
        if len(table) > TABLE_SIZES[name]:
            print(f"Warning: {name} has {len(table)} entries (> {TABLE_SIZES[name]}); "
                  "the controller will truncate it.")

# ---------------------------------------------------------------------------
# Conversion and simulation
# ---------------------------------------------------------------------------

# Build all switch tables from the LUT model's state dict.
# Return them in pkl order: MM1 tables, mm2_direct, mm2_leaf_*, mm2_group_*, mm3_word_*.
def convert_model_to_tables(model):
    model = _require_state_dict(model)

    _validate_mm1_codeblocks(model)
    _validate_architecture(model)

    print('MM shapes:',
          {k: tuple(model[k].shape) for k in ('MM1.T', 'MM2.T', 'MM3.T')})

    # MM1: raw features -> hidden1.
    _check_lane_bound(model['MM1.LUT'], 'MM1')

    mm1_intervals   = leaf_intervals_1d(model['MM1.T'], model['MM1.H'])
    mm1_ternary     = intervals_to_ternary_1d(mm1_intervals, TOFINO_CODEBLOCK_BIT_LEN,
                                              model['MM1.LUT'], signed=0)
    mm1_tables      = build_mm1_tables(mm1_ternary)

    # MM2: hidden1 -> hidden2.
    mm2_lut = model['MM2.LUT']

    _check_lane_bound(mm2_lut, 'MM2')

    mm2_masks_dim0, mm2_masks_dim1 = leaf_intervals_2d(model['MM2.S'], model['MM2.T'],
                                                       model['MM2.H'])

    mm2_direct_tables = intervals_to_ternary_2d(mm2_masks_dim0[0:1], mm2_masks_dim1[0:1],
                                                MM2_CODEBLOCK_BIT_LEN,
                                                MM2_CODEBLOCK_BIT_LEN, mm2_lut[0:1],
                                                signed=1, return_category=False)
    mm2_direct_tables = pack_outputs(mm2_direct_tables, LANE_SHIFT)

    mm2_leaf_tables, used_leaves = intervals_to_ternary_2d(
            mm2_masks_dim0[1:MM2_NUM_CODEBLOCKS], mm2_masks_dim1[1:MM2_NUM_CODEBLOCKS],
            MM2_CODEBLOCK_BIT_LEN, MM2_CODEBLOCK_BIT_LEN,
            mm2_lut[1:MM2_NUM_CODEBLOCKS], signed=1, return_category=True)

    mm2_group_tables = lut_to_exact_tables(mm2_lut[1:MM2_NUM_CODEBLOCKS], used_leaves)
    mm2_group_tables = pack_outputs(link_exact_triples(mm2_group_tables), LANE_SHIFT)

    # MM3: hidden2 -> class logits.
    mm3_intervals   = leaf_intervals_1d(model['MM3.T'], model['MM3.H'])
    mm3_ternary     = intervals_to_ternary_1d(mm3_intervals, MM3_CODEBLOCK_BIT_LEN,
                                              model['MM3.LUT'])
    mm3_tables      = fuse_pairs(mm3_ternary, LANE_SHIFT)

    for tables, key in ((mm1_tables, 'table1'), (mm2_direct_tables, 'table2_0_1'),
                        (mm2_leaf_tables, 'table2_1_16'), (mm2_group_tables, 'table2_3'),
                        (mm3_tables, 'table3')):
        _check_table_sizes(tables, TOFINO_TABLE_NAMES[key])

    return mm1_tables, mm2_direct_tables, mm2_leaf_tables, mm2_group_tables, mm3_tables

# Run one pkt's features through the tables (switch sim) and return the class logits.
# hidden1 and hidden2 are lists of packed words: each hidden1 word is one MM2 key
# (a 2-D codeblock), each hidden2 word one mm3_word_* key.
def simulate_switch(feats, mm1_tables, mm2_direct_tables, mm2_leaf_tables,
                    mm2_group_tables, mm3_tables):
    hidden1 = sum_vectors([
            lookup_dual_ternary(mm1_tables, k, feats[g[0]], feats[g[1]]) if len(g) == 2
            else lookup_ternary(mm1_tables, k, feats[g[0]])
            for k, g in enumerate(MM1_GROUPS)])

    hidden2_parts = [lookup_ternary(mm2_direct_tables, 0, hidden1[0])]
    mm2_leaves    = [lookup_ternary(mm2_leaf_tables, j - 1, hidden1[j])
                     for j in range(1, MM2_NUM_CODEBLOCKS)]

    for group in range(MM2_NUM_GROUPS):
        first = MM2_GROUP_SIZE * group

        hidden2_parts.append(lookup_exact_3key(mm2_group_tables, group, mm2_leaves[first],
                                               mm2_leaves[first + 1], mm2_leaves[first + 2]))

    hidden2     = sum_vectors(hidden2_parts)
    logit_parts = [lookup_ternary(mm3_tables, j, hidden2[j]) for j in range(len(hidden2))]

    return sum_vectors(logit_parts)

# ---------------------------------------------------------------------------
# I/O and evaluation
# ---------------------------------------------------------------------------

# Switch header field: keep only the lowest $bits bits.
def mask_unsigned(value, bits):
    if bits >= 32:
        return int(value) & 0xFFFFFFFF

    return int(value) & ((1 << bits) - 1)

# Index of the largest logit (first one on ties).
def argmax_label(logits, num_classes):
    label   = 0
    best    = logits[0]

    for i in range(1, num_classes):
        if logits[i] > best:
            best    = logits[i]
            label   = i

    return label

# Row-wise softmax, shifted by the row max for numerical stability.
def _logits_to_probs(scores):
    shifted     = scores - scores.max(axis=1, keepdims=True)
    exp_scores  = np.exp(shifted)

    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

# Confidence score: top logit minus the second, in switch logit units.
def logit_margin(logits):
    top, second = sorted(logits, reverse=True)[:2]

    return top - second

# The P4 confidence check (confidence_check table): a packet is confident when its top logit
# beats the second by at least the threshold.
def is_confident(logits, threshold):
    return logit_margin(logits) >= threshold

def open_packets_csv(run_dir, dataset, attack, date_time, num_classes, cur_model):
    path = Path(run_dir) / f'{cur_model}-{dataset}-{attack}-{date_time}.csv'

    f       = open(path, 'w', newline='')
    writer  = csv.writer(f)

    writer.writerow(FLOW_KEY_COLS
                    + [f'feat_{column}' for column in FEAT_COLS]
                    + [f'logit_{c}' for c in range(num_classes)]
                    + ['margin', 'confident', 'label'])

    return f, writer, path

def print_conf_metrics(truth, pred, scores, threshold, num_classes):
    confident   = [is_confident(logits[:num_classes], threshold) for logits in scores]
    n           = len(truth)

    print(f'Confidence threshold (switch logit units, P4 CONF_THRESHOLD): {threshold}')
    print(f'Confident packets: {sum(confident)} / {n} ({sum(confident) / n:.4f})')

    for name, selected in (('confident', True), ('not confident', False)):
        rows = [i for i in range(n) if confident[i] == selected]

        if not rows:
            print(f'  {name}: no packets')
            continue

        t = [truth[i] for i in rows]
        p = [pred[i] for i in rows]

        print(f'  {name}: {len(rows)} packets | accuracy {accuracy_score(t, p):.4f} | '
              f"macro-F1 {f1_score(t, p, average='macro', zero_division=0):.4f}")

def eval_metrics(truth, pred, scores, num_classes):
    n = len(truth)

    if n == 0:
        print('No samples for eval.')
        return

    acc             = accuracy_score(truth, pred)
    precision_macro = precision_score(truth, pred, average='macro', zero_division=0)
    recall_macro    = recall_score(truth, pred, average='macro', zero_division=0)
    f1_macro        = f1_score(truth, pred, average='macro', zero_division=0)

    cm = confusion_matrix(truth, pred, labels=list(range(num_classes)))

    per_class_rates = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        per_class_rates.append((tpr, tnr, fpr, fnr))

    tpr_macro = sum(r[0] for r in per_class_rates) / num_classes
    tnr_macro = sum(r[1] for r in per_class_rates) / num_classes
    fpr_macro = sum(r[2] for r in per_class_rates) / num_classes
    fnr_macro = sum(r[3] for r in per_class_rates) / num_classes

    print(f'Samples: {n}')
    print(f'TPR (macro): {tpr_macro:.4f}')
    print(f'TNR (macro): {tnr_macro:.4f}')
    print(f'FPR (macro): {fpr_macro:.4f}')
    print(f'FNR (macro): {fnr_macro:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision (macro): {precision_macro:.4f}')
    print(f'Recall (macro): {recall_macro:.4f}')
    print(f'F1-score (macro): {f1_macro:.4f}')

    for c in range(num_classes):
        tpr, tnr, fpr, fnr = per_class_rates[c]

        print(f'  class {c}: TPR={tpr:.4f} TNR={tnr:.4f} '
              f'FPR={fpr:.4f} FNR={fnr:.4f}')

    try:
        score_arr = np.asarray(scores, dtype=np.float64)

        if score_arr.shape[1] < num_classes:
            raise ValueError(f'Error: Expected at least {num_classes} class scores, '
                             f'got shape {score_arr.shape}.')

        probs = _logits_to_probs(score_arr[:, :num_classes])

        # Binary AUC takes the positive-class probability; multi-class takes all columns.
        if num_classes == 2:
            auc_roc = roc_auc_score(truth, probs[:, 1])
        else:
            auc_roc = roc_auc_score(truth, probs, multi_class='ovr', average='macro')

        print(f'AUC-ROC (macro, ovr): {auc_roc:.4f}')

        y               = np.asarray(truth)
        per_class_auc   = []

        for c in range(num_classes):
            y_bin = (y == c).astype(int)

            if len(np.unique(y_bin)) < 2:
                per_class_auc.append(None)
                continue

            per_class_auc.append(roc_auc_score(y_bin, probs[:, c]))

        print(f'Per-class OVR AUC: {per_class_auc}')

    except Exception as exc:
        print(f'Warning: AUC-ROC not computed ({type(exc).__name__}: {exc})')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cur_model', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--test_labels', type=str, default=None,
                        help='Labels for a pcap --test_data_path (default: <name>-labels.csv)')
    parser.add_argument('--output_pkl_path', type=str)
    parser.add_argument('--dataset_pkl_path', type=str)
    parser.add_argument('--metadata_pkl_path', type=str)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--max_test_rows', type=int, default=0)
    parser.add_argument('--check', action='store_true',
                        help='Compare table simulation against MlpLut on the same rows')
    parser.add_argument('--verbose', action='store_true',
                        help='Print logits and label for every simulated row')
    parser.add_argument("--conf_tau", type=float, default=None,
                        help='Confidence threshold on the logit margin in model logit units (e.g. '
                             '0.9); converted with the MM3 scale saved by mlp_b_st_lut_convert')
    parser.add_argument('--conf_threshold', type=int, default=None,
                        help='Confidence threshold directly in switch logit units '
                        '(overrides --conf_tau)')
    parser.add_argument('--packets_csv', action='store_true',
                        help='Write every evaluated packet (five-tuple, features, logits, '
                             'margin, confident, label)')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--attack', type=str, default=None)
    parser.add_argument('--date_time', type=str, default=None)

    args = parser.parse_args()
    start = time.time()

    if args.packets_csv and not (args.dataset and args.attack):
        sys.exit('Error: --packets_csv needs --dataset and --attack')

    if not args.test_only and not args.output_pkl_path:
        sys.exit('Error: --output_pkl_path is required unless --test_only')

    model = _require_state_dict(torch.load(args.model_path, weights_only=False,
                                           map_location='cpu'))

    # Saved by mlp_b_st_lut_convert; not part of the tables.
    logit_scale = model.pop('logit_scale', None)

    if args.conf_threshold is not None:
        conf_threshold = args.conf_threshold
    elif args.conf_tau is not None:
        if logit_scale is None:
            sys.exit('Error: --conf_tau needs the MM3 scale; re-run mlp_b_st_lut_convert '
                     'or pass --conf_threshold')
        conf_threshold = round(args.conf_tau * float(logit_scale))
    else:
        conf_threshold = None

    if args.packets_csv and (logit_scale is None or conf_threshold is None):
        sys.exit('Error: --packets_csv needs the MM3 scale saved by mlp_b_st_lut_convert and '
                 'a threshold (--conf_tau or --conf_threshold)')

    # Build the reference model before conversion so it sees the untouched state dict.
    if args.check:
        ref_model = MlpLut(len(FEAT_COLS), HIDDEN_SIZE_1, HIDDEN_SIZE_2, args.num_classes,
                           torch.device('cpu'), *CODEBLOCK_DIMS)
        ref_model.load_state_dict(model)
        ref_model.eval()

    tables = convert_model_to_tables(model)

    print('MM1 table sizes:', [len(t) for t in tables[0]])

    if not args.test_only:
        with open(args.output_pkl_path, 'wb') as f:
            pickle.dump(list(tables), f, protocol=2)

        print(f'Wrote tables: {args.output_pkl_path}')

        if args.metadata_pkl_path:
            meta = {'target'            : 'tofino',
                    'p4_dir'            : f'p4/{args.cur_model}',
                    'feature_columns'   : FEAT_COLS,
                    'codeblock_bit_len' : TOFINO_CODEBLOCK_BIT_LEN,
                    'mm1_groups'        : MM1_GROUPS,
                    'table_names'       : TOFINO_TABLE_NAMES}

            with open(args.metadata_pkl_path, 'wb') as f:
                pickle.dump(meta, f, protocol=2)

            print(f'Wrote metadata: {args.metadata_pkl_path}')

    raw, labels, flow_keys  = load_pkts(args.test_data_path, args.test_labels,
                                           with_flow_keys=True)
    raw                     = raw.tolist()
    labels      = labels.tolist()

    feats = [[mask_unsigned(v, b) for v, b in zip(row, TOFINO_CODEBLOCK_BIT_LEN)]
                for row in raw]

    if not args.test_only and args.dataset_pkl_path:
        with open(args.dataset_pkl_path, 'wb') as f:
            pickle.dump([feats, labels], f, protocol=2)

        print(f'Wrote dataset: {args.dataset_pkl_path}')

    limit               = args.max_test_rows or len(feats)
    pred, truth, scores = [], [], []

    if args.packets_csv:
        date_time = args.date_time or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        packets_file, packets_writer, packets_path = open_packets_csv(
                Path(args.model_path).parent, args.dataset, args.attack, date_time,
                args.num_classes, args.cur_model)

    for i in range(limit):
        logits  = simulate_switch(feats[i], *tables)
        label   = argmax_label(logits, args.num_classes)

        if args.packets_csv:
            class_logits    = list(logits[:args.num_classes])
            margin          = logit_margin(class_logits)

            packets_writer.writerow([*flow_keys[i], *feats[i], *class_logits,
                                     f"{margin / float(logit_scale):.4f}",
                                     int(margin >= conf_threshold), labels[i]])

        if args.verbose:
            print(logits)
            print(label)
            print("---------------------------")

        pred.append(label)
        truth.append(labels[i])
        scores.append(list(logits))

        if (i + 1) % 128 == 0:
            running_acc = sum(p == t for p, t in zip(pred, truth)) / (i + 1)
            print(f'  progress {i + 1}/{limit} acc={running_acc:.4f}')

    if args.packets_csv:
        packets_file.close()
        print(f'Wrote per-packet results: {packets_path}')

    eval_metrics(truth, pred, scores, args.num_classes)

    if conf_threshold is not None:
        print_conf_metrics(truth, pred, scores, conf_threshold, args.num_classes)

    extra = []

    if conf_threshold is not None:
        confident = [is_confident(logits[:args.num_classes], conf_threshold) for logits in scores]
        n_conf    = sum(confident)
        extra     = [f'Confidence threshold (switch logit units): {conf_threshold}',
                     f'Confident packets: {n_conf} ({n_conf / len(truth)})']

        for name, selected in (('confident', True), ('not confident', False)):
            rows = [i for i in range(len(truth)) if confident[i] == selected]
            if rows:
                extra.append(f'Accuracy ({name}): '
                             f'{accuracy_score([truth[i] for i in rows], [pred[i] for i in rows])}')

    probs = _logits_to_probs(np.asarray(scores, dtype=np.float64)[:, :args.num_classes])

    write_metrics(Path(args.model_path).parent, 'switch', truth, pred, probs, args.num_classes,
                  time.time() - start, extra)

    if args.check:
        with torch.no_grad():
            ref_pred = ref_model(torch.tensor(raw[:limit], dtype=torch.float32)).argmax(-1)

        agree = sum(int(p) == int(r) for p, r in zip(pred, ref_pred.tolist())) / limit

        print(f'Agreement with MlpLut: {agree:.4f} ({limit} rows)')


if __name__ == "__main__":
    main()
