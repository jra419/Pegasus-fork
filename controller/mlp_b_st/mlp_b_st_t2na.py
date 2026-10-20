from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

PIPE = "SwitchIngress"

PRIORITY_MAX = (2 ** 31) - 1

MM1_TABLES = [
    ("mm1_proto_ts_rel", ["hdr.ipv4.protocol", "ig_md.feat.ts_rel"]),
    ("mm1_flags_pkt_idx", ["ig_md.feat.tcp_flags", "ig_md.feat.packet_index"]),
    ("mm1_flow_len", ["ig_md.feat.flow_len"]),
]
MM2_DIRECT_TABLE    = ("mm2_direct", ["ig_md.hidden1.w0"])
MM2_LEAF_TABLES     = [(f"mm2_leaf_{c}", [f"ig_md.hidden1.w{c}"]) for c in range(1, 16)]
MM2_GROUP_TABLES    = [(f"mm2_group_{c}_{c + 2}", [f"ig_md.mm2_leaf.cb{c + i}" for i in range(3)])
                       for c in range(1, 16, 3)]
MM3_TABLES          = [(f"mm3_word_{k}", [f"ig_md.hidden2.w{k}"]) for k in range(8)]

FLOW_REGISTERS  = ["flow_sig", "last_ts", "pkt_count", "byte_count"]
FLOW_TABLE_SIZE = 65536
REG_INDEX_KEY   = "$REG_INDEX"

MM1_PARAMS  = [f"w{i}" for i in range(16)]
MM2_PARAMS  = [f"w{i}" for i in range(8)]
LEAF_PARAMS = ["leaf"]
MM3_PARAMS  = [f"class_{i}" for i in range(4)]

LEAF_WIDTH  = 4
WORD_WIDTH  = 32


def _param_width(name):
    return LEAF_WIDTH if name == "leaf" else WORD_WIDTH

@dataclass
class Entry:
    table:      str
    keys:       list    # (field, value, mask) for ternary, (field, value, None) for exact
    action:     str
    params:     list    # (name, value)
    priority:   int | None

def _priority(index, order):
    return index if order == "lower_wins" else PRIORITY_MAX - index

def _ternary_entries(table, key_fields, param_names, rows, order):
    entries = []

    for index, row in enumerate(rows):
        if len(key_fields) == 1:
            _tree_priority, value, mask, action = row
            keys = [(key_fields[0], value, mask)]
        else:
            _tree_priority, value1, value2, mask1, mask2, action = row
            keys = [(key_fields[0], value1, mask1), (key_fields[1], value2, mask2)]

        values = action if isinstance(action, list) else [action]
        values = values + [0] * (len(param_names) - len(values))

        entries.append(Entry(table, keys, f"set_{table}", list(zip(param_names, values)),
                             _priority(index, order)))

    return entries

def _exact_entries(table, key_fields, param_names, rows):
    entries = []

    for row in rows:
        *key_values, action = row
        entries.append(Entry(table, [(f, v, None) for f, v in zip(key_fields, key_values)],
                             f"set_{table}", list(zip(param_names, action)), None))

    return entries

def build_entries(tables, order):
    mm1, mm2_direct, mm2_leaf, mm2_group, mm3 = tables

    for name, expected, got in (("MM1", len(MM1_TABLES), len(mm1)),
                                ("MM2 direct", 1, len(mm2_direct)),
                                ("MM2 leaf", len(MM2_LEAF_TABLES), len(mm2_leaf)),
                                ("MM2 group", len(MM2_GROUP_TABLES), len(mm2_group)),
                                ("MM3", len(MM3_TABLES), len(mm3))):
        if expected != got:
            raise ValueError(f"{name}: pkl has {got} tables, P4 layout expects {expected}.")

    per_table = []

    for (table, keys), rows in zip(MM1_TABLES, mm1):
        per_table.append((table, _ternary_entries(table, keys, MM1_PARAMS, rows, order)))

    table, keys = MM2_DIRECT_TABLE
    per_table.append((table, _ternary_entries(table, keys, MM2_PARAMS, mm2_direct[0], order)))

    for (table, keys), rows in zip(MM2_LEAF_TABLES, mm2_leaf):
        per_table.append((table, _ternary_entries(table, keys, LEAF_PARAMS, rows, order)))

    for (table, keys), rows in zip(MM2_GROUP_TABLES, mm2_group):
        per_table.append((table, _exact_entries(table, keys, MM2_PARAMS, rows)))

    for (table, keys), rows in zip(MM3_TABLES, mm3):
        per_table.append((table, _ternary_entries(table, keys, MM3_PARAMS, rows, order)))

    return per_table

def load_pickles(tables_path, metadata_path):
    with open(tables_path, "rb") as f:
        tables = pickle.load(f)

    if not isinstance(tables, list) or len(tables) != 5:
        raise ValueError("Expected a 5-item table pkl from mlp_b_st_pt_to_pkl.")

    if metadata_path:
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)

        pickle_names = [n.split(".", 1)[1] for group in meta["table_names"].values() for n in group]
        layout_names = ([t for t, _ in MM1_TABLES] + [MM2_DIRECT_TABLE[0]]
                        + [t for t, _ in MM2_LEAF_TABLES] + [t for t, _ in MM2_GROUP_TABLES]
                        + [t for t, _ in MM3_TABLES])

        if pickle_names != layout_names:
            raise ValueError("Metadata table names do not match this controller's layout.")

    return tables

# ---------------------------------------------------------------------------
# Dry run: validate against bfrt.json
# ---------------------------------------------------------------------------

def _fits(value, width):
    return 0 <= value < (1 << width)

def dry_run(per_table, bfrt_json_path):
    with open(bfrt_json_path) as f:
        bfrt = {t["name"]: t for t in json.load(f)["tables"]}

    errors  = []
    total   = 0

    for table, entries in per_table:
        spec = bfrt.get(f"pipe.{PIPE}.{table}")

        if spec is None:
            errors.append(f"{table}: not in bfrt.json.")
            continue

        key_widths  = {k["name"]: k["type"].get("width") for k in spec["key"]}
        key_names   = [k["name"] for k in spec["key"] if k["name"] != "$MATCH_PRIORITY"]
        actions     = {a["name"]: a for a in spec["action_specs"]}

        if len(entries) > spec["size"]:
            errors.append(f"{table}: {len(entries)} entries > size {spec['size']}.")

        for entry in entries:
            if [k[0] for k in entry.keys] != key_names:
                errors.append(f"{table}: key fields {[k[0] for k in entry.keys]} != {key_names}.")
                break

            for field, value, mask in entry.keys:
                if not _fits(value, key_widths[field]) or (mask is not None
                                                           and not _fits(mask, key_widths[field])):
                    errors.append(f"{table}: key {field} value/mask exceeds "
                                    f"{key_widths[field]} bits.")
                    break

            action = actions.get(f"{PIPE}.{entry.action}")

            if action is None:
                errors.append(f"{table}: action {entry.action} not in bfrt.json.")
                break

            if [p[0] for p in entry.params] != [d["name"] for d in action["data"]]:
                errors.append(f"{table}: action params {[p[0] for p in entry.params]} != "
                                f"{[d['name'] for d in action['data']]}.")
                break

            if any(_param_width(d["name"]) != d["type"]["width"] for d in action["data"]):
                errors.append(f"{table}: action param widths differ from bfrt.json.")
                break

            if (entry.priority is None) != ("$MATCH_PRIORITY" not in key_widths):
                errors.append(f"{table}: priority presence does not match the table type.")
                break

        total += len(entries)
        print(f"  {table}: {len(entries)} entries (size {spec['size']}).")

    print(f"Total: {total} entries in {len(per_table)} tables.")

    for register in FLOW_REGISTERS:
        spec = bfrt.get(f"pipe.{PIPE}.{register}")

        if spec is None:
            errors.append(f"register {register}: not in bfrt.json.")
            continue

        data_field = f"{PIPE}.{register}.f1"

        if [k["name"] for k in spec["key"]] != [REG_INDEX_KEY]:
            errors.append(f"register {register}: key is not {REG_INDEX_KEY}")
        if data_field not in json.dumps(spec["data"]):
            errors.append(f"register {register}: no data field {data_field}")
        if spec["size"] != FLOW_TABLE_SIZE:
            errors.append(f"register {register}: size {spec['size']} != {FLOW_TABLE_SIZE}")

    print(f"Flow registers checked: {', '.join(FLOW_REGISTERS)}.")

    return errors

# ---------------------------------------------------------------------------
# BFRT programming
# ---------------------------------------------------------------------------

def _import_bfrt_client():
    sde_site = os.path.expandvars("$SDE/install/lib/python3.10/site-packages/tofino/")

    if sde_site not in sys.path:
        sys.path.append(sde_site)

    from bfrt_grpc import client

    return client

def program_table(client, bfrt_info, target, table, entries):
    t       = bfrt_info.table_get(f"{PIPE}.{table}")
    keys    = []
    datas   = []

    for entry in entries:
        key_tuples = [client.KeyTuple(f, int(v)) if m is None else client.KeyTuple(f, int(v), int(m))
                      for f, v, m in entry.keys]

        if entry.priority is not None:
            key_tuples.append(client.KeyTuple("$MATCH_PRIORITY", entry.priority))

        action  = f"{PIPE}.{entry.action}"
        data    = [client.DataTuple(n, int(v) & ((1 << _param_width(n)) - 1))
                   for n, v in entry.params]

        keys.append(t.make_key(key_tuples))
        datas.append(t.make_data(data, action))

    if keys:
        t.entry_add(target, keys, datas)

    print(f"  {table}: {len(keys)} entries.")

# Zeroes every slot of the flow-state registers.
def reset_flow_state(client, bfrt_info, target):
    for register in FLOW_REGISTERS:
        t       = bfrt_info.table_get(f"{PIPE}.{register}")
        keys    = [t.make_key([client.KeyTuple(REG_INDEX_KEY, i)])
                   for i in range(FLOW_TABLE_SIZE)]
        datas   = [t.make_data([client.DataTuple(f"{PIPE}.{register}.f1", 0)])] * FLOW_TABLE_SIZE

        t.entry_mod(target, keys, datas)
        print(f"  {register}: {FLOW_TABLE_SIZE} slots zeroed.")

def main():
    parser = argparse.ArgumentParser(description="Program mlp-b-st tables via BFRT (T2NA)")

    parser.add_argument("--grpc_addr", default="localhost:50052")
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--tables_pkl", required=True,
                        help="Table pkl from mlp_b_st_pt_to_pkl (--output_pkl_path)")
    parser.add_argument("--metadata_pkl",
                        help="Metadata pkl from mlp_b_st_pt_to_pkl, to cross-check table names")
    parser.add_argument("--priority_order", choices=["lower_wins", "higher_wins"],
                        default="lower_wins",
                        help="Which $MATCH_PRIORITY value wins when ternary entries overlap")
    parser.add_argument("--reset_flow_state", action="store_true",
                        help="Zero the flow-state registers before programming the tables")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate entries against --bfrt_json instead of programming a switch")
    parser.add_argument("--bfrt_json", default="p4/mlp-b/mlp-b-st/build/bfrt.json")

    args = parser.parse_args()

    tables      = load_pickles(args.tables_pkl, args.metadata_pkl)
    per_table   = build_entries(tables, args.priority_order)

    if args.dry_run:
        errors = dry_run(per_table, args.bfrt_json)

        for error in errors:
            print(f"ERROR: {error}.")

        sys.exit(1 if errors else 0)

    client      = _import_bfrt_client()
    grpc_client = client.ClientInterface(grpc_addr=args.grpc_addr, client_id=args.client_id,
                                         device_id=args.device_id)
    bfrt_info   = grpc_client.bfrt_info_get(p4_name=None)

    grpc_client.bind_pipeline_config(p4_name=bfrt_info.p4_name)

    target = client.Target()

    if args.reset_flow_state:
        print("Resetting flow state.")
        reset_flow_state(client, bfrt_info, target)

    print(f"Programming mlp-b-st tables (p4={bfrt_info.p4_name}, "
          f"priority_order={args.priority_order}).")

    for table, entries in per_table:
        program_table(client, bfrt_info, target, table, entries)

    print("mlp-b-st tables programmed.")


if __name__ == "__main__":
    main()
