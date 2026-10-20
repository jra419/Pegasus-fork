import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score

from models.mlp_b.mlp_b_st.mlp_b_st import HIDDEN_SIZE_1, HIDDEN_SIZE_2, MlpLut
from util.reset import (
    adjust_norm_layer,
    reset_fc,
    reset_int,
    reset_int_single,
    reset_norm_to_LUT,
    reset_norm_to_T,
    reset_relu,
)

from .mlp_b_st_dataloader import FEAT_COLS, build_eval_loader


# Fold the float layers of a stage-4 state dict into the MM1..MM3 tables (still float).
def fold_layers_into_tables(state, device):
    lut_state = {}

    lut_state["MM1.S"]   = state["MM1.S"]
    lut_state["MM1.H"]   = state["MM1.H"]
    lut_state["MM1.LUT"] = state["MM1.LUT"]

    lut_state["MM2.S"]   = state["MM2.S"]
    lut_state["MM2.H"]   = state["MM2.H"]
    lut_state["MM2.LUT"] = state["MM2.LUT"]

    lut_state["MM3.S"]   = state["MM3.S"]
    lut_state["MM3.H"]   = state["MM3.H"]

    # Each BatchNorm as a per-feature affine transform: y = scale * x + shift.
    norm0_scale, norm0_shift = adjust_norm_layer(state["norm0.weight"],
                                                 state["norm0.bias"],
                                                 state["norm0.running_var"],
                                                 state["norm0.running_mean"])

    norm1_scale, norm1_shift = adjust_norm_layer(state["norm1.weight"],
                                                 state["norm1.bias"],
                                                 state["norm1.running_var"],
                                                 state["norm1.running_mean"])

    norm2_scale, norm2_shift = adjust_norm_layer(state["norm2.weight"],
                                                 state["norm2.bias"],
                                                 state["norm2.running_var"],
                                                 state["norm2.running_mean"])

    # Linear layer after each quantizer -> that quantizer's LUT.
    lut_state["MM1.LUT"] = reset_fc(state["fc1.weight"],
                                    state["fc1.bias"],
                                    state["MM1.LUT"],
                                    device)

    lut_state["MM2.LUT"] = reset_fc(state["fc2.weight"],
                                    state["fc2.bias"],
                                    state["MM2.LUT"],
                                    device)

    lut_state["MM3.LUT"] = reset_fc(state["output.weight"],
                                    state["output.bias"],
                                    state["MM3.LUT"],
                                    device)

    # BatchNorm after fc1/fc2 -> MM1/MM2 LUTs.
    lut_state["MM1.LUT"] = reset_norm_to_LUT(norm1_scale,
                                             norm1_shift,
                                             lut_state["MM1.LUT"],
                                             device)

    lut_state["MM2.LUT"] = reset_norm_to_LUT(norm2_scale,
                                             norm2_shift,
                                             lut_state["MM2.LUT"],
                                             device)

    # ReLU before MM2/MM3 -> their thresholds.
    lut_state["MM2.T"] = reset_relu(state["MM2.T"])
    lut_state["MM3.T"] = reset_relu(state["MM3.T"])

    # Input BatchNorm -> MM1 thresholds, so MM1 compares raw feature values.
    lut_state["MM1.T"] = reset_norm_to_T(state["MM1.T"],
                                         state["MM1.S"],
                                         norm0_scale,
                                         norm0_shift)

    return lut_state

# Integer-quantize a LUT together with the next layer's thresholds (one shared scale, so the
# comparisons stay consistent). The switch adds one leaf row per codeblock into each packed
# hidden value, so the scale makes the worst-case sum of those rows fit in $BITS magnitude
# bits (a signed bits+1 lane) and can never overflow. Rounding adds at most 0.5 per codeblock,
# so the budget is reduced by one per codeblock.
def quantize_for_lanes(LUT, T, bits):
    finite_T    = T[torch.isfinite(T)]
    max_T       = finite_T.abs().max() if finite_T.numel() > 0 else torch.tensor(0.0)
    worst_sum   = LUT.abs().amax(dim=1).sum(dim=0).max()
    limit       = torch.max(worst_sum, max_T)
    budget      = (2 ** bits - 1) - LUT.shape[0]
    scalar      = budget / limit if limit != 0 else torch.tensor(1.0)

    T_quant = T.clone()
    T_quant[~torch.isfinite(T_quant)] = 0

    return reset_int(LUT, scalar), reset_int(T_quant, scalar)

# Integer-quantize the tables. MM1/MM2 outputs are packed into lanes on the switch, so their
# LUTs are scaled to fit them; MM3 outputs are unpacked 32-bit logits.
def quantize_tables(lut_state, lut_bits_mm1, lut_bits_mm2, lut_bits_mm3):
    lut_state["MM1.T"] = reset_int(lut_state["MM1.T"], 1)

    lut_state["MM1.LUT"], lut_state["MM2.T"] = quantize_for_lanes(lut_state["MM1.LUT"],
                                                                  lut_state["MM2.T"],
                                                                  lut_bits_mm1)

    lut_state["MM2.LUT"], lut_state["MM3.T"] = quantize_for_lanes(lut_state["MM2.LUT"],
                                                                  lut_state["MM3.T"],
                                                                  lut_bits_mm2)

    lut_state["MM3.LUT"] = reset_int_single(lut_state["MM3.LUT"], lut_bits_mm3)

    return lut_state

def eval_f1(lut_state, codeblock_dims, num_classes, test_path, device, test_labels=None):
    model = MlpLut(len(FEAT_COLS), HIDDEN_SIZE_1, HIDDEN_SIZE_2, num_classes, device,
                   *codeblock_dims)
    model.load_state_dict(lut_state)

    test_loader = build_eval_loader(test_path, batch_size=10000, shuffle=False,
                                    labels_path=test_labels)

    all_preds   = []
    all_labels  = []

    model.eval()

    with torch.no_grad():
        for x_batch, label_batch in test_loader:
            label_batch = label_batch.to(model.device)
            output      = model(x_batch)
            pred        = torch.max(output, dim=-1, keepdim=False)[-1]

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_labels  = np.concatenate(all_labels)

    return f1_score(all_labels, all_preds, average="macro")

def main(args):
    device          = torch.device("cpu")
    state           = torch.load(args.ptpth, map_location=device)
    codeblock_dims  = [int(char) for char in args.dlist]

    lut_state   = fold_layers_into_tables(state, device)

    # Integer logits on the switch are about s times the model's logits; a confidence threshold
    # tau in model logit units becomes round(tau * s) (CONF_THRESHOLD in the P4).
    logit_scale = (2 ** args.sn3 - 1) / float(lut_state["MM3.LUT"].abs().max())
    print(f"MM3 logit scale s: {logit_scale:.6f}")

    lut_state   = quantize_tables(lut_state, args.sn1, args.sn2, args.sn3)
    f1_macro    = eval_f1(lut_state, codeblock_dims, args.num_classes, args.testpth, device,
                          args.test_labels)

    # The scale is saved alongside the tables for mlp_b_st_pt_to_pkl's confidence check.
    if args.savepth is not None:
        torch.save({**lut_state, "logit_scale": torch.tensor(logit_scale)}, args.savepth)

    return f1_macro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--len_vocab", default=1501)
    parser.add_argument("--len_embedding_bits", default=10)
    parser.add_argument("--ipd_vocab", type=int, default=2561)
    parser.add_argument("--ipd_embedding_bits", type=int, default=8)
    parser.add_argument("--dlist", type=str, default=None, required=True)
    parser.add_argument("--ptpth", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--testpth", type=str, required=True, help="Per-packet CSV or pcap")
    parser.add_argument("--test_labels", type=str, default=None,
                        help="Labels for a pcap --testpth (default: <name>-labels.csv)")
    parser.add_argument("--savepth", type=str)
    # Signed integer bits for each layer's LUT.
    parser.add_argument("--sn1", type=int, default=12)
    parser.add_argument("--sn2", type=int, default=12)
    parser.add_argument("--sn3", type=int, default=12)

    args = parser.parse_args()

    f1_macro = main(args)
    print(f"F1 Macro: {f1_macro}")
