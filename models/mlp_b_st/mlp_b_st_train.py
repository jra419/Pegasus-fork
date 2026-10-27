import argparse

import torch

from models.autoquanti import quanti_sequence
from models.mlp_b_st.mlp_b_st import MlpStage1, MlpStage2, MlpStage3, MlpStage4

from .mlp_b_st_dataloader import FEAT_COLS, build_eval_loader, build_train_loaders

STAGE_CLASSES = [MlpStage1, MlpStage2, MlpStage3, MlpStage4]


def run(args):
    batch_size  = 256

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    args.device = device

    print(args.device)

    train_loader, val_loader, full_loader = build_train_loaders(args.trainpth,
                                                                batch_size=batch_size,
                                                                shuffle=args.shuffle,
                                                                labels_path=args.train_labels)
    test_loader = build_eval_loader(args.testpth, batch_size=batch_size, shuffle=False,
                                    labels_path=args.test_labels)

    # One digit per quantized layer, e.g. 121 -> MM1 D=1, MM2 D=2, MM3 D=1.
    codeblock_dims  = [int(char) for char in args.dlist]
    stages          = []

    # Build every stage from --modelnum onwards
    # Stage 1 only when starting from scratch.
    for stage, stage_class in enumerate(STAGE_CLASSES, start=1):
        if (stage == 1 and args.modelnum != 1) or args.modelnum > stage:
            continue

        model = stage_class(len(FEAT_COLS), args.mlphs1, args.mlphs2, args.num_classes,
                            device, *codeblock_dims[:stage - 1])

        if (stage == args.modelnum and stage < len(STAGE_CLASSES)
                and args.ptpth is not None and args.trained):
            model.load_state_dict(torch.load(args.ptpth))

        stages.append(model)

    run_dir = f'{args.savepth}/{args.dataset}/{args.cur_run}/'

    taskname = (f'{args.dataset}'
                f'_{args.lr}lr'
                f'_{args.step}s'
                f'_{args.patience}p'
                f'_{args.dlist}D')

    quanti_sequence(args.cur_model, stages, train_loader, val_loader, full_loader, test_loader,
                    device, args.patience, codeblock_dims, run_dir, args.lr, args.step, args,
                    istrained=args.trained, taskname=taskname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--cur_run', type=str, default='')

    # Dataset
    parser.add_argument('--len_vocab', default=1501)
    parser.add_argument('--ipd_vocab', type=int, default=2561)
    parser.add_argument('--dataset', required=True)

    # Model
    parser.add_argument('--cur_model', type=str, default='')
    parser.add_argument('--mlphs1', type=int, default=32)
    parser.add_argument('--mlphs2', type=int, default=16)
    parser.add_argument('--mlphs3', type=int, default=8)
    parser.add_argument('--mlphs4', type=int, default=8)
    parser.add_argument('--mlphs5', type=int, default=8)

    # Training
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--device', choices=['0', '1', '2', '3', 'cpu'], required=True)
    parser.add_argument('--savepth', type=str, default=None, required=True)
    parser.add_argument('--dlist', type=str, default=None, required=True)
    parser.add_argument('--ptpth', type=str, default=None)
    parser.add_argument('--trainpth', type=str, required=True)
    parser.add_argument('--testpth', type=str, required=True)
    parser.add_argument('--train_labels', type=str, default=None,
                        help='Labels for a pcap --trainpth (default: <name>-labels.csv)')
    parser.add_argument('--test_labels', type=str, default=None,
                        help='Labels for a pcap --testpth (default: <name>-labels.csv)')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--trained', type=lambda x: x.lower() in ('true', '1', 'yes'), default=False)
    parser.add_argument('--shuffle', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)
    parser.add_argument('--modelnum', type=int, default=0, required=True)
    parser.add_argument('--confirmterm', type=int, default=10)

    args = parser.parse_args()
    run(args)
