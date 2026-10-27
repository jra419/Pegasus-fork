import os
import copy
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score

from util.early_stopping import EarlyStopping
from util.quanti import model_quanti
from util.getdata import get_data

EARLY_STOP          = 1
BEST_SCORE_UPDATED  = 2


def _format_metric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'

    return f'{value:.4f}'

def _macro_roc_auc(labels, probs):
    labels  = np.asarray(labels)
    probs   = np.asarray(probs, dtype=np.float64)

    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float('nan')

    n_classes = probs.shape[1]

    if n_classes == 2:
        return float(roc_auc_score(labels, probs[:, 1]))

    return float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))

def train(optimizer, model, data):
    model.train()

    n = 0

    for x_batch, label_batch in data:
        x_batch     = x_batch
        label_batch = label_batch.to(model.device)

        output, quantidata  = model(x_batch)
        loss                = F.nll_loss(output, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n += 1

def test(model, data):
    test_loss   = 0
    correct     = 0

    model.eval()

    nsum        = 0
    all_preds   = []
    all_labels  = []
    all_probs   = []

    model.eval()

    with torch.no_grad():
        for x_batch, label_batch in data:
            x_batch     = x_batch
            label_batch = label_batch.to(model.device)

            nsum += label_batch.shape[0]

            output, quantidata  = model(x_batch)
            test_loss          += F.nll_loss(output, label_batch, reduction='sum')
            pred                = torch.max(output, dim=-1, keepdim=False)[-1]
            correct            += pred.eq(label_batch.data).sum()

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())
            all_probs.append(torch.exp(output).cpu().numpy())

        test_loss       = test_loss / nsum
        all_preds       = np.concatenate(all_preds)
        all_labels      = np.concatenate(all_labels)
        all_probs       = np.concatenate(all_probs, axis=0)
        f1_macro        = f1_score(all_labels, all_preds, average='macro')
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        auc_macro       = _macro_roc_auc(all_labels, all_probs)

        return 100. * correct / nsum, test_loss, f1_macro, precision_macro, recall_macro, auc_macro

def quanti_sequence(modelname, modellist, train_loader, val_loader, known_loader, test_loader,
                    device, patience, dlist, savepth, lr, step, args, istrained=True,
                    taskname='NONE'):

    os.makedirs(f'{savepth}/weight', exist_ok=True)

    if not os.path.isfile(f'{savepth}/args.txt'):
        with open(f'{savepth}/args.txt', 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)

    weights = []
    num     = 1

    print('Model quant. stages: {} | lr: {} |step: {} | patience: {} | D: {}'.format(
            len(modellist), lr, step, patience,''.join(map(str, dlist))), flush=True)

    for model in modellist:
        early_stop  = EarlyStopping(patience=patience, delta=0, verbose=False)
        acc         = 0
        f1          = 0
        epoch       = 0

        while True:
            #print(model.state_dict().keys())
            if istrained:
                break
            #optimizer = optim.AdamW(model.parameters(),lr=lr)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

            if epoch != 0:
                train(optimizer,model,train_loader)

            epoch      += 1
            tempacc     = 0

            tempacc, loss, f1_macro, _, _, _ = test(model, val_loader)
            status                          = early_stop(loss)

            print('modelnum: {} {} | epoch: {} | acc: {:.4f} | loss: {:.4f} | status: {} | f1: {:.4f} | flagacc: {:.4f} | flagf1: {:.4f}'.format(modelname, args.modelnum+num-1, epoch, tempacc, loss, status, f1_macro, acc, f1), flush=True)

            if status == EARLY_STOP:
                break

            if f1_macro > f1:
                weights.append(copy.deepcopy(model.state_dict()))

                acc = tempacc
                f1  = f1_macro

                torch.save(
                        weights[-1], f'{savepth}/weight/{modelname}_{num+args.modelnum-1}_temp.pt')

            if epoch % step == 0:
                lr         /= 2
                optimizer   = optim.AdamW(model.parameters(), lr=lr)

            if epoch % args.confirmterm == 0:
                acc = tempacc

                confirm_acc, loss, confirm_f1_macro, _, _, confirm_auc = test(model, test_loader)

                print(
                    f'f1 {f1_macro} confirm f1: {confirm_f1_macro} '
                    f'confirm auc: {_format_metric(confirm_auc)}',
                    flush=True,
                )

                torch.save(model.state_dict(),
                           f'{savepth}/weight/{modelname}_{num+args.modelnum-1}_{epoch}epoch.pt')

        if istrained:
            weights.append(copy.deepcopy(model.state_dict()))
            istrained = False
            stage = num + args.modelnum - 1
            os.makedirs(f'{savepth}/weight', exist_ok=True)
            torch.save(
                weights[-1],
                f'{savepth}/weight/{modelname}_{num+args.modelnum-1}_best.pt',
            )
            print(
                f'istrained: saved {savepth}/weight/{modelname}_{num+args.modelnum-1}_best.pt '
                f'(stage {stage}, skipped training loop)',
                flush=True,
            )

        else:
            model.load_state_dict(weights[-1])
            stage = num + args.modelnum - 1
            val_acc, val_loss, val_f1, _, _, val_auc = test(model, val_loader)
            test_acc, test_loss, test_f1, test_pr, test_re, test_auc = test(model, test_loader)

            print(f'=== Final evaluation {modelname} stage {stage} ===', flush=True)
            print(f'val:  acc={val_acc:.4f} f1={val_f1:.4f} auc={_format_metric(val_auc)}',
                  flush=True)
            print(f'test: acc={test_acc:.4f} f1={test_f1:.4f} '
                  f'precision={test_pr:.4f} recall={test_re:.4f} '
                  f'auc={_format_metric(test_auc)}',
                  flush=True)
            print(f'best val f1 (checkpoint): {f1:.4f}', flush=True)

            metrics = {
                'modelname': modelname,
                'stage': stage,
                'val': {
                    'acc': val_acc,
                    'loss': float(val_loss),
                    'f1': val_f1,
                    'auc': val_auc,
                },
                'test': {
                    'acc': test_acc,
                    'loss': float(test_loss),
                    'f1': test_f1,
                    'precision': test_pr,
                    'recall': test_re,
                    'auc': test_auc,
                },
                'best_val_f1': f1,
            }
            with open(f'{savepth}/eval_metrics_stage{stage}.json', 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

            torch.save(weights[-1],
                       f'{savepth}/weight/{modelname}_{num+args.modelnum-1}_best.pt')

        weight = weights[-1]

        if num != len(modellist):
            train_data  = get_data(model, known_loader, weight)
            pth         = savepth

            if len(train_data) == 1:
                if modelname == 'mlp_b_1':
                    mm_idx = num
                    dlist_idx = num - 1
                    S, T, LUT = model_quanti(train_data[0], dlist[dlist_idx], mm_idx, pth)
                    weight[f'MM{mm_idx}.S'] = S
                    weight[f'MM{mm_idx}.LUT'] = LUT
                    weight[f'MM{mm_idx}.T'] = T
                    weight[f'MM{mm_idx}.H'] = torch.load('./util/H.pth', weights_only=False)
                elif modelname != 'cnn_l' or args.modelnum == 1:
                    S, T, LUT = model_quanti(train_data[0],dlist[args.modelnum+num-2],
                                             args.modelnum+num-1,pth)

                    weight[f'MM{args.modelnum+num-1}.S']    = S
                    weight[f'MM{args.modelnum+num-1}.LUT']  = LUT
                    weight[f'MM{args.modelnum+num-1}.T']    = T
                    weight[f'MM{args.modelnum+num-1}.H']    = \
                            torch.load('./util/H.pth', weights_only=False)

                else:
                    S, T, LUT = model_quanti(train_data[0],dlist[args.modelnum+num-3],
                                             args.modelnum+num-2,pth)

                    weight[f'MM{args.modelnum+num-2}.S']    = S
                    weight[f'MM{args.modelnum+num-2}.LUT']  = LUT
                    weight[f'MM{args.modelnum+num-2}.T']    = T
                    weight[f'MM{args.modelnum+num-2}.H']    = \
                            torch.load('./util/H.pth', weights_only=False)

            else:
                for i in range(len(train_data)):
                    S, T, LUT = model_quanti(train_data[i],dlist[args.modelnum+num-2],
                                             args.modelnum+num-1,pth)

                    weight[f'MM{num+args.modelnum-1}_{i+1}.S']      = S
                    weight[f'MM{num+args.modelnum-1}_{i+1}.LUT']    = LUT
                    weight[f'MM{num+args.modelnum-1}_{i+1}.T']      = T
                    weight[f'MM{num+args.modelnum-1}_{i+1}.H']      = torch.load('./util/H.pth')

            nextmodel   = modellist[num]
            nextmodel.load_state_dict(weight)

            num        +=1
