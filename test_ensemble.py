"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

def getModel(word_vectors,
            char_vectors,
            hidden_size,
            use_char_emb,
            use_attention,
            use_dynamic_decoder,
            log):
    log.info('Building model...')
    model = BiDAF(word_vectors=word_vectors,
                  char_vectors = char_vectors,
                  hidden_size=hidden_size,
                  use_char_emb=use_char_emb,
                  use_dynamic_coattention = False,
                  use_self_attention = False,
                  use_attention = use_attention,
                  use_dynamic_decoder=use_dynamic_decoder)
    return model

def ensemble(log_p1_models, log_p2_models, loss_models, args):
    # Perform ensemble and select starts and end indexes for whole batch, combinging probs from each model.
    # Discretize will be called in this method.
    # shape log_p1_models : (n_models, batch_size, seq_len)
    ans_starts = []
    ans_edns = []

    n_models = len(log_p1_models)
    batch_size = len(log_p1_models[0])


    # for i in range(batch_size):
    #     # for ith data point, get probs for each model.
    #     log_p1_model = log_p1_models[:, i]
    #     log_p2_model = log_p2_models[:, i]

    #     p1, p2 = log_p1_model.exp(), log_p2_model.exp()

    #     starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

    # select 1st model for now.
    log_p1 = log_p1_models[0]
    log_p2 = log_p2_models[0]
    loss = loss_models[0]
    p1, p2 = log_p1.exp(), log_p2.exp()
    starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

    return starts, ends, loss


def main(args_list):
    
    # common args, pull from first configuration.
    args = args_list[0]

    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    char_vectors = util.torch_from_json(args.char_emb_file)
    models = []

    for args_model in args_list:
        # Get model
        model = getModel(word_vectors, char_vectors, args_model.hidden_size,
                    args_model.use_char_emb,
                    use_attention = args_model.use_attention,
                    use_dynamic_decoder=args_model.use_dynamic_decoder,
                    log=log)

        model = nn.DataParallel(model, gpu_ids)
        log.info(f'Loading checkpoint from {args_model.load_path}...')
        model = util.load_model(model, args_model.load_path, gpu_ids, return_step=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            log_p1_models = []
            log_p2_models = []
            loss_models = []

            for model in models:
                # Forward
                log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                y1, y2 = y1.to(device), y2.to(device)

                st_idx_i_1 = None
                end_idx_i_1 = None
                curr_mask_1 = None
                curr_mask_2 = None

                if len(log_p1.shape) == 3:
                    n_iter = log_p1.shape[1]
                    aggregated_loss = 0
                    for i in range(n_iter):
                        log_p1_i = log_p1[:,i,:]
                        log_p2_i = log_p2[:,i,:]

                        step_loss1 = F.nll_loss(log_p1_i, y1, reduce=False) 
                        step_loss2 =  F.nll_loss(log_p2_i, y2, reduce=False)
                        
                        _, st_idx_i = torch.max(log_p1_i, dim=1)
                        _, end_idx_i = torch.max(log_p2_i, dim=1)
                        
                        if curr_mask_1 == None:
                            curr_mask_1 = (st_idx_i == st_idx_i)
                            curr_mask_2 = (end_idx_i == end_idx_i)
                        else:
                            curr_mask_1 = (st_idx_i != st_idx_i_1) * curr_mask_1
                            curr_mask_2 = (end_idx_i != end_idx_i_1) * curr_mask_2
                        
                        # print('iteration {} mask1: {}, mask2: {}'.format(i, curr_mask_1, curr_mask_2))
                        # print('st_idx: {}, end: {}'.format(st_idx_i, end_idx_i))
                        # print(step_loss1, step_loss2)

                        step_loss1 = step_loss1 * curr_mask_1.float()
                        step_loss2 = step_loss2 * curr_mask_2.float()
                        

                        aggregated_loss += (step_loss1 + step_loss2)
                        # print(aggregated_loss)

                        st_idx_i_1 = st_idx_i
                        end_idx_i_1 = end_idx_i
                    
                    loss = torch.mean(aggregated_loss)
                    # print('aggregated loss: {}'.format(loss))
                    log_p1 = log_p1[:,-1,:] # take prob of last iteration for EM, F1 scores and predictions.
                    log_p2 = log_p2[:,-1,:]
                else:
                    loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                
                # Add the log probs and losses from each model to these lists.
                log_p1_models.append(log_p1)
                log_p2_models.append(log_p2)
                loss_models.append(loss)


            starts, ends, loss =  ensemble(log_p1_models, log_p2_models, loss_models, args)

            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            # p1, p2 = log_p1.exp(), log_p2.exp()
            # starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    checkpoints = ["save/train/bidaf-char-02/best.pth.tar"]
    num_models = len(checkpoints)
    args_list = []
    for i in range(num_models):
        args = get_test_args()
        args.load_path = checkpoints[i]

        # Override some args for each model/
        if i == 0:
            args.use_char_emb = True
            args.use_attention = False
            args.use_dynamic_decoder = False

        args_list.append(args)

    main(args_list)
