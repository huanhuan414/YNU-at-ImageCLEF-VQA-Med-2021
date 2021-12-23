# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:
# Author:       Boliu.Kelvin
# Date:         2020/4/8
#-------------------------------------------------------------------------------
import os
import time
import torch
import utils
from datetime import datetime
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np

def compute_score_with_logits(logits, labels):

    logits = torch.max(logits, 1)[1].data  # argmax

    one_hots = torch.zeros(*labels.size()).to(logits.device)

    #labels_true = torch.max(labels, 1)[1].data  # argmax
    #print(logits)

    #print(labels_true)

    one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (one_hots * labels)

    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp
# Train phase
def train(args, model, train_loader, eval_loader,s_opt=None, s_epoch=0):
    device = args.device
    model = model.to(device)
    # create packet for output
    # for every train, create a packet for saving .pth and .log
    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join(args.output,run_timestamp)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())
    # Adamax optimizer
    optim = torch.optim.Adamax(params=model.parameters())
    # Scheduler learning rate
    #lr_decay = lr_scheduler.CosineAnnealingLR(optim,T_max=len(train_loader))  # only fit for sgdr

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss()


    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase
    for epoch in range(s_epoch, args.epochs):
        total_loss = 0
        train_score = 0
        number=0
        model.train()

        # Predicting and computing score
        for i, (v, q, a,image_name) in enumerate(train_loader):


            v = v.to(device)

            a = a.to(device)

            preds_close = model(v, q)

            batch_close_score = 0.

            batch_close_score = compute_score_with_logits(preds_close.data, a.data).sum()

            #L1正则化
            # regularization_loss = 0.
            # for param in model.parameters():
            #     regularization_loss += torch.sum(torch.abs(param))

            loss = criterion(preds_close.float(), a.float())
            # loss = classify_loss + args.lamda*regularization_loss

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            total_loss += loss.item()
            train_score += batch_close_score
            number+= v.shape[0]

        print(train_score.item(),number)

        total_loss /= len(train_loader)
        train_score = 100 * train_score / number
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        # Evaluation
        if eval_loader is not None:
            eval_score = evaluate_classifier(model, eval_loader, args,logger)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                # Save the best acc epoch
                model_path = os.path.join(ckpt_path, '{}.pth'.format(best_epoch))
                utils.save_model(model_path, model, best_epoch, optim)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))


# Evaluation
def evaluate_classifier(model, dataloader, args,logger):
    device = args.device
    score = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for i,(v, q, a,image_name) in enumerate(dataloader):

            v = v.to(device)

            a = a.to(device)

            preds_close = model(v, q)

            batch_close_score = 0.

            batch_close_score = compute_score_with_logits(preds_close.float(), a.float()).sum()

            score += batch_close_score

            size = v.shape[0]
            total += size  # batch number

    print(score.item(), total)
    score = 100* score / total

    logger.info('[Validate] Val_Acc:{:.6f}%' .format(score))
    return score
