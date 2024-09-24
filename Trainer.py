import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
import datetime
import math
from sklearn.metrics import confusion_matrix
import warnings

from sam import SAM


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, weighted_train_loader=None,
                 per_class_num=[], log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.label_weighting = args.label_weighting
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.per_cls_weights = None
        self.cls_num_list = per_class_num
        self.contrast_weight = args.contrast_weight
        self.model = model
        if args.rho > 0:
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(base_optimizer=base_optimizer, rho=args.rho, params=model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
            self.rho = args.rho
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr,
                                             weight_decay=args.weight_decay)
            self.rho = args.rho

        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log
        self.beta = args.beta
        self.update_weight()

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')

            # switch to train mode
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets) in enumerate(self.train_loader):

                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1),
                                                                                           1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # Data augmentation
                lam = np.random.beta(self.beta, self.beta)

                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1,
                                                                                         org2=input_org_2,
                                                                                         invs1=input_invs_1,
                                                                                         invs2=input_invs_2,
                                                                                         label_org=one_hot_org,
                                                                                         label_invs=one_hot_invs,
                                                                                         label_org_w=one_hot_org_w,
                                                                                         label_invs_w=one_hot_invs_w)

                output_1, output_cb_1, z1, p1 = self.model(mix_x, mixup_y, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, mixcut_y, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                # FSGL
                score_thr = 0.7
                device_1 = output_1.device
                n_i_1, n_c_1 = output_1.size()
                with torch.no_grad():
                    sigmoid_output_1 = torch.sigmoid(output_1)
                high_score_inds_1 = torch.nonzero(sigmoid_output_1 >= score_thr, as_tuple=False)
                weight_mask_1 = torch.sparse_coo_tensor(high_score_inds_1.t(),
                                                        output_1.new_ones(high_score_inds_1.shape[0]),
                                                        size=(n_i_1, n_c_1), device=device_1).to_dense()
                inds_1 = torch.nonzero(mixup_y != 0, as_tuple=False)
                for ind in inds_1:
                    weight_mask_1[ind[0]][ind[1]] = 1

                device_2 = output_2.device
                n_i_2, n_c_2 = output_2.size()
                with torch.no_grad():
                    sigmoid_output_2 = torch.sigmoid(output_2)
                high_score_inds_2 = torch.nonzero(sigmoid_output_2 >= score_thr, as_tuple=False)
                weight_mask_2 = torch.sparse_coo_tensor(high_score_inds_2.t(),
                                                        output_2.new_ones(high_score_inds_2.shape[0]),
                                                        size=(n_i_2, n_c_2), device=device_2).to_dense()
                inds_2 = torch.nonzero(mixcut_y != 0, as_tuple=False)
                for ind in inds_2:
                    weight_mask_2[ind[0]][ind[1]] = 1

                device_1_w = output_cb_1.device
                n_i_1_w, n_c_1_w = output_cb_1.size()
                with torch.no_grad():
                    sigmoid_output_cb_1 = torch.sigmoid(output_cb_1)
                high_score_inds_1_w = torch.nonzero(sigmoid_output_cb_1 >= score_thr, as_tuple=False)
                weight_mask_1_w = torch.sparse_coo_tensor(high_score_inds_1_w.t(),
                                                          output_cb_1.new_ones(high_score_inds_1_w.shape[0]),
                                                          size=(n_i_1_w, n_c_1_w), device=device_1_w).to_dense()
                inds_1_w = torch.nonzero(mixup_y_w != 0, as_tuple=False)
                for ind in inds_1_w:
                    weight_mask_1_w[ind[0]][ind[1]] = 1

                device_2_w = output_cb_2.device
                n_i_2_w, n_c_2_w = output_cb_2.size()
                with torch.no_grad():
                    sigmoid_output_cb_2 = torch.sigmoid(output_cb_2)
                high_score_inds_2_w = torch.nonzero(sigmoid_output_cb_2 >= score_thr, as_tuple=False)
                weight_mask_2_w = torch.sparse_coo_tensor(high_score_inds_2_w.t(),
                                                          output_cb_2.new_ones(high_score_inds_2_w.shape[0]),
                                                          size=(n_i_2_w, n_c_2_w), device=device_2_w).to_dense()
                inds_2_w = torch.nonzero(cutmix_y_w != 0, as_tuple=False)
                for ind in inds_2_w:
                    weight_mask_2_w[ind[0]][ind[1]] = 1

                # loss_mix = F.binary_cross_entropy_with_logits(output_1, mixup_y.float(), reduction='none')
                # loss_mix = torch.sum(loss_mix) / n_i_1
                # loss_cut = F.binary_cross_entropy_with_logits(output_2, mixcut_y.float(), reduction='none')
                # loss_cut = torch.sum(loss_cut) / n_i_2
                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))

                # loss_mix_w = F.binary_cross_entropy_with_logits(output_cb_1, mixup_y_w.float(), reduction='none')
                # loss_mix_w = torch.sum(weight_mask_1_w * loss_mix_w) / n_i_1_w
                # loss_cut_w = F.binary_cross_entropy_with_logits(output_cb_2, cutmix_y_w.float(), reduction='none')
                # loss_cut_w = torch.sum(weight_mask_2_w * loss_cut_w) / n_i_2_w
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                # acsl_loss_mix = F.binary_cross_entropy_with_logits(output_1, mixup_y.float(), reduction='none')
                # acsl_loss_mix = torch.sum(weight_mask_1 * acsl_loss_mix) / n_i_1
                # acsl_loss_cut = F.binary_cross_entropy_with_logits(output_2, mixcut_y.float(), reduction='none')
                # acsl_loss_cut = torch.sum(weight_mask_2 * acsl_loss_cut) / n_i_2
                # acsl_loss = acsl_loss_mix + acsl_loss_cut

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss  # + acsl_loss

                losses.update(loss.item(), inputs[0].size(0))

                # compute gradient and do SGD step
                if self.rho > 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # Second step of SAM
                else:
                    loss += self.contrast_weight * contrastive_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.rho > 0:  # Do second step of SAM
                    output_1, output_cb_1, z1, p1 = self.model(mix_x, mixup_y, train=True)
                    output_2, output_cb_2, z2, p2 = self.model(cut_x, mixcut_y, train=True)
                    contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                    score_thr = 0.7
                    device_1 = output_1.device
                    n_i_1, n_c_1 = output_1.size()
                    with torch.no_grad():
                        sigmoid_output_1 = torch.sigmoid(output_1)
                    high_score_inds_1 = torch.nonzero(sigmoid_output_1 >= score_thr, as_tuple=False)
                    weight_mask_1 = torch.sparse_coo_tensor(high_score_inds_1.t(),
                                                            output_1.new_ones(high_score_inds_1.shape[0]),
                                                            size=(n_i_1, n_c_1), device=device_1).to_dense()
                    inds_1 = torch.nonzero(mixup_y != 0, as_tuple=False)
                    for ind in inds_1:
                        weight_mask_1[ind[0]][ind[1]] = 1

                    device_2 = output_2.device
                    n_i_2, n_c_2 = output_2.size()
                    with torch.no_grad():
                        sigmoid_output_2 = torch.sigmoid(output_2)
                    high_score_inds_2 = torch.nonzero(sigmoid_output_2 >= score_thr, as_tuple=False)
                    weight_mask_2 = torch.sparse_coo_tensor(high_score_inds_2.t(),
                                                            output_2.new_ones(high_score_inds_2.shape[0]),
                                                            size=(n_i_2, n_c_2), device=device_2).to_dense()
                    inds_2 = torch.nonzero(mixcut_y != 0, as_tuple=False)
                    for ind in inds_2:
                        weight_mask_2[ind[0]][ind[1]] = 1

                    device_1_w = output_cb_1.device
                    n_i_1_w, n_c_1_w = output_cb_1.size()
                    with torch.no_grad():
                        sigmoid_output_cb_1 = torch.sigmoid(output_cb_1)
                    high_score_inds_1_w = torch.nonzero(sigmoid_output_cb_1 >= score_thr, as_tuple=False)
                    weight_mask_1_w = torch.sparse_coo_tensor(high_score_inds_1_w.t(),
                                                              output_cb_1.new_ones(high_score_inds_1_w.shape[0]),
                                                              size=(n_i_1_w, n_c_1_w), device=device_1_w).to_dense()
                    inds_1_w = torch.nonzero(mixup_y_w != 0, as_tuple=False)
                    for ind in inds_1_w:
                        weight_mask_1_w[ind[0]][ind[1]] = 1

                    device_2_w = output_cb_2.device
                    n_i_2_w, n_c_2_w = output_cb_2.size()
                    with torch.no_grad():
                        sigmoid_output_cb_2 = torch.sigmoid(output_cb_2)
                    high_score_inds_2_w = torch.nonzero(sigmoid_output_cb_2 >= score_thr, as_tuple=False)
                    weight_mask_2_w = torch.sparse_coo_tensor(high_score_inds_2_w.t(),
                                                              output_cb_2.new_ones(high_score_inds_2_w.shape[0]),
                                                              size=(n_i_2_w, n_c_2_w), device=device_2_w).to_dense()
                    inds_2_w = torch.nonzero(cutmix_y_w != 0, as_tuple=False)
                    for ind in inds_2_w:
                        weight_mask_2_w[ind[0]][ind[1]] = 1

                    # loss_mix = F.binary_cross_entropy_with_logits(output_1, mixup_y.float(), reduction='none')
                    # loss_mix = torch.sum(loss_mix) / n_i_1
                    # loss_cut = F.binary_cross_entropy_with_logits(output_2, mixcut_y.float(), reduction='none')
                    # loss_cut = torch.sum(loss_cut) / n_i_2
                    loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                    loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))

                    # loss_mix_w = F.binary_cross_entropy_with_logits(output_1, mixup_y.float(), reduction='none')
                    # loss_mix_w = torch.sum(weight_mask_1 * loss_mix_w) / n_i_1
                    # loss_cut_w = F.binary_cross_entropy_with_logits(output_2, mixcut_y.float(), reduction='none')
                    # loss_cut_w = torch.sum(weight_mask_2 * loss_cut_w) / n_i_2
                    # loss_mix_w = F.binary_cross_entropy_with_logits(output_cb_1, mixup_y_w.float(), reduction='none')
                    # loss_mix_w = torch.sum(weight_mask_1_w * loss_mix_w) / n_i_1_w
                    # loss_cut_w = F.binary_cross_entropy_with_logits(output_cb_2, cutmix_y_w.float(), reduction='none')
                    # loss_cut_w = torch.sum(weight_mask_2_w * loss_cut_w) / n_i_2_w
                    loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                    loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                    # acsl_loss_mix = F.binary_cross_entropy_with_logits(output_1, mixup_y.float(), reduction='none')
                    # acsl_loss_mix = torch.sum(weight_mask_1 * acsl_loss_mix) / n_i_1
                    # acsl_loss_cut = F.binary_cross_entropy_with_logits(output_2, mixcut_y.float(), reduction='none')
                    # acsl_loss_cut = torch.sum(weight_mask_2 * acsl_loss_cut) / n_i_2
                    # acsl_loss = acsl_loss_mix + acsl_loss_cut

                    balance_loss = loss_mix + loss_cut
                    rebalance_loss = loss_mix_w + loss_cut_w

                    loss = alpha * balance_loss + (
                            1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss  # + acsl_loss # hardcoding contrast weight for analysis

                    loss.backward()
                    self.optimizer.second_step(zero_grad=True)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))  # TODO
                    print(output)
                    # evaluate on validation set
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.paco_adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch + 1)

    def validate(self, epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = self.model(input, train=False, label=None)

                # measure accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = (
                'EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1,
                                                                                                    flag='val',
                                                                                                    top1=top1,
                                                                                                    top5=top5))

            self.log.info(output)
            out_cls_acc = '%s Class Accuracy: %s' % (
                'val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg

    def SimSiamLoss(self, p, z, version='simplified'):  # negative cosine similarity
        z = z.detach()  # stop gradient

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def paco_adjust_learning_rate(self, optimizer, epoch, args):
        warmup_epochs = 10
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
