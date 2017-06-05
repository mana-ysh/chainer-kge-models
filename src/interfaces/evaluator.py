
import numpy as np
import sys

sys.path.append('../')
from utils.dataset import batch_iter, PathQueryDataset, TripletDataset, bucket_batch_iter


BATCHSIZE = 100

class Evaluator(object):
    def __init__(self, metric, nbest=None):
        assert metric in ['mr', 'mrr', 'hits'], 'Invalid metric: {}'.format(metric)
        if metric == 'hits':
            assert nbest, 'Please indicate n-best in using hits'
        self.metric = metric
        self.nbest = nbest
        self.batchsize = BATCHSIZE
        self.ress = []

    def run(self, model, dataset):
        if isinstance(dataset, TripletDataset):
            if self.metric == 'mr':
                res = self.cal_mr(model, dataset)
            elif self.metric == 'mrr':
                res = self.cal_mrr(model, dataset)
                # res = self.cal_batch_mrr(model, dataset)
            elif self.metric == 'hits':
                res = self.cal_hits(model, dataset, self.nbest)
            else:
                raise ValueError
        elif isinstance(dataset, PathQueryDataset):
            if self.metric == 'mr':
                res = self.cal_path_mr(model, dataset)
            elif self.metric == 'mrr':
                res = self.cal_path_mrr(model, dataset)
                # res = self.cal_batch_mrr(model, dataset)
            elif self.metric == 'hits':
                res = self.cal_path_hits(model, dataset, self.nbest)
            else:
                raise ValueError
        else:
            raise ValueError('Invalid dataset type: {}'.format(dataset))
        self.ress.append(res)
        return res

    def cal_mr(self, model, dataset):
        n_sample = len(dataset)
        sum_r = 0.
        for subs, rels, objs in batch_iter(dataset, self.batchsize, rand_flg=False):
            # subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            scores = model.cal_scores(subs.astype(np.int32), rels.astype(np.int32))
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_rr += sum(rank for rank in ranks)
        return float(sum_r/n_sample)

    def cal_path_mr(self, model, dataset):
        n_sample = len(dataset)
        sum_r = 0.
        for subs, rels, objs in bucket_batch_iter(dataset, self.batchsize, rand_flg=False):
            scores = model.cal_path_scores(np.array(subs).astype(np.int32), np.array(rels).astype(np.int32))
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_rr += sum(rank for rank in ranks)
        return float(sum_r/n_sample)

    def cal_mrr(self, model, dataset):
        n_sample = len(dataset)
        sum_rr = 0.
        for subs, rels, objs in batch_iter(dataset, self.batchsize, rand_flg=False):
            # subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            scores = model.cal_scores(subs.astype(np.int32), rels.astype(np.int32))
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_rr += sum(float(1/rank) for rank in ranks)
        return float(sum_rr/n_sample)

    def cal_path_mrr(self, model, dataset):
        n_sample = len(dataset)
        sum_rr = 0.
        for subs, rels, objs in bucket_batch_iter(dataset, self.batchsize, rand_flg=False):
            scores = model.cal_path_scores(np.array(subs).astype(np.int32), np.array(rels).astype(np.int32))
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_rr += sum(float(1/rank) for rank in ranks)
        return float(sum_rr/n_sample)

    def cal_hits(self, model, dataset, nbest):
        n_sample = len(dataset)
        n_corr = 0
        for subs, rels, objs in batch_iter(dataset, self.batchsize, rand_flg=False):
            # subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            scores = model.cal_scores(subs.astype(np.int32), rels.astype(np.int32))
            res = np.flip(np.argsort(scores), 1)[:, :nbest]
            n_corr += sum(1 for i in range(len(objs)) if objs[i] in res[i])
        return float(n_corr/n_sample)

    def cal_path_hits(self, model, dataset, nbest):
        n_sample = len(dataset)
        n_corr = 0
        for subs, rels, objs in bucket_batch_iter(dataset, self.batchsize, rand_flg=False):
            # subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            scores = model.cal_path_scores(np.array(subs).astype(np.int32), np.array(rels).astype(np.int32))
            res = np.flip(np.argsort(scores), 1)[:, :nbest]
            n_corr += sum(1 for i in range(len(objs)) if objs[i] in res[i])
        return float(n_corr/n_sample)

    def get_best_info(self):
        if self.metric == 'mrr' or self.metric == 'hits':  # higher value is better
            best_val = max(self.ress)
        elif self.metric == 'mr':
            best_val = min(self.ress)
        else:
            raise ValueError('Invalid')
        best_epoch = self.ress.index(best_val) + 1
        return best_epoch, best_val
