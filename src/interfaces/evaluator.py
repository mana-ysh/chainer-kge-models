
from chainer import cuda
import numpy as np
import sys

sys.path.append('../')
from utils.dataset import batch_iter, PathQueryDataset, TripletDataset, bucket_batch_iter


BATCHSIZE = 100
NOGPU = -1

class Evaluator(object):
    def __init__(self, metric, nbest=None, gpu_id=-1):
        assert metric in ['mr', 'mrr', 'hits'], 'Invalid metric: {}'.format(metric)
        if metric == 'hits':
            assert nbest, 'Please indicate n-best in using hits'
        self.metric = metric
        self.nbest = nbest
        self.batchsize = BATCHSIZE
        self.ress = []
        self.gpu_id = NOGPU  # CAUTION: force not to use GPU for evaluation
        if self.gpu_id > -1:
            self.xp = cuda.cupy
        else:
            self.xp = np

    def run(self, model, dataset):
        if isinstance(dataset, TripletDataset):
            self.dat_iter = batch_iter
        elif isinstance(dataset, PathQueryDataset):
            self.dat_iter = bucket_batch_iter
        else:
            raise ValueError('Invalid dataset type: {}'.format(dataset))

        if self.metric == 'mr':
            res = self.cal_mr(model, dataset)
        elif self.metric == 'mrr':
            res = self.cal_mrr(model, dataset)
            # res = self.cal_batch_mrr(model, dataset)
        elif self.metric == 'hits':
            res = self.cal_hits(model, dataset, self.nbest)
        else:
            raise ValueError

        self.ress.append(res)
        return res

    def cal_mr(self, model, dataset):
        n_sample = len(dataset)
        sum_r = 0.
        for subs, rels, objs in self.dat_iter(dataset, self.batchsize, rand_flg=False):
            scores = model.cal_scores(self.xp.array(subs, dtype=self.xp.int32), self.xp.array(rels, dtype=self.xp.int32))
            if self.gpu_id > -1:
                scores = cuda.to_cpu(scores)
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_r += sum(rank for rank in ranks)
        return float(sum_r/n_sample)

    def cal_mrr(self, model, dataset):
        n_sample = len(dataset)
        sum_rr = 0.
        for subs, rels, objs in self.dat_iter(dataset, self.batchsize, rand_flg=False):
            scores = model.cal_scores(self.xp.array(subs, dtype=self.xp.int32), self.xp.array(rels, dtype=self.xp.int32))
            if self.gpu_id > -1:
                scores = cuda.to_cpu(scores)
            res = np.flip(np.argsort(scores), 1)
            ranks = [np.where(order == obj)[0][0] + 1 for order, obj in zip(res, objs)]  # TODO: maybe inefficient
            sum_rr += sum(float(1/rank) for rank in ranks)
        return float(sum_rr/n_sample)

    def cal_hits(self, model, dataset, nbest):
        n_sample = len(dataset)
        n_corr = 0
        for subs, rels, objs in self.dat_iter(dataset, self.batchsize, rand_flg=False):
            scores = model.cal_scores(self.xp.array(subs, dtype=self.xp.int32), self.xp.array(rels, dtype=self.xp.int32))
            if self.gpu_id > -1:
                scores = cuda.to_cpu(scores)
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
