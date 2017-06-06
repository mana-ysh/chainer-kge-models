
from chainer import serializers, cuda
import copy
import numpy as np
import os
import sys
import time

sys.path.append('../')
from utils.dataset import batch_iter, bucket_batch_iter

np.random.seed(46)


class PairwiseTrainer(object):
    def __init__(self, model, opt, **kwargs):
        self.model = model
        self.n_entity = model.n_entity
        self.opt = opt
        self.n_epoch = kwargs.pop('epoch')
        self.batchsize = kwargs.pop('batchsize')
        self.logger = kwargs.pop('logger')
        self.log_dir = kwargs.pop('model_dir')
        self.evaluator = kwargs.pop('evaluator')
        self.valid_dat = kwargs.pop('valid_dat')
        self.n_negative = kwargs.pop('n_negative')
        self.pretrain_model_path = kwargs.pop('restart')
        self.gpu_id = kwargs.pop('gpu_id')
        self.model_path = os.path.join(self.log_dir, self.model.__class__.__name__)

        if self.gpu_id > -1:
            self.xp = cuda.cupy
            cuda.get_device(self.gpu_id).use()
        else:
            self.xp = np

        """
        TODO: make other negative sampling methods available. and assuming only objects are corrupted
        """
        self.neg_sampler = UniformIntSampler(0, self.n_entity)

    def fit(self, samples):
        self.logger.info('setup trainer...')
        self.opt.setup(self.model)

        if self.gpu_id > -1:
            self.model.to_gpu()

        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            self.epoch = epoch
            for _subs, _rels, _objs in batch_iter(samples, self.batchsize):
                # setup for mini-batch training
                _batchsize = len(_subs)
                n_samples = _batchsize * self.n_negative
                # pos_triples = np.tile(batch, (self.n_negative, 1))
                neg_objs = self.neg_sampler.sample(n_samples).astype(self.xp.int32)  # assuming only objects are corrupted
                subs = self.xp.tile(_subs, (self.n_negative)).astype(self.xp.int32)
                rels = self.xp.tile(_rels, (self.n_negative)).astype(self.xp.int32)
                pos_objs = self.xp.tile(_objs, (self.n_negative)).astype(self.xp.int32)
                pos_samples = [subs, rels, pos_objs]
                neg_samples = [subs, rels, neg_objs]
                loss = self.model(pos_samples, neg_samples)
                self.model.zerograds()
                loss.backward()
                self.opt.update()
                sum_loss += loss.data
            self.validation()
            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

            # saving
            model_path = os.path.join(self.log_dir, 'model{}'.format(epoch+1))
            if self.gpu_id > -1:
                self.model.to_cpu()
                serializers.save_hdf5(model_path, self.model)
                self.model.to_gpu()
            else:
                serializers.save_hdf5(model_path, self.model)

        self._finalize()

    def validation(self):
        if self.valid_dat:  # run validation
            valid_start = time.time()
            res = self.evaluator.run(self.model, self.valid_dat)
            self.logger.info('evaluation metric in {} epoch: {}'.format(self.epoch+1, res))
            self.logger.info('evaluation time in {} epoch: {}'.format(self.epoch+1, time.time()-valid_start))
        else:
            pass

    def _finalize(self):
        if self.valid_dat:
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))


class PathPairwiseTrainer(PairwiseTrainer):
    def __init__(self, model, opt, **kwargs):
        self.model = model
        self.n_entity = model.n_entity
        self.opt = opt
        self.n_epoch = kwargs.pop('epoch')
        self.batchsize = kwargs.pop('batchsize')
        self.logger = kwargs.pop('logger')
        self.log_dir = kwargs.pop('model_dir')
        self.evaluator = kwargs.pop('evaluator')
        self.valid_dat = kwargs.pop('valid_dat')
        self.n_negative = kwargs.pop('n_negative')
        self.pretrain_model_path = kwargs.pop('restart')
        self.single_epoch = kwargs.pop('single')
        self.gpu_id = kwargs.pop('gpu_id')
        self.model_path = os.path.join(self.log_dir, self.model.__class__.__name__)

        if self.gpu_id > -1:
            self.xp = cuda.cupy
            cuda.get_device(self.gpu_id).use()
        else:
            self.xp = np

        """
        TODO: make other negative sampling methods available. and assuming only objects are corrupted
        """
        self.neg_sampler = UniformIntSampler(0, self.n_entity)

    def fit(self, single_samples, path_samples):
        self.logger.info('setup trainer...')
        self.opt.setup(self.model)

        # single training
        self.logger.info('start single training')
        for s_epoch in range(self.single_epoch):
            start = time.time()
            sum_loss = 0.
            self.epoch = s_epoch
            self.logger.info('start {} epoch'.format(s_epoch+1))
            for _subs, _rels, _objs in batch_iter(single_samples, self.batchsize):
                # setup for mini-batch training
                _batchsize = len(_subs)
                n_samples = _batchsize * self.n_negative
                neg_objs = self.xp.array(self.neg_sampler.sample(n_samples), dtype=self.xp.int32)  # assuming only objects are corrupted
                subs = self.xp.tile(_subs, (self.n_negative)).astype(self.xp.int32)
                rels = self.xp.tile(_rels, (self.n_negative)).astype(self.xp.int32)
                pos_objs = self.xp.tile(_objs, (self.n_negative)).astype(self.xp.int32)
                pos_samples = [subs, rels, pos_objs]
                neg_samples = [subs, rels, neg_objs]
                loss = self.model(pos_samples, neg_samples)
                self.model.zerograds()
                loss.backward()
                self.opt.update()
                sum_loss += loss.data

            self.model.init_reverse()  # initialize latent representations of reverse relations
            self.validation()
            self.logger.info('training loss in {} epoch: {}'.format(s_epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(s_epoch+1, time.time()-start))

            # saving
            model_path = os.path.join(self.log_dir, 'model{}.single'.format(s_epoch+1))
            if self.gpu_id > -1:
                self.model.to_cpu()
                serializers.save_hdf5(model_path, self.model)
                self.model.to_gpu()
            else:
                serializers.save_hdf5(model_path, self.model)

        # path training
        self.logger.info('start path training')
        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            self.epoch = epoch
            for (_subs, _rels, _pos_objs) in bucket_batch_iter(path_samples, self.batchsize):
                # setup for mini-batch training
                _batchsize = len(_subs)
                n_samples = _batchsize * self.n_negative
                neg_objs = self.neg_sampler.sample(n_samples).astype(self.xp.int32)  # assuming only objects are corrupted
                subs = self.xp.tile(_subs, (self.n_negative)).astype(self.xp.int32)
                rels = self.xp.tile(_rels, (self.n_negative, 1)).astype(self.xp.int32)
                pos_objs = self.xp.tile(_pos_objs, (self.n_negative)).astype(self.xp.int32)
                pos_samples = [subs, rels, pos_objs]
                neg_samples = [subs, rels, neg_objs]
                loss = self.model(pos_samples, neg_samples)
                self.model.zerograds()
                loss.backward()
                self.opt.update()
                sum_loss += loss.data
            self.validation()
            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

            # saving
            model_path = os.path.join(self.log_dir, 'model{}.path'.format(epoch+1))
            if self.gpu_id > -1:
                self.model.to_cpu()
                serializers.save_hdf5(model_path, self.model)
                self.model.to_gpu()
            else:
                serializers.save_hdf5(model_path, self.model)

    def _finalize(self):
        if self.valid_dat:
            best_epoch, best_val = self.evaluator.get_best_info()
            self.logger.info('===== Best metric: {} ({} epoch) ====='.format(best_val, best_epoch))


class UniformIntSampler(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def sample(self, size):
        return np.random.randint(self.lower, self.upper, size=size)


class DebugPairwiseTrainer(PairwiseTrainer):
    def __init__(self, **kwargs):
        super(DebugPairwiseTrainer, self).__init__(**kwargs)
        self.n_negative = 1

    def fit(self, samples):
        self.logger.info('setup trainer...')
        self.opt.setup(self.model)
        for epoch in range(self.n_epoch):
            start = time.time()
            sum_loss = 0.
            self.logger.info('start {} epoch'.format(epoch+1))
            for batch in batch_iter(samples, 1):
                # setup for mini-batch training
                _batchsize = len(batch)
                n_samples = _batchsize * self.n_negative
                pos_triples = np.tile(batch, (self.n_negative, 1))
                neg_objs = self.neg_sampler.sample(n_samples).astype(np.int32)  # assuming only objects are corrupted
                subs = pos_triples[:, 0].astype(np.int32)
                rels = pos_triples[:, 1].astype(np.int32)
                pos_objs = pos_triples[:, 2].astype(np.int32)
                pos_samples = [subs, rels, pos_objs]
                neg_samples = [subs, rels, neg_objs]
                print('sub: {}'.format(subs))
                print('rel: {}'.format(rels))
                print('pos_obj: {}'.format(pos_objs))
                print('neg_obj: {}'.format(neg_objs))

                print('===== before =====')
                sub_embed = copy.copy(self.model.ent_embeds.W.data[subs])
                rel_d_embed = copy.copy(self.model.rel_diag_embeds.W.data[rels])
                rel_u_embed = copy.copy(self.model.rel_up_embeds.W.data[rels])
                rel_l_embed = copy.copy(self.model.rel_low_embeds.W.data[rels])
                pos_obj_embed = copy.copy(self.model.ent_embeds.W.data[pos_objs])
                neg_obj_embed = copy.copy(self.model.ent_embeds.W.data[neg_objs])
                print('sub_embed: {}'.format(self.model.ent_embeds.W.data[subs]))
                print('rel_d_embed: {}'.format(self.model.rel_diag_embeds.W.data[rels]))
                print('rel_u_embed: {}'.format(self.model.rel_up_embeds.W.data[rels]))
                print('rel_l_embed: {}'.format(self.model.rel_low_embeds.W.data[rels]))
                print('pos_obj_embed: {}'.format(self.model.ent_embeds.W.data[pos_objs]))
                print('neg_obj_embed: {}'.format(self.model.ent_embeds.W.data[neg_objs]))

                loss = self.model(pos_samples, neg_samples)
                self.model.zerograds()
                loss.backward()
                self.opt.update()

                print('===== after =====')
                print('sub_embed: {}'.format(self.model.ent_embeds.W.data[subs]))
                print('rel_d_embed: {}'.format(self.model.rel_diag_embeds.W.data[rels]))
                print('rel_u_embed: {}'.format(self.model.rel_up_embeds.W.data[rels]))
                print('rel_l_embed: {}'.format(self.model.rel_low_embeds.W.data[rels]))
                print('pos_obj_embed: {}'.format(self.model.ent_embeds.W.data[pos_objs]))
                print('neg_obj_embed: {}'.format(self.model.ent_embeds.W.data[neg_objs]))

                print('===== gradients by AD =====')
                print('sub_embed: {}'.format(self.model.ent_embeds.W.grad[subs]))
                print('rel_d_embed: {}'.format(self.model.rel_diag_embeds.W.grad[rels]))
                print('rel_u_embed: {}'.format(self.model.rel_up_embeds.W.grad[rels]))
                print('rel_l_embed: {}'.format(self.model.rel_low_embeds.W.grad[rels]))
                print('pos_obj_embed: {}'.format(self.model.ent_embeds.W.grad[pos_objs]))
                print('neg_obj_embed: {}'.format(self.model.ent_embeds.W.grad[neg_objs]))

                print('===== true gradients =====')
                pad = np.zeros((1, 1))
                grad_s = rel_d_embed * pos_obj_embed + np.concatenate([rel_u_embed * pos_obj_embed[:, 1:], pad], axis=1) + np.concatenate([pad, rel_l_embed * pos_obj_embed[:, :-1]], axis=1) - (rel_d_embed * neg_obj_embed + np.concatenate([rel_u_embed * neg_obj_embed[:, 1:], pad], axis=1) + np.concatenate([pad, rel_l_embed * neg_obj_embed[:, :-1]], axis=1))
                print('sub_embed: {}'.format(grad_s))
                grad_rd = sub_embed * pos_obj_embed - sub_embed * neg_obj_embed
                print('rel_d_embed: {}'.format(grad_rd))
                grad_ru = sub_embed[:, :-1] * (pos_obj_embed[:, 1:] - neg_obj_embed[:, 1:])
                print('rel_u_embed: {}'.format(grad_ru))
                grad_rl = sub_embed[:, 1:] * (pos_obj_embed[:, :-1] - neg_obj_embed[:, :-1])
                print('rel_l_embed: {}'.format(grad_rl))
                grad_po = rel_d_embed * sub_embed + np.concatenate([pad, rel_u_embed * sub_embed[:, :-1]], axis=1) + np.concatenate([rel_l_embed * sub_embed[:, 1:], pad], axis=1)
                print('pos_obj_embed: {}'.format(grad_po))
                # print('neg_obj_embed: {}'.format(self.model.ent_embeds.W.grad[neg_objs]))
                sum_loss += loss
                raise
            self.validation()
            self.logger.info('training loss in {} epoch: {}'.format(epoch+1, sum_loss))
            self.logger.info('training time in {} epoch: {}'.format(epoch+1, time.time()-start))

            # saving
            model_path = os.path.join(self.log_dir, 'model{}'.format(epoch+1))
            serializers.save_hdf5(model_path, self.model)
