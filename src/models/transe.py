"""
TODO
- normalize updated entity vectors in each batch
"""

import copy
import sys
import chainer
from chainer import links as L, Variable, functions as F
import numpy as np
from sklearn.preprocessing import normalize

sys.path.append('../')
from models.base_model import BaseModel


class TransE(BaseModel, chainer.Chain):
    def __init__(self, **kwargs):
        self.model_config = copy.copy(kwargs)
        self.n_entity = kwargs['n_entity']
        self.n_relation = kwargs['n_relation']
        self.dim = kwargs['dim']
        self.unit = kwargs['unit']
        self.dist = kwargs.pop('dist', 2)
        assert self.dist in [1, 2]  # L1 or L2
        super(TransE, self).__init__(
            ent_embeds = L.EmbedID(self.n_entity, self.dim),
            rel_embeds = L.EmbedID(self.n_relation, self.dim)
        )


    def _single_forward(self, pos_samples, neg_samples):
        pos_subs = Variable(pos_samples[0])
        pos_rels = Variable(pos_samples[1])
        pos_objs = Variable(pos_samples[2])
        neg_subs = Variable(neg_samples[0])
        neg_rels = Variable(neg_samples[1])
        neg_objs = Variable(neg_samples[2])

        if self.unit:
            ent_list = pos_samples[0].tolist() + pos_samples[2].tolist() + neg_samples[0].tolist() + neg_samples[2].tolist()
            self._normalize(set(ent_list))

        pos_query = self._traverse(self.ent_embeds(pos_subs), pos_rels)
        neg_query = self._traverse(self.ent_embeds(neg_subs), neg_rels)

        pos_score = self._cal_similarity(pos_query, pos_objs)
        neg_score = self._cal_similarity(neg_query, neg_objs)

        return F.sum(F.relu(1. - (pos_score - neg_score)))

    def init_reverse(self):
        assert self.n_relation % 2 == 0
        n_re_rel = self.n_relation // 2
        re_rel_embs = -1 * self.rel_embeds.W.data[:n_re_rel]
        self.rel_embeds.W.data[n_re_rel:] = re_rel_embs

    def _path_forward(self, pos_samples, neg_samples):
        pos_subs = Variable(pos_samples[0])
        pos_objs = Variable(pos_samples[2])
        neg_subs = Variable(neg_samples[0])
        neg_objs = Variable(neg_samples[2])
        _pos_rels = pos_samples[1]
        _neg_rels = neg_samples[1]
        assert _pos_rels.shape == _neg_rels.shape  # batchsize x n_rel

        pos_query = self.ent_embeds(pos_subs)
        neg_query = self.ent_embeds(neg_subs)
        _batchsize, n_rel = _pos_rels.shape

        for i in range(n_rel):
            cur_pos_rels = Variable(_pos_rels[:, i])
            cur_neg_rels = Variable(_neg_rels[:, i])
            pos_query = self._traverse(pos_query, cur_pos_rels)
            neg_query = self._traverse(neg_query, cur_neg_rels)

        pos_score = self._cal_similarity(pos_query, pos_objs)
        neg_score = self._cal_similarity(neg_query, neg_objs)

        return F.sum(F.relu(1 - (pos_score - neg_score)))

    def _path_scores(self, subs, rels):
        query = self.ent_embeds(Variable(subs))
        _batchsize, n_rel = rels.shape

        for i in range(n_rel):
            cur_rs = Variable(rels[:, i])
            query = self._traverse(query, cur_rs)

        query = F.tile(F.expand_dims(query, axis=1), (1, self.n_entity, 1))
        _ent_embeds = F.tile(F.expand_dims(self.ent_embeds.W, axis=0), (_batchsize, 1, 1))
        return - F.sum((query - _ent_embeds)**self.dist, axis=2).data

    def _traverse(self, query, rels):
        rel_embs = self.rel_embeds(rels)
        return query + rel_embs

    def _cal_similarity(self, query, objs):
        obj_embed = self.ent_embeds(objs)
        return - F.sum((query - obj_embed)**self.dist, axis=1)

    def _single_scores(self, subs, rels):
        _batchsize = len(subs)
        _subs = Variable(subs)
        _rels = Variable(rels)

        if self.unit:
            self._normalize(set(subs.tolist()))

        query = F.tile(F.expand_dims(self._traverse(self.ent_embeds(_subs), _rels), 1), (1, self.n_entity, 1))
        _ent_embeds = F.tile(F.expand_dims(self.ent_embeds.W, axis=0), (_batchsize, 1, 1))
        return - F.sum((query - _ent_embeds)**self.dist, axis=2).data

    def _normalize(self, ent_set):
        ents = list(ent_set)
        self.ent_embeds.W.data[ents] = normalize(self.ent_embeds.W.data[ents])
