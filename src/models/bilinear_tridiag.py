"""
"""


import copy
import sys
import chainer
from chainer import links as L, Variable, functions as F

sys.path.append('../')
from models.base_model import BaseModel
from chainer_math.matrix_operation import tridiag_matmul


class BilinearTridiag(BaseModel, chainer.Chain):
    def __init__(self, **kwargs):
        self.model_config = copy.copy(kwargs)
        self.n_entity = kwargs['n_entity']
        self.n_relation = kwargs['n_relation']
        self.dim = kwargs['dim']
        self.margin = kwargs['margin']
        super(BilinearTridiag, self).__init__(
            ent_embeds = L.EmbedID(self.n_entity, self.dim),
            rel_diag_embeds = L.EmbedID(self.n_relation, self.dim),
            rel_up_embeds = L.EmbedID(self.n_relation, self.dim-1),
            rel_low_embeds = L.EmbedID(self.n_relation, self.dim-1)
        )

    def __call__(self, pos_samples, neg_samples):
        pos_subs = Variable(pos_samples[0])
        pos_rels = Variable(pos_samples[1])
        pos_objs = Variable(pos_samples[2])
        neg_subs = Variable(neg_samples[0])
        neg_rels = Variable(neg_samples[1])
        neg_objs = Variable(neg_samples[2])

        pos_query = self._composite(pos_subs, pos_rels)
        neg_query = self._composite(neg_subs, neg_rels)

        pos_score = self._cal_similarity(pos_query, pos_objs)
        neg_score = self._cal_similarity(neg_query, neg_objs)

        return F.sum(F.relu(1 - (pos_score - neg_score)))

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
        _, n_rel = _pos_rels.shape

        for i in range(n_rel):
            cur_pos_rels = Veriable(_pos_rels[:, i])
            cur_neg_rels = Variable(_neg_rels[:, i])
            pos_query = self._traverse(pos_query, cur_pos_rels)
            neg_query = self._traverse(neg_query, cur_neg_rels)

        pos_score = self._cal_similarity(pos_query, pos_objs)
        neg_score = self._cal_similarity(neg_query, neg_objs)

        return F.sum(F.relu(1 - (pos_score - neg_score)))

    def cal_scores(self, subs, rels):
        _subs = Variable(subs)
        _rels = Variable(rels)
        query = self._composite(_subs, _rels).data
        return query.dot(self.ent_embeds.W.data.T)

    def _composite(self, subs, rels):
        ent_vs = self.ent_embeds(subs)
        rel_d_vs = self.rel_diag_embeds(rels)
        rel_u_vs = self.rel_up_embeds(rels)
        rel_l_vs = self.rel_low_embeds(rels)
        return tridiag_matmul(ent_vs, rel_d_vs, rel_u_vs, rel_l_vs)

    def _traverse(self, query, rels):
        rel_d_vs = self.rel_diag_embeds(rels)
        rel_u_vs = self.rel_up_embeds(rels)
        rel_l_vs = self.rel_low_embeds(rels)
        return tridiag_matmul(query, rel_d_vs, rel_u_vs, rel_l_vs)

    def _cal_similarity(self, query, objs):
        obj_vs = self.ent_embeds(objs)
        return F.reshape(F.batch_matmul(query, obj_vs, transa=True), (-1, ))  # inner
