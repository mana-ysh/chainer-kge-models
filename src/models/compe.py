
import chainer
from chainer import links as L, Variable, functions as F
import numpy as np
import sys

sys.path.append('../')
from models.base_model import BaseModel


class ComplexE(BaseModel):
    def __init__(self, **kwargs):
        self.model_config = copy.copy(kwargs)
        self.n_entity = kwargs['n_entity']
        self.n_relation = kwargs['n_relation']
        self.dim = kwargs['dim']
        super(ComplexE, self).__init__(
            ent_re_embeds = L.EmbedID(self.n_entity, self.dim),
            ent_im_embeds = L.EmbedID(self.n_entity, self.dim),
            rel_re_embeds = L.EmbedID(self.n_relation, self.dim),
            rel_im_embeds = L.EmbedID(self.n_relation, self.dim)
        )

    def __call__(self, pos_samples, neg_samples):
        pos_subs = Variable(pos_samples[0])
        pos_rels = Variable(pos_samples[1])
        pos_objs = Variable(pos_samples[2])
        neg_subs = Variable(neg_samples[0])
        neg_rels = Variable(neg_samples[1])
        neg_objs = Variable(neg_samples[2])

        pos_re_query, pos_im_query = self._composite(pos_subs, pos_rels)
        neg_re_query, neg_im_query = self._composite(neg_subs, neg_rels)

        pos_score = self._cal_similarity(pos_re_query, pos_im_query, pos_objs)
        neg_score = self._cal_similarity(neg_re_query, neg_im_query, neg_objs)

        return F.sum(F.relu(1. - (pos_score - neg_score)))

    def _path_forward(self, pos_samples, neg_samples):
        pos_subs = Variable(pos_samples[0])
        pos_objs = Variable(pos_samples[2])
        neg_subs = Variable(neg_samples[0])
        neg_objs = Variable(neg_samples[2])
        _pos_rels = pos_samples[1]
        _neg_rels = neg_samples[1]
        assert _pos_rels.shape == _neg_rels.shape  # batchsize x n_rel

        pos_re_query = self.ent_re_embeds(pos_subs)
        pos_im_query = self.ent_im_embeds(pos_subs)
        neg_re_query = self.ent_re_embeds(neg_subs)
        neg_im_query = self.ent_im_embeds(neg_subs)
        _, n_rel = _pos_rels.shape

        for i in range(n_rel):
            cur_pos_rels = Veriable(_pos_rels[:, i])
            cur_neg_rels = Variable(_neg_rels[:, i])
            pos_re_query, pos_im_query = self._traverse(pos_re_query, pos_im_query, cur_pos_rels)
            neg_re_query, neg_im_query = self._traverse(neg_re_query, neg_im_query, cur_neg_rels)

        pos_score = self._cal_similarity(pos_re_query, pos_im_query, pos_objs)
        neg_score = self._cal_similarity(neg_re_query, neg_im_query, neg_objs)

        return F.sum(F.relu(1 - (pos_score - neg_score)))

    def cal_scores(self, subs, rels):
        _subs = Variable(subs)
        _rels = Variable(rels)
        re_query, im_query = self._composite(_subs, _rels)
        re_query = re_query.data
        im_query = im_query.data
        return re_query.dot(self.ent_re_embeds.W.data.T) - im_query.dot(self.ent_im_embeds.W.data.T)

    def _composite(self, subs, rels):
        sub_re_embed = self.ent_re_embeds(subs)
        sub_im_embed = self.ent_im_embeds(subs)
        rel_re_embed = self.rel_re_embeds(rels)
        rel_im_embed = self.rel_im_embeds(rels)
        re_query = sub_re_embed * rel_re_embed - sub_im_embed * rel_im_embed
        im_query = - sub_re_embed * rel_im_embed - sub_im_embed * rel_re_embed  # complex conjugate
        return re_query, im_query

    def _traverse(self, re_query, im_query, rels):
        rel_re_emb = self.rel_re_embeds(rels)
        rel_im_emb = self.rel_im_embeds(rels)
        new_re_query = re_query * rel_re_emb - im_query * rel_im_emb
        new_im_query = - re_query * rel_im_emb - im_query * rel_re_emb
        return new_re_query, new_im_query

    def _cal_similarity(self, re_query, im_query, objs):
        obj_re_embed = self.ent_re_embeds(objs)
        obj_im_embed = self.ent_im_embeds(objs)
        _inner = F.batch_matmul(re_query, obj_re_embed, transa=True) - F.batch_matmul(im_query, obj_im_embed, transa=True)  # No complex conjugate in obj
        return F.reshape(_inner, (-1))
