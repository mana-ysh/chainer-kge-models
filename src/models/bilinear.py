
import numpy as np
import sys
import tensorflow as tf

sys.path.append('../')
from models.base_model import BaseModel


class Bilinear(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs['n_entity']
        self.n_relation = kwargs['n_relation']
        self.dim = kwargs['dim']
        if 'margin' in kwargs:
            self.margin = kwargs['margin']
            self._setup_pairwise()
        else:
            raise NotImplementedError
            self._setup_single()

        self._setup_scorer()
        self._setup_batch_scorer()
        self.model_config = kwargs

    def _setup_pairwise(self):
        self.pos_sub = tf.placeholder(tf.int32, [None])
        self.pos_rel = tf.placeholder(tf.int32, [None])
        self.pos_obj = tf.placeholder(tf.int32, [None])
        self.neg_sub = tf.placeholder(tf.int32, [None])
        self.neg_rel = tf.placeholder(tf.int32, [None])
        self.neg_obj = tf.placeholder(tf.int32, [None])

        with tf.name_scope('params'):
            initializer = tf.contrib.layers.xavier_initializer()
            self.ent_embeds = tf.get_variable(name='ent_embeds', shape=(self.n_entity, self.dim), initializer=initializer)
            self.rel_mats = tf.get_variable(name='rel_mats', shape=(self.n_relation, self.dim, self.dim), initializer=initializer)

            pos_query = self._composite(self.pos_sub, self.pos_rel)
            neg_query = self._composite(self.neg_sub, self.neg_rel)

            pos_score = self._cal_similarity(pos_query, self.pos_obj)
            neg_score = self._cal_similarity(neg_query, self.neg_obj)

        with tf.name_scope('output'):
            self.loss = tf.reduce_sum(tf.maximum(0., self.margin - (pos_score - neg_score)))

    def _composite(self, sub, rel):
        sub_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, sub), [-1, 1, self.dim])
        rel_mat = tf.nn.embedding_lookup(self.rel_mats, rel)
        return tf.matmul(sub_embed, rel_mat)  # shape = (batchsize, 1, dim)

    def _cal_similarity(self, query, obj):
        obj_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, obj), [-1, self.dim, 1])
        return tf.reshape(tf.matmul(query, obj_embed), [-1])

    def _cal_similarity_all(self, query):
        batchsize = tf.shape(query)[0]
        _query = tf.reshape(query, [-1, self.dim])
        return tf.matmul(_query, tf.transpose(self.ent_embeds))

    def _setup_scorer(self):
        self.sub = tf.placeholder(tf.int32, 1)
        self.rel = tf.placeholder(tf.int32, 1)
        with tf.name_scope('predict'):
            sub_embed = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.sub), [-1, 1, self.dim])
            rel_mat = tf.nn.embedding_lookup(self.rel_mats, self.rel)
            query = tf.reshape(tf.matmul(sub_embed, rel_mat), [-1, self.dim])
            self.scores = tf.reshape(tf.matmul(query, tf.transpose(self.ent_embeds)), [self.n_entity])

    def _setup_batch_scorer(self):
        self.batch_sub = tf.placeholder(tf.int32, [None])
        self.batch_rel = tf.placeholder(tf.int32, [None])
        with tf.name_scope('predict'):
            query = self._composite(self.batch_sub, self.batch_rel)
            self.batch_scores = self._cal_similarity_all(query)

    def _setup_single(self):
        raise NotImplementedError
