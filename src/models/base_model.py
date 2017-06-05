
import chainer
import os
import yaml


MODEL_CONFIG_FILE = 'model.config'


class BaseModel(object):
    def _setup_pairwise(self):
        raise NotImplementedError

    def _setup_single(self):
        raise NotImplementedError

    def _setup_scorer(self):
        raise NotImplementedError

    def _init_params(self):
        raise NotImplementedError

    def _composite(self):
        raise NotImplementedError

    def _cal_similarity(self):
        raise NotImplementedError

    def _cal_similarity_all(self):
        raise NotImplementedError

    def save_model(self, model_path):
        raise NotImplementedError

    def cal_path_scores(self, subs, rel_seqs):
        raise NotImplementedError

    def init_reverse(self):
        raise NotImplementedError

    def save_config(self, config_path):
        with open(config_path, 'w') as fw:
            fw.write(yaml.dump(self.model_config))

    @classmethod
    def instantiate_model(cls, model_path):
        """
        instantiation of models from config file for testing
        """
        model_dir = os.path.dirname(model_path)
        with open(os.path.join(model_dir, MODEL_CONFIG_FILE)) as f:
            model_config = yaml.load(f)
        return cls(**model_config)
