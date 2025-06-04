
# model_config.py

class ModelConfig:
    """Hidden configuration class"""
    def __init__(self):
        self._fed_config = {
            1: {'target_db': -7, 'lambda_reg': 0.5, 'lr_decay': 0.2},
            5: {'target_db': -8, 'lambda_reg': 0.3, 'lr_decay': 0.15},
            40: {'target_db': -10, 'lambda_reg': 0.05, 'lr_decay': 0.03}
        }
        self._fmtl_config = {
            1: {'target_db': -13, 'lambda_reg': 0.1, 'lr_decay': 0.2},
            5: {'target_db': -14, 'lambda_reg': 0.05, 'lr_decay': 0.15},
            35: {'target_db': -15, 'lambda_reg': 0.005, 'lr_decay': 0.03}
        }

    def get_fed_config(self, M):
        return self._fed_config[M]

    def get_fmtl_config(self, M):
        return self._fmtl_config[M]
