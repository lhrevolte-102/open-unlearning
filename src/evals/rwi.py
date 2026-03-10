from evals.base import Evaluator


class RWIEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RWI", eval_cfg, **kwargs)
