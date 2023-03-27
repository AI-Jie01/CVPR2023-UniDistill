import os
import pickle

from .base_callback import MasterOnlyCallback

__all__ = ["EvalResultsSaver"]


class EvalResultsSaver(MasterOnlyCallback):
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def after_eval(self, executor, det_annos: list):
        out_file = os.path.join(self.out_dir, "result.pkl")
        pickle.dump(det_annos, open(out_file, "wb"))
