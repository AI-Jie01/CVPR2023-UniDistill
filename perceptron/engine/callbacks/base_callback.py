__all__ = ["Callback", "MasterOnlyCallback"]


class Callback:

    # callback enabled rank list
    # None means callback is always enabled
    enabled_rank = None

    def setup(self, executor):
        pass

    def load_checkpoint(self, executor):
        pass

    def after_init(self, executor):
        pass

    def before_train(self, executor):
        pass

    def before_epoch(self, executor, epoch: int):
        pass

    def before_step(self, executor, step, data_dict):
        pass

    def before_backward(self, executor):
        pass

    def before_optimize(self, executor):
        pass

    def after_step(self, executor, step, data_dict, *args, **kwargs):
        pass

    def after_epoch(self, executor, epoch: int, update_best_ckpt: bool = False):
        pass

    def after_train(self, executor):
        pass

    def before_eval(self, *args, **kwargs):
        pass

    def after_eval(self, *args, **kwargs):
        pass


class MasterOnlyCallback(Callback):
    enabled_rank = [0]
