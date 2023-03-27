from unidistill.layers.ema import ModelEMA
from unidistill.engine.callbacks import MasterOnlyCallback


class EMACallback(MasterOnlyCallback):
    def before_train(self, trainer):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in trainer.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        trainer.ema_model = ModelEMA(trainer.model.cuda(), 0.9990)

        for bn_model, dist_group in zip(bn_model_list, bn_model_dist_group_list):
            bn_model.process_group = dist_group
        trainer.ema_model.updates = len(trainer.train_dataloader) * trainer.epoch

    def after_step(self, trainer, step, data_dict, *args, **kwargs):
        trainer.ema_model.update(trainer.model)
