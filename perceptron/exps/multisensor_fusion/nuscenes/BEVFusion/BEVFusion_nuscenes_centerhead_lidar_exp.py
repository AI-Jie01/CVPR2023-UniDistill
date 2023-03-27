from perceptron.engine.cli.det3d_cli import Det3DCli

from perceptron.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_centerhead_fusion_exp import (
    Exp as BaseExp,
)


class Exp(BaseExp):
    def __init__(self, batch_size_per_device=4, total_devices=1, max_epoch=20, **kwargs):
        super(Exp, self).__init__(batch_size_per_device, total_devices, max_epoch)
        self.data_cfg["img_key_list"] = []
        self.model_cfg["camera_encoder"] = None

    def _change_cfg_params(self):
        self.data_cfg["aug_cfg"]["gt_sampling_cfg"] = None


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    Det3DCli(Exp).run()
