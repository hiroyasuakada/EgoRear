import torch
from pytorch_lightning.cli import LightningCLI

from pose_estimation.pl_wrappers.egoposeformer import PoseHeatmapLightningModel, PoseHeatmapMVFEXLightningModel, Pose3DMVFEXLightningModel


def setup_model(model):
    if model.compile:
        model.network = torch.compile(model.network, mode=model.compile_mode)

class TorchCompileCLI(LightningCLI):
    def fit(self, model, **kwargs):
        setup_model(model)
        self.trainer.fit(model, **kwargs)

    def test(self, model, **kwargs):
        setup_model(model)
        self.trainer.test(model, **kwargs)

    def predict(self, model, **kwargs):
        setup_model(model)
        self.trainer.predict(model, **kwargs)

if __name__ == "__main__":
    TorchCompileCLI()