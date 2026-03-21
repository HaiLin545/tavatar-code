import lightning as L
from model.deformer import HumanDeformer
from gaussian import render
from utils.eval import Evaluator

class TavatarModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.deformer = HumanDeformer(cfg)
        self.evaluator = Evaluator()
        self.pred_step_outputs = []

    def predict_step(self, batch, batch_idx):
        smpl_params = batch["smpl_params"]
        camera_params = batch["camera_params"]
        camera_param = {k: v[0] for k, v in camera_params.items()}

        gaussians = self.deformer(smpl_params)
        rendered = render(self.cfg, gaussians, camera_param)

        output = (gaussians, rendered)
        return output
