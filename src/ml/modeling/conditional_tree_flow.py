import torch

from src.ml.modeling.normalizing_flow import NormalizingFlow


class ConditionalTreeFlow(NormalizingFlow):

    def encode(self, batch) -> dict:
        return {
            "z": batch["branch_lengths"],
            "y": batch["clades_one_hot"],
            "log_dj": torch.zeros(len(batch["branch_lengths"]), device=self.device),
        }
