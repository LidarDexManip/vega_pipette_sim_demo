"""The openpi configuration of the released Vega-U + Wuji pipette policy (pi0.5, inference only).

The policy reads four cameras (head, right thumb, left wrist, right wrist) at 224 x 224 and the 54-D joint state, and
predicts 32 future steps of the 54-D joint targets.  Inside the model both are 46-D: the 8 hand joints that the Wuji
hand couples to their neighbours are dropped and restored as 0 on the way out.  The arms are predicted relative to the
current state and the hands as absolute positions.  The prompt is the task sentence the live runner sends.

`make_train_config()` returns the openpi `TrainConfig` that `openpi.policies.policy_config.create_trained_policy`
expects.  Only the parts used for inference are set; the training recipe is not part of this release.
"""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.vega_policy as vega_policy
import openpi.training.config as _config
import openpi.transforms as _transforms

CONFIG_NAME = "pi05_vega_wuji_pipette"
# Directory under <checkpoint>/assets/ that holds norm_stats.json.
ASSET_ID = "vega_wuji_pipette"
# The four cameras the model reads, by model image slot: head (ego), right-thumb, left wrist, right wrist; all at 224.
CAMERAS = vega_policy.CAMERAS_B2
# Arms relative to the current state, hands absolute (46-D order: left arm, left hand, right arm, right hand).
DELTA_MASK = vega_policy.DELTA_MASK
# The task sentence the policy was trained with for the first half of the task (S1-S5).
TASK_PROMPT = vega_policy.VEGA_TASK_PROMPT


@dataclasses.dataclass(frozen=True)
class VegaInferenceDataConfig(_config.DataConfigFactory):
    """Policy input / output transforms of the Vega pipette policy (no dataset-side transforms)."""

    cameras: vega_policy.VegaCameras = CAMERAS
    delta_mask: tuple[bool, ...] = DELTA_MASK
    # The 8 coupled hand joints are dropped even when the state holds a nonzero value there.
    allow_nonzero_drop: bool = True

    def create(self, assets_dirs: pathlib.Path, model_config: pi0_config.Pi0Config) -> _config.DataConfig:
        if model_config.image_keys != self.cameras.image_keys() or model_config.image_sizes != self.cameras.sizes():
            raise ValueError(f"model image slots {model_config.image_keys} / {model_config.image_sizes} do not match "
                             f"the cameras {self.cameras.image_keys()} / {self.cameras.sizes()}")
        data_transforms = _transforms.Group(
            inputs=[vega_policy.VegaInputs(self.cameras.image_keys(), self.allow_nonzero_drop)],
            outputs=[vega_policy.VegaOutputs()],
        ).push(
            inputs=[_transforms.DeltaActions(self.delta_mask)],
            outputs=[_transforms.AbsoluteActions(self.delta_mask)],
        )
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImagesPerKey(tuple(model_config.image_resolutions.items())),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def make_model_config() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=vega_policy.DIM_46,
        action_horizon=32,
        max_token_len=256,
        discrete_state_input=True,
        image_keys=CAMERAS.image_keys(),
        image_sizes=CAMERAS.sizes(),
        # training-time augmentation flag of this checkpoint; it has no effect at inference
        wrist_geom_aug=True,
    )


def make_train_config() -> _config.TrainConfig:
    return _config.TrainConfig(
        name=CONFIG_NAME,
        model=make_model_config(),
        data=VegaInferenceDataConfig(repo_id=ASSET_ID),
    )
