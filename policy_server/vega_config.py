"""The openpi configurations of the released Vega-U + Wuji policies (pi0.5, inference only), one per task.

pipette     four cameras (head, right thumb, left wrist, right wrist) at 224 x 224; the 54-D joint state; 32 future
            steps of the 54-D joint targets. Inside the model both are 46-D: the 8 hand joints that the Wuji hand
            couples to their neighbours are dropped and restored as 0 on the way out. Prompt: the task sentence.
pick-place  two cameras (head at 224 x 126, padded to 224 by the resize; the palm-side left wrist camera at 224 x 224);
            the same state and actions on a 50-D layout (the left hand's four coupled joints kept); prompt: the
            instruction of the current stage (pick, place, release). The place instruction ends in "Pocket: bx by"
            (the hand target, from the runner), and the server appends "Rel: dx dy" (vega_policy.PocketRelativePP).
Both predict the arms relative to the current state and the hands as absolute positions.

`make_train_config(task)` returns the openpi `TrainConfig` that `openpi.policies.policy_config.create_trained_policy`
expects. Only the parts used for inference are set; the training recipe is not part of this release.
"""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.vega_policy as vega_policy
import openpi.training.config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    """What differs between the two policies."""

    config_name: str
    asset_id: str                         # the directory under <checkpoint>/assets/ that holds norm_stats.json
    cameras: vega_policy.VegaCameras      # the cameras the model reads, by model image slot
    keep_dims: tuple[int, ...]            # the 54-D dims of the model's state and action layout
    delta_mask: tuple[bool, ...]          # on that layout: True = relative to the current state
    max_token_len: int
    warmup_prompt: str
    wrist_geom_aug: bool                  # the checkpoint's training-time augmentation flag (no effect at inference)
    place_hint: bool                      # append "Rel: dx dy" to a prompt with "Pocket: bx by"


POLICIES = {
    "pipette": PolicySpec(
        config_name="pi05_vega_wuji_pipette", asset_id="vega_wuji_pipette", cameras=vega_policy.CAMERAS_B2,
        keep_dims=vega_policy.KEEP_DIMS, delta_mask=vega_policy.DELTA_MASK, max_token_len=256,
        warmup_prompt=vega_policy.VEGA_TASK_PROMPT, wrist_geom_aug=True, place_hint=False),
    "pick-place": PolicySpec(
        config_name="pi05_vega_tube_pnp", asset_id="vega_tube_pnp",
        cameras=vega_policy.CAMERAS_PP, keep_dims=vega_policy.KEEP_DIMS_50, delta_mask=vega_policy.DELTA_MASK_50,
        max_token_len=288, warmup_prompt="Place the tube into column 3, row 2 of the blue rack. Pocket: 198 128",
        wrist_geom_aug=False, place_hint=True),
}


@dataclasses.dataclass(frozen=True)
class VegaInferenceDataConfig(_config.DataConfigFactory):
    """Policy input / output transforms of one Vega policy (no dataset-side transforms)."""

    spec: PolicySpec = POLICIES["pipette"]

    def create(self, assets_dirs: pathlib.Path, model_config: pi0_config.Pi0Config) -> _config.DataConfig:
        cameras = self.spec.cameras
        if model_config.image_keys != cameras.image_keys() or model_config.image_sizes != cameras.sizes():
            raise ValueError(f"model image slots {model_config.image_keys} / {model_config.image_sizes} do not match "
                             f"the cameras {cameras.image_keys()} / {cameras.sizes()}")
        data_transforms = _transforms.Group(
            # allow_nonzero_drop only concerns training labels: at inference there are no actions to check
            inputs=[*([vega_policy.PocketRelativePP()] if self.spec.place_hint else []),
                    vega_policy.VegaInputs(cameras.image_keys(), True, self.spec.keep_dims)],
            outputs=[vega_policy.VegaOutputs(self.spec.keep_dims)],
        ).push(
            inputs=[_transforms.DeltaActions(self.spec.delta_mask)],
            outputs=[_transforms.AbsoluteActions(self.spec.delta_mask)],
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


def make_model_config(spec: PolicySpec) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=len(spec.keep_dims),
        action_horizon=32,
        max_token_len=spec.max_token_len,
        discrete_state_input=True,
        image_keys=spec.cameras.image_keys(),
        image_sizes=spec.cameras.sizes(),
        wrist_geom_aug=spec.wrist_geom_aug,
    )


def make_train_config(task: str) -> _config.TrainConfig:
    spec = POLICIES[task]
    return _config.TrainConfig(
        name=spec.config_name,
        model=make_model_config(spec),
        data=VegaInferenceDataConfig(repo_id=spec.asset_id, spec=spec),
    )
