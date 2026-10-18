# Modified by LIDAR from openpi (https://github.com/Physical-Intelligence/openpi, commit 215abfb),
# Apache License 2.0. Changes: image_keys / image_sizes (per-camera image slots) and two augmentation flags.
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Vega O3/O4: the image slots the model reads, in token order, and the square input side of each slot (multiples
    # of the 14-px patch). Defaults: the stock three slots, all at 224. A slot at another size keeps the pretrained
    # 16x16 SigLIP position table, resized to its grid (siglip.py); the model transforms resize per slot.
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    image_sizes: tuple[int, ...] | None = None
    # [vega V3.0] slots without "wrist" in the name that train with wrist-style augmentation (color jitter only, no
    # crop / rotate): the right-thumb camera in base_1_rgb (the thumb camera is treated like a wrist camera).
    wrist_aug_keys: tuple[str, ...] = ()
    # [V3.1a3 10-14] True: the "wrist" image slots get the base cameras' crop + rotate as well (default: colour only)
    wrist_geom_aug: bool = False

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        object.__setattr__(self, "image_keys", tuple(self.image_keys))
        object.__setattr__(self, "wrist_aug_keys", tuple(self.wrist_aug_keys))
        if self.image_sizes is not None:
            object.__setattr__(self, "image_sizes", tuple(int(s) for s in self.image_sizes))
            if len(self.image_sizes) != len(self.image_keys) or any(s % 14 for s in self.image_sizes):
                raise ValueError(f"image_sizes {self.image_sizes} must give one multiple of 14 per key {self.image_keys}")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @property
    def image_resolutions(self) -> dict[str, tuple[int, int]]:
        """Input resolution (height, width) per image slot (Vega O4)."""
        if self.image_sizes is None:
            return {key: tuple(_model.IMAGE_RESOLUTION) for key in self.image_keys}
        return {key: (size, size) for key, size in zip(self.image_keys, self.image_sizes, strict=True)}

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    key: jax.ShapeDtypeStruct([batch_size, *resolution, 3], jnp.float32)
                    for key, resolution in self.image_resolutions.items()
                },
                image_masks={key: image_mask_spec for key in self.image_keys},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
