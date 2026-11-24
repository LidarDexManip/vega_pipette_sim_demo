# Policy server

`serve.py` runs one of the released pi0.5 policies, inference only, behind the ZeroMQ protocol the live runners speak
(`vega_pipette/control/policy_client.py`): the pipette policy (`--task pipette`, the default) or the tube pick-and-place
policy (`--task pick-place`). One server answers one simulator. It can run on the same GPU as Isaac Sim: one 24 GB GPU
holds both.

## Setup

1. Build the environment as in the
   [Installation section of the main README](https://github.com/LidarDexManip/vega_pipette_sim_demo/tree/main#installation)
   (openpi at a pinned commit with `openpi_overlay/` copied over it), and set `VEGA_POLICY_PY` in `config/vega.env` to
   its Python.
2. Download the checkpoints (about 12 GB each):
   ```bash
   ~/envs/openpi/.venv/bin/huggingface-cli download birbirll/vega-wuji-pipette-pi05 \
     --local-dir checkpoints/vega-wuji-pipette-pi05
   ~/envs/openpi/.venv/bin/huggingface-cli download birbirll/vega-tube-pnp --local-dir checkpoints/vega-tube-pnp
   ```

## Run

```bash
$VEGA_POLICY_PY policy_server/serve.py --port 5555
$VEGA_POLICY_PY policy_server/serve.py --task pick-place --port 5555
```

- `--checkpoint DIR`: another checkpoint location. The default is `$VEGA_CHECKPOINT`, else `checkpoints/vega-wuji-pipette-pi05`;
  with `--task pick-place`, `$VEGA_PICK_PLACE_CHECKPOINT`, else `checkpoints/vega-tube-pnp`.
- `--host`: the interface to listen on. The default is localhost only.
- `--ready-file FILE`: written once the server answers.

At start-up the server loads the model and runs one warm-up inference, and only then opens the port. The first start compiles the model, which takes a few minutes. After that the compiled program is loaded from `$VEGA_RUNS/jax_cache`. The PaliGemma tokenizer is downloaded once into `$VEGA_RUNS/openpi_cache`; it is public and needs no account.

`scripts/run_sim_gui.sh` and `scripts/run_sim_headless.sh` start and stop the server for you. The protocol (endpoints and message fields) is documented at the top of `serve.py`.

## How the model is set up

The policy is built with openpi (github.com/Physical-Intelligence/openpi) at commit `215abfb`, with the files in `openpi_overlay/` copied over it:

| File | Change |
|---|---|
| `src/openpi/models/model.py`, `pi0.py`, `pi0_config.py` | The model's camera image slots and per-slot input sizes are set by the config. |
| `src/openpi/models/siglip.py` | The vision encoder's position table can be resized for another input grid. |
| `src/openpi/transforms.py` | Per-slot image resize; an optional prompt format without the state. |
| `src/openpi/policies/vega_policy.py` | New: the Vega-U + Wuji input/output transforms (54-D joints <-> the model's 46-D or 50-D, camera slots; the tube task's "Rel" text). |
| `src/openpi/policies/vega_fk.py` | New: the position of the left hand from the arm joints, read from `assets/robot/vega_u_wuji_hand2_kinematics.urdf`. |
| `pyproject.toml`, `uv.lock` | JAX 0.6.2 and CUDA 12.9 runtime libraries, the versions the released checkpoint was evaluated with. |

`vega_config.py` holds the openpi configurations of the two policies:

| | Pipette | Tube pick-and-place |
|---|---|---|
| **Cameras** | head, right thumb, left wrist and right wrist, each at 224 x 224 | head (224 x 126, padded to 224 x 224) and the palm-side left wrist camera (224 x 224) |
| **State and actions** | 46-D | 50-D (the left hand's four finger-spread joints kept) |
| **Prompt** | the task sentence | the current instruction (pick, place, release); the server appends "Rel: dx dy" to a place instruction that ends in "Pocket: bx by" |

Both predict 32-step action chunks, normalize the state and actions by quantiles, and predict the arm actions relative
to the current state and the hand actions as absolute positions.

## License

- **Code:** Apache License 2.0. The files in `openpi_overlay/` are modified openpi files (Apache 2.0) and say so at the top.
- **Model weights:** a Model Derivative of Gemma, subject to the Gemma Terms of Use. See the model card and NOTICE.
