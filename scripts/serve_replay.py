"""Tape server: replays hf_v0 training-episode command streams over the
standard websocket policy protocol.

Purpose: EMPTY-TABLE open-loop replay on the real robot. The unchanged client
queries this server exactly like a policy; the server ignores the images and
returns the pre-recorded native normalised command chunks of one training
episode after another. Each inference's incoming proprio (measured joints /
gripper) is logged next to the chunk that was served, which yields
(command -> response) calibration data for the deployment machine on exactly
the training command distribution.

One rollout consumes ceil(600/8) = 75 inferences; the tape advances to the
next episode automatically after 75, so N episodes = N client rollouts run
back to back. Table must be EMPTY (the arm sweeps the poses where the bottle
would be, and the gripper closes on air).

Run (dinglab):
  cd /playpen-ssd/ting/openpi && \
  uv run scripts/serve_replay.py --port 8000 \
    --tapes-dir /playpen-ssd/ting/LFHV/data/r2r2r_gs/hf_v0_gate_passed \
    --episodes 3 \
    --log-dir /playpen-ssd/ting/LFHV/data/real2sim_replay/hf_v0_tape_replay
"""

import dataclasses
import glob
import json
import logging
import os
import pathlib

import numpy as np
import tyro

from openpi.serving import websocket_policy_server

CLIENT_STEPS = 600            # client rollout length (guide constant)
H = 8                         # client open-loop horizon
CHUNK = 16                    # chunk length the client expects


class TapePolicy:
    """Serves recorded action chunks in order; logs proprio per inference."""

    def __init__(self, tapes: list[np.ndarray], names: list[str], log_dir: str):
        self._tapes = tapes
        self._names = names
        self._log_dir = pathlib.Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._infer = 0
        self._rows = []
        self.metadata = {"tape_server": True}

    def _chunk_at(self, tape: np.ndarray, start: int) -> np.ndarray:
        out = np.zeros((CHUNK, 8), dtype=np.float64)
        if start < len(tape):
            seg = tape[start:start + CHUNK]
            out[: len(seg)] = seg
            if len(seg) < CHUNK:                     # hold the last gripper value
                out[len(seg):, 7] = seg[-1, 7]
        elif len(tape):                              # past the end: arm zeros, gripper held
            out[:, 7] = tape[-1, 7]
        return out

    def infer(self, obs: dict) -> dict:
        tape = self._tapes[self._episode % len(self._tapes)]
        chunk = self._chunk_at(tape, self._infer * H)
        self._rows.append(dict(
            infer=self._infer,
            joint_position=np.asarray(obs["observation/joint_position"]).tolist(),
            gripper_position=np.asarray(obs["observation/gripper_position"]).tolist(),
            served=chunk[:H].tolist(),
        ))
        self._infer += 1
        if self._infer >= CLIENT_STEPS // H:
            name = self._names[self._episode % len(self._tapes)]
            out = self._log_dir / f"tape_{self._episode:02d}_{name}.json"
            out.write_text(json.dumps(self._rows))
            logging.info("tape %d (%s) finished -> %s", self._episode, name, out)
            self._episode += 1
            self._infer = 0
            self._rows = []
        return {"actions": chunk}


@dataclasses.dataclass
class Args:
    tapes_dir: str
    log_dir: str
    episodes: int = 3
    port: int = 8000


def main(args: Args) -> None:
    files = sorted(glob.glob(os.path.join(args.tapes_dir, "state*.npz")))[: args.episodes]
    assert files, f"no tapes in {args.tapes_dir}"
    tapes, names = [], []
    for f in files:
        st = np.load(f)
        tapes.append(st["actions"].astype(np.float64))
        names.append(os.path.basename(f)[:-4])
        logging.info("tape loaded: %s T=%d", names[-1], len(tapes[-1]))
    policy = TapePolicy(tapes, names, args.log_dir)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata)
    logging.info("tape server on :%d, %d episodes, EMPTY TABLE ONLY", args.port, len(tapes))
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
