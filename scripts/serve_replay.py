"""Guarded tape server: replays hf_v0 training-episode command streams over
the standard websocket policy protocol, with a divergence guard.

Purpose: EMPTY-TABLE open-loop replay on the real robot to verify plant
consistency. The unchanged client queries this server exactly like a policy;
the server returns the pre-recorded native normalised command chunks of one
training episode after another and logs each inference's incoming proprio
next to the chunk that was served.

Safety guard (added after the 2026-08-08 table-edge incident): every
inference compares the robot's measured joint displacement against the
episode's stored sim-executed trajectory (qpos7) at the same step. Once the
max-joint displacement divergence exceeds --guard-thresh, the rest of the
episode is served as zero arm commands with the gripper held, so a plant
mismatch terminates the episode in place instead of integrating into a novel
Cartesian path. --max-anchors serves only the first N anchors (staged
release: e.g. approach segment only) and holds afterwards.

One rollout consumes ceil(600/8) = 75 inferences; the tape advances to the
next episode automatically after 75, so N episodes = N client rollouts run
back to back. Table must be EMPTY (the arm sweeps the poses where the bottle
would be, and the gripper closes on air).

Run (dinglab):
  cd /playpen-ssd/ting/openpi && \
  uv run scripts/serve_replay.py --port 8000 \
    --tapes-dir /playpen-ssd/ting/LFHV/data/r2r2r_gs/hf_v0_gate_passed \
    --episodes 3 \
    --log-dir /playpen-ssd/ting/LFHV/data/real2sim_replay/hf_v0_guarded_replay
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
    """Serves recorded action chunks in order; guards on sim divergence."""

    def __init__(self, tapes, names, refs, log_dir, guard_thresh, max_anchors, start_thresh):
        self._tapes = tapes
        self._names = names
        self._refs = refs                 # per-episode reference qpos7 [T,7]
        self._log_dir = pathlib.Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._guard_thresh = guard_thresh
        self._max_anchors = max_anchors
        self._start_thresh = start_thresh
        self._episode = 0
        self._reset_episode_state()
        self.metadata = {"tape_server": True, "guarded": True}

    def _reset_episode_state(self):
        self._infer = 0
        self._rows = []
        self._q0 = None
        self._start_off = None
        self._start_hold = False
        self._tripped_at = None
        self._last_grip = 0.0

    def _tape_chunk(self, tape, start):
        out = np.zeros((CHUNK, 8), dtype=np.float64)
        if start < len(tape):
            seg = tape[start:start + CHUNK]
            out[: len(seg)] = seg
            if len(seg) < CHUNK:                     # hold the last gripper value
                out[len(seg):, 7] = seg[-1, 7]
        elif len(tape):                              # past the end: arm zeros, gripper held
            out[:, 7] = tape[-1, 7]
        return out

    def _hold_chunk(self):
        out = np.zeros((CHUNK, 8), dtype=np.float64)
        out[:, 7] = self._last_grip
        return out

    def infer(self, obs: dict) -> dict:
        ep = self._episode % len(self._tapes)
        tape, ref = self._tapes[ep], self._refs[ep]
        i = self._infer
        q = np.asarray(obs["observation/joint_position"], dtype=np.float64)

        if i == 0:
            self._q0 = q.copy()
            # Start gate: replaying a trajectory from a wrong start would sweep
            # a shifted Cartesian path; hold the whole episode instead.
            self._start_off = float(np.abs(q - ref[0]).max())
            if self._start_off > self._start_thresh:
                self._start_hold = True
                logging.warning("start gate: offset %.3f rad > %.2f — holding whole episode",
                                self._start_off, self._start_thresh)
        # Displacement-based divergence: subtracting the anchor-0 offset makes
        # the check robust to the small home-vs-episode-start mismatch
        # (measured 0.053 rad max on tape_00).
        t = min(i * H, len(ref) - 1)
        div = float(np.abs((q - self._q0) - (ref[t] - ref[0])).max())
        if not self._start_hold and self._tripped_at is None and div > self._guard_thresh:
            self._tripped_at = i
            logging.warning("guard TRIPPED at anchor %d (div %.3f rad) — serving zeros", i, div)

        held = self._max_anchors is not None and i >= self._max_anchors
        if self._start_hold:
            mode = "start_gate"
        elif self._tripped_at is not None:
            mode = "guard"
        elif held:
            mode = "hold"
        else:
            mode = "tape"
        chunk = self._tape_chunk(tape, i * H) if mode == "tape" else self._hold_chunk()
        self._last_grip = float(chunk[H - 1, 7])

        self._rows.append(dict(
            infer=i,
            mode=mode,
            div=round(div, 4),
            joint_position=q.tolist(),
            gripper_position=np.asarray(obs["observation/gripper_position"]).tolist(),
            served=chunk[:H].tolist(),
        ))
        self._infer += 1
        if self._infer >= CLIENT_STEPS // H:
            name = self._names[ep]
            summary = dict(
                tape=name,
                max_div=max(r["div"] for r in self._rows),
                tripped_at=self._tripped_at,
                start_off=self._start_off,
                start_hold=self._start_hold,
                max_anchors=self._max_anchors,
                guard_thresh=self._guard_thresh,
            )
            out = self._log_dir / f"tape_{self._episode:02d}_{name}.json"
            out.write_text(json.dumps(dict(summary=summary, rows=self._rows)))
            logging.info("tape %d (%s) finished, max_div %.3f, tripped_at %s -> %s",
                         self._episode, name, summary["max_div"], self._tripped_at, out)
            self._episode += 1
            self._reset_episode_state()
        return {"actions": chunk}


@dataclasses.dataclass
class Args:
    tapes_dir: str
    log_dir: str
    episodes: int = 3
    port: int = 8000
    # Max-joint displacement divergence [rad] vs the episode's sim trajectory
    # before the guard cuts the episode. Plant fit error alone is ~0.14 e2e,
    # so 0.3 leaves margin above a healthy plant and cut the 2026-08-08
    # incident trace at anchor 3 (retrospective check).
    guard_thresh: float = 0.3
    # Serve only the first N anchors, hold afterwards (staged release);
    # None = full episode. 12 anchors ~= the approach segment.
    max_anchors: int | None = None
    # Max-joint offset [rad] between the measured start pose and the tape's
    # reference start before the whole episode is held (never served).
    start_thresh: float = 0.35


def main(args: Args) -> None:
    files = sorted(glob.glob(os.path.join(args.tapes_dir, "state*.npz")))[: args.episodes]
    assert files, f"no tapes in {args.tapes_dir}"
    tapes, names, refs = [], [], []
    for f in files:
        st = np.load(f)
        tapes.append(st["actions"].astype(np.float64))
        refs.append(st["qpos7"].astype(np.float64))
        names.append(os.path.basename(f)[:-4])
        logging.info("tape loaded: %s T=%d", names[-1], len(tapes[-1]))
    policy = TapePolicy(tapes, names, refs, args.log_dir, args.guard_thresh,
                        args.max_anchors, args.start_thresh)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata)
    logging.info("guarded tape server on :%d, %d episodes, thresh %.2f, max_anchors %s, EMPTY TABLE ONLY",
                 args.port, len(tapes), args.guard_thresh, args.max_anchors)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
