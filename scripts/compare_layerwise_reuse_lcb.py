#!/usr/bin/env python3
"""Pair frozen LCB per-task results across runs without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAPS = 10_000
SEED = 42
EXPECTED_TASK_COUNT = 182
EXPECTED_TASK_ID_SHA256 = "ad8b7e21195710bd04fbe4794013aeddb9be2172e3ee2309dbb0caaa2dea5383"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_outcomes(path: Path) -> tuple[dict[str, tuple[int, str]], str]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected non-empty LCB row list in {path}")
    outcomes: dict[str, tuple[int, str]] = {}
    for row in rows:
        question_id = str(row.get("question_id", ""))
        grade = row.get("graded_list")
        contest = str(row.get("contest_id", ""))
        if not question_id or question_id in outcomes:
            raise ValueError(f"missing or duplicate question_id in {path}: {question_id!r}")
        if not isinstance(grade, list) or len(grade) != 1 or not isinstance(grade[0], bool):
            raise ValueError(f"expected exactly one boolean grade for {question_id} in {path}")
        if not contest:
            raise ValueError(f"missing contest_id for {question_id} in {path}")
        outcomes[question_id] = (int(grade[0]), contest)
    ids = sorted(outcomes)
    return outcomes, hashlib.sha256("\n".join(ids).encode()).hexdigest()


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_metrics(
    candidate: dict[str, tuple[int, str]],
    control: dict[str, tuple[int, str]],
    seed: int = SEED,
    n_bootstrap: int = BOOTSTRAPS,
) -> dict[str, Any]:
    if candidate.keys() != control.keys():
        raise ValueError("candidate and control question-ID sets differ")
    ids = sorted(candidate)
    differences = np.asarray([candidate[key][0] - control[key][0] for key in ids], dtype=np.int8)
    wins = int((differences == 1).sum())
    losses = int((differences == -1).sum())
    rng = np.random.default_rng(seed)
    task_bootstrap = rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True).mean(axis=1)

    contests = sorted({candidate[key][1] for key in ids})
    contest_deltas = np.asarray(
        [sum(candidate[key][0] - control[key][0] for key in ids if candidate[key][1] == contest) for contest in contests],
        dtype=np.float64,
    )
    contest_sizes = np.asarray(
        [sum(candidate[key][1] == contest for key in ids) for contest in contests], dtype=np.float64
    )
    sampled_contests = rng.integers(0, len(contests), size=(n_bootstrap, len(contests)))
    cluster_bootstrap = contest_deltas[sampled_contests].sum(axis=1) / contest_sizes[sampled_contests].sum(axis=1)
    return {
        "tasks": len(ids),
        "candidate_solved": int(sum(value[0] for value in candidate.values())),
        "control_solved": int(sum(value[0] for value in control.values())),
        "paired_delta_solved": wins - losses,
        "paired_delta_pass_rate": float(differences.mean()),
        "candidate_only_passes": wins,
        "control_only_passes": losses,
        "task_bootstrap_95_ci": [float(v) for v in np.quantile(task_bootstrap, [0.025, 0.975])],
        "contest_cluster_bootstrap_95_ci": [float(v) for v in np.quantile(cluster_bootstrap, [0.025, 0.975])],
        "exact_mcnemar_p": exact_mcnemar(wins, losses),
        "contest_clusters": len(contests),
        "task_ids_sha256_sorted_newline_joined": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="candidate *_eval_all.json")
    parser.add_argument("--control", action="append", required=True, metavar="LABEL=PATH", help="reference *_eval_all.json (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_hash = load_outcomes(args.candidate)
    if len(candidate) != EXPECTED_TASK_COUNT or candidate_hash != EXPECTED_TASK_ID_SHA256:
        raise ValueError(
            "candidate does not match the frozen historical LCB task set: "
            f"count={len(candidate)}, hash={candidate_hash}"
        )
    comparisons = []
    for specification in args.control:
        if "=" not in specification:
            parser.error("each --control must be LABEL=PATH")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        control, control_hash = load_outcomes(path)
        if candidate_hash != control_hash:
            raise ValueError(f"frozen question ID hash differs for control {label}")
        comparisons.append({
            "candidate": str(args.candidate),
            "control": label,
            "candidate_file_sha256": sha256(args.candidate),
            "control_file": str(path),
            "control_file_sha256": control_hash and sha256(path),
            **paired_metrics(candidate, control),
        })
    result = {
        "evaluation_note": "Exploratory paired comparison on the repeatedly inspected historical LCB set; intervals are descriptive.",
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
