from pathlib import Path
import os

from evalplus.evaluate import evaluate


def main() -> None:
    root = Path(os.environ.get("EVALPLUS_OUTPUT", "artifacts/evalplus_qwen38_base"))
    root.mkdir(parents=True, exist_ok=True)
    common = dict(
        model=os.environ.get("EVALPLUS_MODEL", "Qwen/Qwen3.8-Flash-Next-FP8"),
        backend="openai",
        base_url="http://127.0.0.1:8010/v1",
        greedy=True,
        temperature=0.0,
        n_samples=1,
        parallel_tasks=1,
        enable_thinking=False,
        trust_remote_code=True,
    )
    for dataset in ("mbpp", "humaneval"):
        print(f"START {dataset}", flush=True)
        result = evaluate(
            dataset=dataset,
            root=str(root / dataset),
            output_file=str(root / f"{dataset}.json"),
            parallel=32,
            **common,
        )
        print(f"FINISH {dataset}: {result}", flush=True)


if __name__ == "__main__":
    main()
