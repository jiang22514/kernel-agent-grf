"""Replay the six public Delft searches using Python's standard library only."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    require(math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= 1e-9,
            "NUMERIC_MISMATCH")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    before = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = before
    return module


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(root):
    core = load("public_search_core", root / "source/prompt_core.py")
    strategy = load("public_search_strategy", root / "source/strategy_core.py")
    plan, assets, ledger = [read(root / name) for name in ("plan.json", "assets.json", "ledger.json")]
    require(set(assets["cases"]) == {"main_12"}, "EXPECTED_ONLY_PUBLIC_MAIN12")
    require(plan["candidate_cap"] == 20 and len(plan["runs"]) == 6 and len(ledger) == 27,
            "SEARCH_DESIGN_MISMATCH")
    expected = {("astra", "control", i) for i in range(1, 4)} | {
        ("fable51", "strategy", i) for i in range(1, 4)}
    require({(r["model_label"], r["method"], r["repeat"]) for r in plan["runs"]} == expected,
            "REPETITION_DESIGN_MISMATCH")
    names, case = assets["kernels"], assets["cases"]["main_12"]
    require(len(names) == len(set(names)) == 94 and len(case["rows"]) == 376,
            "CATALOG_SIZE_MISMATCH")
    score = {(r["kernel"], r["mean_id"]): r for r in case["rows"]}
    require(set(score) == {(k, m) for k in names for m in range(1, 5)}, "SCORE_KEYS_MISMATCH")
    require(all(r["ok"] is True and math.isfinite(r["aic"]) for r in score.values()), "INVALID_SCORE")
    optimum = min(r["aic"] for r in score.values())
    entries = {r["request_file"]: r for r in ledger}
    require(len(entries) == 27 and all(not e["billing_unknown"] for e in entries.values()),
            "DUPLICATE_OR_UNRESOLVED_LEDGER")
    run_rows, round_rows, access_rows, usage_rows = [], [], [], []
    requests_seen = set()
    for run in plan["runs"]:
        rid = run["run_id"]
        folder = root / "runs" / rid
        result = read(folder / "result.json")
        require(all(result[k] == run[k] for k in
                    ("run_id", "case", "model_label", "model", "method", "repeat")), "RUN_IDENTITY_MISMATCH")
        require(result["state"] == "complete" and result["n_evals"] == 20, "INCOMPLETE_SEARCH")
        require(run["initial_pairs"] == [] and run["reasoning_effort"] == "medium", "CONFIGURATION_MISMATCH")
        batch, rounds = (5, 4) if run["method"] == "control" else (4, 5)
        require(len(result["rounds"]) == rounds and result["api_requests"] == rounds, "ROUND_COUNT_MISMATCH")
        system = (root / "prompts" / (run["method"] + ".txt")).read_text(encoding="utf-8")
        visited, seen, decision_summary = [], set(), ""
        for rd in range(1, rounds + 1):
            relative = f"runs/{rid}/round_{rd}.json"
            record = read(root / relative)
            request, response = record["request"], record["response"]
            require(request["model"] == run["model"] and request["max_tokens"] == 8192 and
                    request["reasoning"] == {"effort": "medium"} and
                    request["provider"] == {"require_parameters": True}, "MODEL_SETTINGS_MISMATCH")
            user = (core.structured_prompt(case["profile"], visited, rd, names, decision_summary)
                    if run["method"] == "control" else
                    strategy.strategy_prompt(core, case["profile"], visited, rd, names, decision_summary))
            require(request["messages"] == [{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                    f"PROMPT_MISMATCH:{rid}:{rd}")
            # Expected prompts are constructed only from the training profile,
            # unranked catalogue and this search's already visited rows.
            require(set(response["message"]) == {"role", "content"} and
                    response["message"]["role"] == "assistant", "VISIBLE_RESPONSE_SCHEMA_MISMATCH")
            obj = core.parse(response["message"]["content"])
            pairs, rejected = (core.accepted(obj, "structured", names, seen)
                               if run["method"] == "control" else
                               strategy.accepted(obj, names, seen, batch))
            require(len(pairs) <= batch, "BATCH_BUDGET_EXCEEDED")
            for key in pairs:
                require(key not in seen and key in score, "INVALID_OR_REPEATED_ACCESS")
                seen.add(key)
                visited.append({**score[key], "round": rd})
                best = min(r["aic"] for r in visited)
                access_rows.append({"run_id": rid, "model_label": run["model_label"],
                                    "repeat": run["repeat"], "access": len(visited), "round": rd,
                                    "kernel": key[0], "mean_id": key[1], "aic": score[key]["aic"],
                                    "best_so_far_aic": best, "aic_gap": best - optimum})
            saved = result["rounds"][rd - 1]
            require(saved["round"] == rd and saved["new_evals"] == len(pairs) and saved["rejected"] == rejected,
                    "ACCEPTANCE_HISTORY_MISMATCH")
            decision_summary = str(obj.get("summary", ""))[:1200]
            require(saved["decision_summary"] == decision_summary, "VISIBLE_SUMMARY_MISMATCH")
            entry = entries[relative]
            require(entry["run_id"] == rid and entry["state"] == "returned", "LEDGER_IDENTITY_MISMATCH")
            close(response["usage"]["cost"], entry["charged_or_reserved_usd"])
            close(saved["cost_usd"], entry["charged_or_reserved_usd"])
            usage_rows.append({"run_id": rid, "round": rd,
                               **{k: response["usage"].get(k) for k in
                                  ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}})
            requests_seen.add(relative)
            best = min(r["aic"] for r in visited)
            round_rows.append({"run_id": rid, "model_label": run["model_label"],
                               "repeat": run["repeat"], "round": rd, "cumulative_accesses": len(visited),
                               "best_so_far_aic": best, "aic_gap": best - optimum})
        require(len(visited) == len(seen) == 20, "SEARCH_DID_NOT_ACCESS_20_DISTINCT_MODELS")
        require(visited == result["visited"], "VISITED_ROWS_MISMATCH")
        best = min(visited, key=lambda r: r["aic"])
        gap = best["aic"] - optimum
        close(gap, result["regret"])
        require(result["hit"] == (gap <= 1e-9) and result["near_optimal"] == (gap <= 2),
                "FINAL_GRADE_MISMATCH")
        trajectory = [min(r["aic"] for r in visited[:i]) - optimum for i in range(1, 21)]
        require(len(result["regret_trajectory"]) == 20, "TRAJECTORY_LENGTH_MISMATCH")
        for left, right in zip(trajectory, result["regret_trajectory"]):
            close(left, right)
        cost = sum(e["charged_or_reserved_usd"] for e in ledger if e["run_id"] == rid)
        close(cost, result["cost_usd"])
        run_rows.append({"run_id": rid, "model_label": run["model_label"], "method": run["method"],
                         "repeat": run["repeat"], "accesses": 20, "best_kernel": best["kernel"],
                         "best_mean_id": best["mean_id"], "best_aic": best["aic"], "aic_gap": gap,
                         "champion": gap <= 1e-9, "near_optimal": gap <= 2,
                         "api_requests": rounds, "cost_usd": cost})
    require(requests_seen == set(entries), "UNACCOUNTED_REQUESTS")
    groups = {}
    for label in ("astra", "fable51"):
        group = [r for r in run_rows if r["model_label"] == label]
        groups[label] = {"searches": 3, "champions": sum(r["champion"] for r in group),
                         "near_optimal": sum(r["near_optimal"] for r in group),
                         "mean_aic_gap": sum(r["aic_gap"] for r in group) / 3}
    report = {"status": "PASS", "verified_searches": 6, "verified_requests": 27,
              "training_library_minimum_aic": optimum, "model_summary": groups,
              "experiment_cost_usd": sum(e["charged_or_reserved_usd"] for e in ledger),
              "returned_models": dict(Counter(e.get("returned_model") or "unknown" for e in ledger)),
              "providers": dict(Counter(e.get("provider") or "unknown" for e in ledger)),
              "network_calls": 0, "gp_fits": 0,
              "scope": "Three stochastic searches per procedure on one fixed public-site training set."}
    output = root / "replay_output"
    output.mkdir(exist_ok=True)
    dump(output / "summary.json", report)
    for filename, rows in (("runs.csv", run_rows), ("round_best_so_far.csv", round_rows),
                           ("access_history.csv", access_rows), ("usage.csv", usage_rows)):
        csv_write(output / filename, rows)
    return report


def verify(root):
    from unittest.mock import patch
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    for relative, expected in manifest["sha256"].items():
        target = (root / relative).resolve()
        target.relative_to(root)
        require(sha(target) == expected, "HASH_MISMATCH:" + relative)
    # No API libraries or MATLAB are imported; also block accidental networking.
    with patch("socket.create_connection", side_effect=RuntimeError("NETWORK_DISABLED")), \
            patch("socket.socket.connect", side_effect=RuntimeError("NETWORK_DISABLED")), \
            patch("urllib.request.urlopen", side_effect=RuntimeError("NETWORK_DISABLED")):
        return replay(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, ensure_ascii=False))
