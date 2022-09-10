import collections
import json
import os

TOTAL = 793
CURDIR = os.path.dirname(os.path.abspath(__file__))


def read_codex_results(path):
    with open(path, "r") as f:
        raw_res = json.load(f)

    res = {}

    for key in ["PandasEval1", "PandasEval2"]:
        res[key] = {}
        for set_id, set_values in raw_res[key].items():
            for b_id, b_value in set_values.items():
                res[key][set_id, b_id] = b_value

    return res


def read_alg1_results(path):
    with open(path, "r") as f:
        raw_res = json.load(f)

    res = {}

    for key in ["PandasEval1", "PandasEval2"]:
        res[key] = {}
        for set_id, set_values in raw_res[key].items():
            for b_id, b_value in set_values.items():
                res[key][set_id, b_id] = b_value

    return res


def report_alg1_results(path, codex_baseline_path):
    codex_res = read_codex_results(codex_baseline_path)
    alg1_res = read_alg1_results(path)

    for b_key in ["PandasEval1", "PandasEval2"]:
        print(f"Current: {b_key}")
        codex_solved = sum(1 for i in codex_res[b_key].values() if i["success"])
        total = len(codex_res[b_key])
        print(f"Codex Solved: {100 * codex_solved/ total:.1f}")
        trail_ctr = collections.defaultdict(int)
        for val in alg1_res[b_key].values():
            if val["success"]:
                trail_ctr[len(val["trail"])] += 1

        print(trail_ctr)
        trail_ctr = {
            k: sum(trail_ctr.get(i, 0) for i in range(1, k + 1)) for k in trail_ctr
        }

        for ln in sorted(trail_ctr.keys()):
            print(
                f"Datana Solved (top-{ln+1}): {100 * (codex_solved + trail_ctr[ln]) / total:.1f}"
            )

        print(f"Remaining: {(total - codex_solved) - len(alg1_res[b_key])}")


if __name__ == "__main__":
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_v1_001.json"),
        os.path.join(CURDIR, "codex_baseline_results_code001-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_v1_002.json"),
        os.path.join(CURDIR, "codex_baseline_results_code002-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_v2_001.json"),
        os.path.join(CURDIR, "codex_baseline_results_code001-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_v2_002.json"),
        os.path.join(CURDIR, "codex_baseline_results_code002-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_001_no_add_desc.json"),
        os.path.join(CURDIR, "codex_baseline_results_code001-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_002_no_add_desc.json"),
        os.path.join(CURDIR, "codex_baseline_results_code002-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_001_vanilla_codebert.json"),
        os.path.join(CURDIR, "codex_baseline_results_code001-temp0.0.json"),
    )
    report_alg1_results(
        os.path.join(CURDIR, "alg1_results_run_002_vanilla_codebert.json"),
        os.path.join(CURDIR, "codex_baseline_results_code002-temp0.0.json"),
    )
