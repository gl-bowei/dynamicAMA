from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auction import AuctionMDP
from lp_and_ama import AMAParams, CVXPY_AVAILABLE, MDPLinearProgram, calcrevenue
import mdp_lp


MIN_EFFECTIVE_X = 1e-300


def _objective_coeffs(lp: MDPLinearProgram, types: Any, ama: AMAParams) -> np.ndarray:
    coeffs = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            reward = np.asarray(lp.mdp.reward_from_alloc(state, action, types), dtype=float)
            coeffs[state_idx, action_idx] = float(reward @ ama.weights) + float(
                ama.boosts[state_idx, action_idx]
            )
    return coeffs


def _build_flow_matrix(lp: MDPLinearProgram) -> tuple[np.ndarray, np.ndarray]:
    num_states = len(lp.mdp.state_list)
    num_actions = len(lp.mdp.action_list)
    num_variables = num_states * num_actions
    row_indices: list[int] = []
    rhs_values: list[float] = []

    for state_idx, state in enumerate(lp.state_list):
        if not lp.mdp.nonterminal(state):
            continue
        row_indices.append(state_idx)
        rhs_values.append(1.0 if state == lp.mdp.startstate() else 0.0)

    matrix = np.zeros((len(row_indices), num_variables), dtype=float)
    rhs = np.asarray(rhs_values, dtype=float)
    for row_idx, state_idx in enumerate(row_indices):
        state_slice = slice(state_idx * num_actions, (state_idx + 1) * num_actions)
        matrix[row_idx, state_slice] += 1.0
        for prev_state_idx in range(num_states):
            for action_idx in range(num_actions):
                transition_prob = float(lp._transition_tensor[prev_state_idx, action_idx, state_idx])
                if transition_prob == 0.0:
                    continue
                col_idx = prev_state_idx * num_actions + action_idx
                matrix[row_idx, col_idx] -= lp.gamma * transition_prob
    return matrix, rhs


def _flow_residual(flow_matrix: np.ndarray, rhs: np.ndarray, x: np.ndarray) -> float:
    x_flat = np.asarray(x, dtype=float).reshape(-1)
    return float(np.max(np.abs(rhs - flow_matrix.dot(x_flat))))


def _regularized_primal_value(coeffs: np.ndarray, x: np.ndarray, alpha: float) -> float:
    coeffs_flat = np.asarray(coeffs, dtype=float).reshape(-1)
    x_flat = np.clip(np.asarray(x, dtype=float).reshape(-1), MIN_EFFECTIVE_X, None)
    return float(coeffs_flat @ x_flat - alpha * np.sum(x_flat * np.log(x_flat)))


def _reset_cpu_solver(diff_lp: mdp_lp.DifferentiableMDPLinearProgram) -> None:
    diff_lp._dual_warm_start.fill(0.0)


def _build_type_cases(lp: MDPLinearProgram, base_types: list[Any], include_counterfactuals: bool) -> list[tuple[str, Any]]:
    cases: list[tuple[str, Any]] = [(f"sample_{idx}", types) for idx, types in enumerate(base_types)]
    if include_counterfactuals:
        for idx, types in enumerate(base_types):
            for agent_idx in range(lp.mdp.n_agents):
                cases.append((f"sample_{idx}_cf_{agent_idx}", lp.mdp.counterfactualtype(types, agent_idx)))
    return cases


def _benchmark_inner(
    solver_lp_factory,
    diff_lp_factory,
    cases: list[tuple[str, Any]],
    ama: AMAParams,
    alpha: float,
    repeats: int,
    fresh_each: bool,
) -> dict[str, Any]:
    solver_lp = solver_lp_factory() if not fresh_each else None
    diff_lp = diff_lp_factory(solver_lp if solver_lp is not None else solver_lp_factory()) if not fresh_each else None

    start_solver = time.perf_counter()
    solver_status_counter: Counter[str] = Counter()
    solver_count = 0
    for _ in range(repeats):
        if fresh_each:
            solver_lp = solver_lp_factory()
        for _, types in cases:
            result = solver_lp.solve(types, ama, alpha)
            solver_status_counter.update([result.status])
            solver_count += 1
    solver_elapsed = time.perf_counter() - start_solver

    start_cpu = time.perf_counter()
    cpu_count = 0
    for _ in range(repeats):
        if fresh_each:
            diff_lp = diff_lp_factory(solver_lp_factory())
        for _, types in cases:
            diff_lp.solve_types(types, ama)
            cpu_count += 1
    cpu_elapsed = time.perf_counter() - start_cpu

    return {
        "solver": {
            "wall_seconds": solver_elapsed,
            "solve_count": solver_count,
            "avg_seconds_per_solve": solver_elapsed / max(solver_count, 1),
            "status_counts": dict(solver_status_counter),
        },
        "cpu_reglp": {
            "wall_seconds": cpu_elapsed,
            "solve_count": cpu_count,
            "avg_seconds_per_solve": cpu_elapsed / max(cpu_count, 1),
        },
        "speed_ratio_cpu_over_solver": cpu_elapsed / max(solver_elapsed, 1e-12),
    }


def _benchmark_revenue(
    solver_lp_factory,
    diff_lp_factory,
    base_types: list[Any],
    ama: AMAParams,
    alpha: float,
    repeats: int,
    fresh_each: bool,
) -> dict[str, Any]:
    solver_lp = solver_lp_factory() if not fresh_each else None
    diff_lp = diff_lp_factory(solver_lp if solver_lp is not None else solver_lp_factory()) if not fresh_each else None

    start_solver = time.perf_counter()
    solver_values = []
    for _ in range(repeats):
        if fresh_each:
            solver_lp = solver_lp_factory()
        for types in base_types:
            solver_values.append(calcrevenue(solver_lp, types, ama, alpha, require_optimal=False))
    solver_elapsed = time.perf_counter() - start_solver

    start_cpu = time.perf_counter()
    cpu_values = []
    for _ in range(repeats):
        if fresh_each:
            diff_lp = diff_lp_factory(solver_lp_factory())
        for types in base_types:
            cpu_values.append(mdp_lp._calcrevenue_diff(diff_lp.lp, diff_lp, types, ama, alpha))
    cpu_elapsed = time.perf_counter() - start_cpu

    paired_diff = [float(abs(a - b)) for a, b in zip(solver_values, cpu_values)]
    return {
        "solver": {
            "wall_seconds": solver_elapsed,
            "num_values": len(solver_values),
            "avg_seconds_per_eval": solver_elapsed / max(len(solver_values), 1),
        },
        "cpu_reglp": {
            "wall_seconds": cpu_elapsed,
            "num_values": len(cpu_values),
            "avg_seconds_per_eval": cpu_elapsed / max(len(cpu_values), 1),
        },
        "speed_ratio_cpu_over_solver": cpu_elapsed / max(solver_elapsed, 1e-12),
        "max_abs_value_diff": float(max(paired_diff)) if paired_diff else 0.0,
        "mean_abs_value_diff": float(np.mean(paired_diff)) if paired_diff else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CPU reglp against original cvxpy solver LP.")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-items", type=int, default=2)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--dist-type", type=str, default="uniform")
    parser.add_argument("--solver", type=str, default="SCS")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--fresh-each", action="store_true")
    parser.add_argument("--include-counterfactuals", action="store_true")
    parser.add_argument("--skip-revenue-benchmark", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    if not CVXPY_AVAILABLE:
        raise RuntimeError("cvxpy is required for compare_cpu_reglp_vs_solver.py")

    np.random.seed(args.seed)
    mdp = AuctionMDP(args.num_agents, args.num_items, args.gamma, args.dist_type)
    base_lp = MDPLinearProgram(mdp, solver=args.solver)
    boosts = np.random.rand(*base_lp.x.shape)
    ama = AMAParams(np.ones(base_lp.mdp.n_agents, dtype=float), boosts.copy())
    base_types = list(base_lp.mdp.sampletypes(args.num_cases))
    cases = _build_type_cases(base_lp, base_types, args.include_counterfactuals)
    flow_matrix, rhs = _build_flow_matrix(base_lp)

    def solver_lp_factory() -> MDPLinearProgram:
        return MDPLinearProgram(mdp, solver=args.solver)

    def diff_lp_factory(lp: MDPLinearProgram) -> mdp_lp.DifferentiableMDPLinearProgram:
        return mdp_lp.DifferentiableMDPLinearProgram(lp, alpha=args.alpha)

    case_results = []
    status_counter: Counter[str] = Counter()
    for label, types in cases:
        solver_lp = solver_lp_factory()
        diff_lp = diff_lp_factory(solver_lp_factory())
        _reset_cpu_solver(diff_lp)
        solver_result = solver_lp.solve(types, ama, args.alpha)
        x_cpu, coeffs = diff_lp.solve_types(types, ama)
        cpu_residual = _flow_residual(flow_matrix, rhs, x_cpu)
        solver_residual = _flow_residual(flow_matrix, rhs, solver_result.x)
        status_counter.update([solver_result.status])
        case_results.append(
            {
                "label": label,
                "solver_status": solver_result.status,
                "solver_objective_value": float(solver_result.objective_value),
                "cpu_objective_value": _regularized_primal_value(coeffs, x_cpu, args.alpha),
                "objective_abs_diff": float(
                    abs(float(solver_result.objective_value) - _regularized_primal_value(coeffs, x_cpu, args.alpha))
                ),
                "solver_residual": solver_residual,
                "cpu_residual": cpu_residual,
                "x_linf_diff": float(np.max(np.abs(solver_result.x - x_cpu))),
                "x_l1_diff": float(np.sum(np.abs(solver_result.x - x_cpu))),
            }
        )

    summary = {
        "num_cases": len(case_results),
        "solver_status_counts": dict(status_counter),
        "max_x_linf_diff": float(max(case["x_linf_diff"] for case in case_results)),
        "mean_x_linf_diff": float(np.mean([case["x_linf_diff"] for case in case_results])),
        "max_x_l1_diff": float(max(case["x_l1_diff"] for case in case_results)),
        "mean_cpu_residual": float(np.mean([case["cpu_residual"] for case in case_results])),
        "mean_solver_residual": float(np.mean([case["solver_residual"] for case in case_results])),
        "max_objective_abs_diff": float(max(case["objective_abs_diff"] for case in case_results)),
        "mean_objective_abs_diff": float(np.mean([case["objective_abs_diff"] for case in case_results])),
    }

    benchmark = {
        "inner_solve": _benchmark_inner(
            solver_lp_factory,
            diff_lp_factory,
            cases,
            ama,
            args.alpha,
            args.benchmark_repeats,
            args.fresh_each,
        )
    }
    if not args.skip_revenue_benchmark:
        benchmark["revenue_eval"] = _benchmark_revenue(
            solver_lp_factory,
            diff_lp_factory,
            base_types,
            ama,
            args.alpha,
            args.benchmark_repeats,
            args.fresh_each,
        )

    payload = {
        "config": vars(args),
        "summary": summary,
        "benchmark": benchmark,
        "cases": case_results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_output:
        pathlib.Path(args.json_output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
