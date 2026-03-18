from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auction import AuctionMDP
from lp_and_ama import AMAParams, MDPLinearProgram, dsw_dx
import mdp_lp
import mdp_lp_torch


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


def _reset_warm_start(solver: Any) -> None:
    warm = getattr(solver, "_dual_warm_start", None)
    if warm is None:
        return
    if isinstance(warm, np.ndarray):
        warm.fill(0.0)
        return
    if torch.is_tensor(warm):
        with torch.no_grad():
            warm.zero_()


def _sync(device: str | torch.device | None) -> None:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _benchmark_solver(
    solver_factory,
    coeffs_list: list[np.ndarray],
    repeats: int,
    device: str | torch.device | None,
    fresh_each: bool,
) -> dict[str, float]:
    start = time.perf_counter()
    solve_count = 0
    for _ in range(repeats):
        solver = solver_factory() if not fresh_each else None
        for coeffs in coeffs_list:
            current_solver = solver_factory() if fresh_each else solver
            _reset_warm_start(current_solver)
            _sync(device)
            current_solver.solve_x(coeffs)
            _sync(device)
            solve_count += 1
    elapsed = time.perf_counter() - start
    return {
        "wall_seconds": elapsed,
        "solve_count": solve_count,
        "avg_seconds_per_solve": elapsed / max(solve_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-solve CPU reglp vs torch single solver.")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-items", type=int, default=2)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--dist-type", type=str, default="uniform")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--optimizer-name", type=str, default="lbfgs")
    parser.add_argument("--optimizer-lr", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=250)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--fresh-each", action="store_true")
    parser.add_argument("--include-counterfactuals", action="store_true")
    parser.add_argument("--check-gradient", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mdp = AuctionMDP(args.num_agents, args.num_items, args.gamma, args.dist_type)
    lp = MDPLinearProgram(mdp)
    boosts = np.random.rand(*lp.x.shape)
    ama = AMAParams(np.ones(lp.mdp.n_agents, dtype=float), boosts.copy())
    flow_matrix, rhs = _build_flow_matrix(lp)

    base_types = list(lp.mdp.sampletypes(args.num_cases))
    type_cases: list[tuple[str, Any]] = [(f"sample_{idx}", types) for idx, types in enumerate(base_types)]
    if args.include_counterfactuals:
        for idx, types in enumerate(base_types):
            for agent_idx in range(lp.mdp.n_agents):
                type_cases.append((f"sample_{idx}_cf_{agent_idx}", lp.mdp.counterfactualtype(types, agent_idx)))

    coeffs_list = [_objective_coeffs(lp, types, ama) for _, types in type_cases]

    def cpu_factory():
        return mdp_lp.DifferentiableMDPLinearProgram(lp, alpha=args.alpha)

    def torch_factory():
        return mdp_lp_torch.TorchDifferentiableMDPLinearProgram(
            lp,
            alpha=args.alpha,
            device=args.device,
            optimizer_name=args.optimizer_name,
            optimizer_lr=args.optimizer_lr,
            max_iters=args.max_iters,
        )

    cases = []
    optimizer_counter: Counter[str] = Counter()
    for (label, types), coeffs in zip(type_cases, coeffs_list):
        cpu_solver = cpu_factory()
        torch_solver = torch_factory()
        _reset_warm_start(cpu_solver)
        _reset_warm_start(torch_solver)

        x_cpu = cpu_solver.solve_x(coeffs)
        x_torch = torch_solver.solve_x(coeffs)

        cpu_residual = _flow_residual(flow_matrix, rhs, x_cpu)
        torch_residual = float(torch_solver.last_solve_info.get("best_residual", np.nan))
        optimizer_counter.update([str(torch_solver.last_solve_info.get("optimizer_name", "unknown"))])

        case = {
            "label": label,
            "cpu_residual": cpu_residual,
            "torch_residual": torch_residual,
            "x_linf_diff": float(np.max(np.abs(x_cpu - x_torch))),
            "x_l1_diff": float(np.sum(np.abs(x_cpu - x_torch))),
            "cpu_primal_value": _regularized_primal_value(coeffs, x_cpu, args.alpha),
            "torch_primal_value": _regularized_primal_value(coeffs, x_torch, args.alpha),
            "torch_optimizer_name": torch_solver.last_solve_info.get("optimizer_name"),
            "torch_optimizer_steps": torch_solver.last_solve_info.get("optimizer_steps"),
        }
        if args.check_gradient:
            grad_wrt_x = dsw_dx(lp.mdp, x_cpu, types)
            grad_cpu = cpu_solver.reverse_objective_gradient(coeffs, grad_wrt_x)
            grad_torch = torch_solver.reverse_objective_gradient(coeffs, grad_wrt_x)
            case["grad_linf_diff"] = float(np.max(np.abs(grad_cpu - grad_torch)))
            case["grad_l1_diff"] = float(np.sum(np.abs(grad_cpu - grad_torch)))
        cases.append(case)

    cpu_bench = _benchmark_solver(cpu_factory, coeffs_list, args.benchmark_repeats, None, args.fresh_each)
    torch_bench = _benchmark_solver(torch_factory, coeffs_list, args.benchmark_repeats, args.device, args.fresh_each)

    summary = {
        "num_cases": len(cases),
        "max_x_linf_diff": float(max(case["x_linf_diff"] for case in cases)),
        "mean_x_linf_diff": float(np.mean([case["x_linf_diff"] for case in cases])),
        "max_x_l1_diff": float(max(case["x_l1_diff"] for case in cases)),
        "mean_cpu_residual": float(np.mean([case["cpu_residual"] for case in cases])),
        "mean_torch_residual": float(np.mean([case["torch_residual"] for case in cases])),
        "max_torch_residual": float(max(case["torch_residual"] for case in cases)),
        "max_primal_value_diff": float(
            max(abs(case["cpu_primal_value"] - case["torch_primal_value"]) for case in cases)
        ),
        "torch_optimizer_counts": dict(optimizer_counter),
    }
    if args.check_gradient:
        summary["max_grad_linf_diff"] = float(max(case["grad_linf_diff"] for case in cases))
        summary["mean_grad_linf_diff"] = float(np.mean([case["grad_linf_diff"] for case in cases]))

    payload = {
        "config": vars(args),
        "summary": summary,
        "benchmark": {
            "cpu": cpu_bench,
            "torch": torch_bench,
            "speed_ratio_torch_over_cpu": torch_bench["wall_seconds"] / max(cpu_bench["wall_seconds"], 1e-12),
        },
        "cases": cases,
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
