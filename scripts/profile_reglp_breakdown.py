from __future__ import annotations

import argparse
import cProfile
import importlib
import inspect
import io
import json
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auction import AuctionMDP


@dataclass
class TimingStore:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for name, total in sorted(self.totals.items(), key=lambda item: item[1], reverse=True):
            count = self.counts[name]
            rows.append(
                {
                    "name": name,
                    "total_seconds": total,
                    "count": count,
                    "avg_seconds": total / count if count else 0.0,
                }
            )
        return rows


def _sync_from_obj(obj: Any) -> None:
    sync = getattr(obj, "_sync", None)
    if callable(sync):
        sync()


def _timed_call(timer: TimingStore, name: str, fn: Callable[..., Any], *args: Any, sync_obj: Any = None, **kwargs: Any) -> Any:
    if sync_obj is not None:
        _sync_from_obj(sync_obj)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    if sync_obj is not None:
        _sync_from_obj(sync_obj)
    timer.add(name, time.perf_counter() - start)
    return result


class PatchManager:
    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any]] = []

    def patch(self, obj: Any, attr: str, replacement: Any) -> None:
        original = getattr(obj, attr)
        self._patches.append((obj, attr, original))
        setattr(obj, attr, replacement)

    def restore(self) -> None:
        while self._patches:
            obj, attr, original = self._patches.pop()
            setattr(obj, attr, original)


def _instrument_module(module: Any, timers: TimingStore) -> PatchManager:
    patches = PatchManager()

    def wrap_function(name: str, label: str) -> None:
        if not hasattr(module, name):
            return
        original = getattr(module, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            sync_obj = args[1] if len(args) > 1 else None
            return _timed_call(timers, label, original, *args, sync_obj=sync_obj, **kwargs)

        patches.patch(module, name, wrapped)

    if hasattr(module, "_objective_coeffs"):
        original_objective_coeffs = module._objective_coeffs

        def wrapped_objective_coeffs(*args: Any, **kwargs: Any) -> Any:
            return _timed_call(timers, "objective_coeffs", original_objective_coeffs, *args, **kwargs)

        patches.patch(module, "_objective_coeffs", wrapped_objective_coeffs)

    if hasattr(module, "_reward_matrix"):
        original_reward_matrix = module._reward_matrix

        def wrapped_reward_matrix(*args: Any, **kwargs: Any) -> Any:
            return _timed_call(timers, "reward_matrix", original_reward_matrix, *args, **kwargs)

        patches.patch(module, "_reward_matrix", wrapped_reward_matrix)

    if hasattr(module, "dsw_dx"):
        original_dsw_dx = module.dsw_dx

        def wrapped_dsw_dx(*args: Any, **kwargs: Any) -> Any:
            return _timed_call(timers, "dsw_dx", original_dsw_dx, *args, **kwargs)

        patches.patch(module, "dsw_dx", wrapped_dsw_dx)

    if hasattr(module, "dasw_dx"):
        original_dasw_dx = module.dasw_dx

        def wrapped_dasw_dx(*args: Any, **kwargs: Any) -> Any:
            return _timed_call(timers, "dasw_dx", original_dasw_dx, *args, **kwargs)

        patches.patch(module, "dasw_dx", wrapped_dasw_dx)

    diff_cls = getattr(module, "DifferentiableMDPLinearProgram", None)
    if diff_cls is not None:
        if hasattr(diff_cls, "_solve_x_numpy"):
            original_inner = diff_cls._solve_x_numpy

            def wrapped_inner(self: Any, *args: Any, **kwargs: Any) -> Any:
                return _timed_call(timers, "inner_solve", original_inner, self, *args, sync_obj=self, **kwargs)

            patches.patch(diff_cls, "_solve_x_numpy", wrapped_inner)

        if hasattr(diff_cls, "_solve_x_torch"):
            original_inner_torch = diff_cls._solve_x_torch

            def wrapped_inner_torch(self: Any, *args: Any, **kwargs: Any) -> Any:
                return _timed_call(
                    timers, "inner_solve", original_inner_torch, self, *args, sync_obj=self, **kwargs
                )

            patches.patch(diff_cls, "_solve_x_torch", wrapped_inner_torch)

        if hasattr(diff_cls, "solve_types"):
            original_solve_types = diff_cls.solve_types

            def wrapped_solve_types(self: Any, *args: Any, **kwargs: Any) -> Any:
                return _timed_call(
                    timers, "solve_types", original_solve_types, self, *args, sync_obj=self, **kwargs
                )

            patches.patch(diff_cls, "solve_types", wrapped_solve_types)

        if hasattr(diff_cls, "reverse_objective_gradient"):
            original_reverse = diff_cls.reverse_objective_gradient

            def wrapped_reverse(self: Any, *args: Any, **kwargs: Any) -> Any:
                return _timed_call(
                    timers,
                    "reverse_objective_gradient",
                    original_reverse,
                    self,
                    *args,
                    sync_obj=self,
                    **kwargs,
                )

            patches.patch(diff_cls, "reverse_objective_gradient", wrapped_reverse)

    wrap_function("_expectedrevenue_diff", "expected_value_loop")
    wrap_function("expectedrevenuegrad", "expected_grad_loop")
    wrap_function("_expectedrevenue_diff_stats", "eval_revenue_loop")
    wrap_function("_expectedperformance_diff_stats", "eval_performance_loop")

    if hasattr(module, "_calcrevenue_diff"):

        def instrumented_calcrevenue_diff(lp: Any, diff_lp: Any, types: Any, ama: Any, alpha: float) -> float:
            module._validate_diff_alpha(diff_lp, alpha)
            main_x, _ = diff_lp.solve_types(types, ama)
            main_asw = module.asw(lp.mdp, main_x, types, ama)
            revenue = module.sw(lp.mdp, main_x, types)

            start_cf = time.perf_counter()
            for i in range(lp.mdp.n_agents):
                cf_types = lp.mdp.counterfactualtype(types, i)
                cf_x, _ = diff_lp.solve_types(cf_types, ama)
                revenue += (module.asw(lp.mdp, cf_x, cf_types, ama) - main_asw) / ama.weights[i]
            _sync_from_obj(diff_lp)
            timers.add("counterfactual_revenue_loop", time.perf_counter() - start_cf)
            return revenue

        patches.patch(module, "_calcrevenue_diff", instrumented_calcrevenue_diff)

    if hasattr(module, "revenuegradb_asw_envelope"):

        def instrumented_revenuegradb_asw_envelope(lp: Any, diff_lp: Any, types: Any, ama: Any, alpha: float) -> Any:
            module._validate_diff_alpha(diff_lp, alpha)
            main_x, coeffs = diff_lp.solve_types(types, ama)
            grad_sw = module.dsw_dx(lp.mdp, main_x, types)
            rev_grad_b = diff_lp.reverse_objective_gradient(coeffs, grad_sw)

            start_cf = time.perf_counter()
            for i in range(lp.mdp.n_agents):
                cf_types = lp.mdp.counterfactualtype(types, i)
                cf_x, _ = diff_lp.solve_types(cf_types, ama)
                rev_grad_b += (cf_x - main_x) / ama.weights[i]
            _sync_from_obj(diff_lp)
            timers.add("counterfactual_grad_loop", time.perf_counter() - start_cf)
            return rev_grad_b

        patches.patch(module, "revenuegradb_asw_envelope", instrumented_revenuegradb_asw_envelope)

    if hasattr(module, "revenuegrad_asw_envelope_wb"):

        def instrumented_revenuegrad_asw_envelope_wb(
            lp: Any, diff_lp: Any, types: Any, ama: Any, alpha: float
        ) -> Any:
            module._validate_diff_alpha(diff_lp, alpha)
            main_x, coeffs = diff_lp.solve_types(types, ama)
            main_asw = module.asw(lp.mdp, main_x, types, ama)

            grad_sw = module.dsw_dx(lp.mdp, main_x, types)
            rev_grad_b = diff_lp.reverse_objective_gradient(coeffs, grad_sw)

            counterfactual_asws = np.zeros(lp.mdp.n_agents, dtype=float)
            factual_sw_by_agent = np.zeros(lp.mdp.n_agents, dtype=float)
            counterfactual_sw_by_agent = np.zeros((lp.mdp.n_agents, lp.mdp.n_agents), dtype=float)
            rewards = module._reward_matrix(lp, types)
            factual_sw_by_agent = np.tensordot(main_x, rewards, axes=([0, 1], [0, 1]))

            start_cf = time.perf_counter()
            for i in range(lp.mdp.n_agents):
                cf_types = lp.mdp.counterfactualtype(types, i)
                cf_x, _ = diff_lp.solve_types(cf_types, ama)
                counterfactual_asws[i] = module.asw(lp.mdp, cf_x, cf_types, ama)
                rev_grad_b += (cf_x - main_x) / ama.weights[i]
                cf_rewards = module._reward_matrix(lp, cf_types)
                counterfactual_sw_by_agent[i, :] = np.tensordot(cf_x, cf_rewards, axes=([0, 1], [0, 1]))
            _sync_from_obj(diff_lp)
            timers.add("counterfactual_grad_loop", time.perf_counter() - start_cf)

            w_grads = np.tensordot(rev_grad_b, rewards, axes=([0, 1], [0, 1]))
            w_grads -= np.sum(1.0 / ama.weights) * factual_sw_by_agent
            w_grads += (main_asw / (ama.weights**2)) - (counterfactual_asws / (ama.weights**2))
            for j in range(lp.mdp.n_agents):
                for i in range(lp.mdp.n_agents):
                    if i != j:
                        w_grads[j] += counterfactual_sw_by_agent[i, j] / ama.weights[i]
            return w_grads, rev_grad_b

        patches.patch(module, "revenuegrad_asw_envelope_wb", instrumented_revenuegrad_asw_envelope_wb)

    return patches


def _run_trial(module: Any, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    runtrial_signature = inspect.signature(module.runtrial)
    kwargs = {
        "num_agents": args.num_agents,
        "num_items": args.num_items,
        "num_samples": args.num_samples,
        "num_training_iters": args.num_training_iters,
        "seed": args.seed,
        "mdp_factory": AuctionMDP,
        "lr": args.lr,
        "reg_strength": args.reg_strength,
        "dist_type": args.dist_type,
        "optimize_weights": args.optimize_weights,
        "gamma": args.gamma,
    }
    if "device" in runtrial_signature.parameters:
        kwargs["device"] = args.device
    if "optimizer_name" in runtrial_signature.parameters:
        kwargs["optimizer_name"] = args.optimizer_name
    if "optimizer_lr" in runtrial_signature.parameters:
        kwargs["optimizer_lr"] = args.optimizer_lr
    return module.runtrial(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile reglp bottlenecks by backend.")
    parser.add_argument("--module", default="mdp_lp", choices=["mdp_lp", "mdp_lp_torch", "mdp_lp_torch_batch"])
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-items", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-training-iters", type=int, default=20)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg-strength", type=float, default=0.01)
    parser.add_argument("--dist-type", default="uniform", choices=["uniform", "asymmetric"])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--optimize-weights", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--optimizer-name", default="lbfgs", choices=["lbfgs", "adam"])
    parser.add_argument("--optimizer-lr", type=float, default=1.0)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--top-functions", type=int, default=20)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    module.TEST_SAMPLES = args.test_samples

    timers = TimingStore()
    patches = _instrument_module(module, timers)

    profile_text = None
    wall_start = time.perf_counter()
    try:
        if args.cprofile:
            profiler = cProfile.Profile()
            result, aux = profiler.runcall(_run_trial, module, args)
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
            stats.print_stats(args.top_functions)
            profile_text = stream.getvalue()
        else:
            result, aux = _run_trial(module, args)
    finally:
        patches.restore()
    wall_seconds = time.perf_counter() - wall_start

    payload = {
        "module": args.module,
        "wall_seconds": wall_seconds,
        "result": {k: (v.total_seconds() if hasattr(v, "total_seconds") else v) for k, v in result.items()},
        "vals_last": float(aux["vals"][-1]) if len(aux["vals"]) else None,
        "timings": timers.summary(),
        "nested_timings_overlap": True,
    }
    if profile_text is not None:
        payload["cprofile_top"] = profile_text

    print(json.dumps(payload, indent=2, default=str))

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
