from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
import torch

from lp_and_ama import (
    AMAParams,
    MDPLinearProgram,
    asw,
    dasw_dx,
    dsw_dx,
    sw,
)


TEST_SAMPLES = 10000
MIN_WEIGHT = 1e-6
DUAL_MAX_ITERS = 250
LOG_X_CLIP = 700.0
LINEAR_SOLVE_EPS = 1e-9
FLOW_RESIDUAL_TOL = 1e-6


def _objective_coeffs(lp: MDPLinearProgram, types: Any, ama: AMAParams) -> np.ndarray:
    coeffs = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            reward = np.asarray(lp.mdp.reward_from_alloc(state, action, types), dtype=float)
            coeffs[state_idx, action_idx] = float(reward @ ama.weights) + float(
                ama.boosts[state_idx, action_idx]
            )
    return coeffs


def _reward_matrix(lp: MDPLinearProgram, types: Any) -> np.ndarray:
    rewards = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list), lp.mdp.n_agents), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            rewards[state_idx, action_idx, :] = np.asarray(
                lp.mdp.reward_from_alloc(state, action, types), dtype=float
            )
    return rewards


def _project_positive_weights(weights: np.ndarray, min_weight: float = MIN_WEIGHT) -> np.ndarray:
    return np.maximum(weights, min_weight)


def _validate_diff_alpha(diff_lp: "DifferentiableMDPLinearProgram", alpha: float) -> None:
    if alpha <= 0.0:
        raise ValueError("Differentiable LP gradients require alpha > 0.")
    if not np.isclose(diff_lp.alpha, alpha, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"alpha mismatch: diff_lp.alpha={diff_lp.alpha} but function was called with alpha={alpha}."
        )


@dataclass
class DifferentiableMDPLinearProgram:
    lp: MDPLinearProgram
    alpha: float = 0.01

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("DifferentiableMDPLinearProgram requires alpha > 0.")
        self.num_states = len(self.lp.mdp.state_list)
        self.num_actions = len(self.lp.mdp.action_list)
        self.num_variables = self.num_states * self.num_actions
        flow_matrix, rhs = self._build_flow_matrix()
        self._flow_matrix = flow_matrix
        self._flow_matrix_T = flow_matrix.T
        self._rhs = rhs
        if flow_matrix.size:
            gram = flow_matrix.dot(flow_matrix.T) + LINEAR_SOLVE_EPS * np.eye(flow_matrix.shape[0])
            self._dual_init = np.linalg.solve(gram, flow_matrix)
        else:
            self._dual_init = np.zeros((0, self.num_variables), dtype=float)
        self._dual_warm_start = np.zeros(flow_matrix.shape[0], dtype=float)

    def _build_flow_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        row_indices: list[int] = []
        rhs_values: list[float] = []
        for state_idx, state in enumerate(self.lp.state_list):
            if not self.lp.mdp.nonterminal(state):
                continue
            row_indices.append(state_idx)
            rhs_values.append(1.0 if state == self.lp.mdp.startstate() else 0.0)

        num_rows = len(row_indices)
        matrix = np.zeros((num_rows, self.num_variables), dtype=float)
        rhs = np.asarray(rhs_values, dtype=float)

        for row_idx, state_idx in enumerate(row_indices):
            state_slice = slice(state_idx * self.num_actions, (state_idx + 1) * self.num_actions)
            matrix[row_idx, state_slice] += 1.0
            for prev_state_idx in range(self.num_states):
                for action_idx in range(self.num_actions):
                    transition_prob = float(
                        self.lp._transition_tensor[prev_state_idx, action_idx, state_idx]
                    )
                    if transition_prob == 0.0:
                        continue
                    col_idx = prev_state_idx * self.num_actions + action_idx
                    matrix[row_idx, col_idx] -= self.lp.gamma * transition_prob

        return matrix, rhs

    def _initial_dual(self, coeffs_flat: np.ndarray) -> np.ndarray:
        if self._dual_warm_start.size == 0:
            return self._dual_warm_start
        if not np.any(self._dual_warm_start) and self._dual_init.size:
            return self._dual_init.dot(coeffs_flat)
        return self._dual_warm_start.copy()

    def _primal_from_dual(
        self, dual: np.ndarray, coeffs_flat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        reduced_cost = coeffs_flat - self._flow_matrix_T.dot(dual)
        log_x = np.clip((reduced_cost / self.alpha) - 1.0, a_min=-LOG_X_CLIP, a_max=LOG_X_CLIP)
        x_flat = np.exp(log_x)
        return x_flat, log_x

    def _dual_objective_and_grad(self, dual: np.ndarray, coeffs_flat: np.ndarray) -> tuple[float, np.ndarray]:
        x_flat, _ = self._primal_from_dual(dual, coeffs_flat)
        objective = float(self._rhs.dot(dual) + self.alpha * np.sum(x_flat))
        gradient = self._rhs - self._flow_matrix.dot(x_flat)
        return objective, gradient

    def _solve_from_initial_dual(
        self, coeffs_flat: np.ndarray, dual0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        result = minimize(
            fun=lambda dual: self._dual_objective_and_grad(dual, coeffs_flat)[0],
            x0=dual0,
            jac=lambda dual: self._dual_objective_and_grad(dual, coeffs_flat)[1],
            method="L-BFGS-B",
            options={
                "maxiter": DUAL_MAX_ITERS,
                "ftol": 1e-15,
                "gtol": 1e-12,
                "maxls": 50,
            },
        )
        dual = result.x
        x_flat, _ = self._primal_from_dual(dual, coeffs_flat)
        residual = float(np.max(np.abs(self._rhs - self._flow_matrix.dot(x_flat))))
        return x_flat, dual, residual

    def _solve_x_numpy(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coeffs_flat = np.asarray(coeffs, dtype=float).reshape(-1)
        initializations = [self._initial_dual(coeffs_flat)]
        if self._dual_init.size:
            initializations.append(self._dual_init.dot(coeffs_flat))
        initializations.append(np.zeros_like(self._dual_warm_start))

        best_x_flat: np.ndarray | None = None
        best_dual: np.ndarray | None = None
        best_residual = float("inf")
        for dual0 in initializations:
            x_flat, dual, residual = self._solve_from_initial_dual(coeffs_flat, dual0)
            if residual < best_residual:
                best_x_flat = x_flat
                best_dual = dual
                best_residual = residual
            if residual <= FLOW_RESIDUAL_TOL:
                break

        if best_x_flat is None or best_dual is None:
            raise RuntimeError("Dual gradient solver failed to produce a candidate solution.")

        x_flat = best_x_flat
        dual = best_dual
        self._dual_warm_start = dual.copy()
        return x_flat.reshape(self.num_states, self.num_actions), dual

    def solve_x(self, coeffs: np.ndarray) -> np.ndarray:
        x, _ = self._solve_x_numpy(coeffs)
        return x

    def solve_types(self, types: Any, ama: AMAParams) -> tuple[np.ndarray, np.ndarray]:
        coeffs = _objective_coeffs(self.lp, types, ama)
        return self.solve_x(coeffs), coeffs

    def reverse_objective_gradient(self, coeffs: np.ndarray, grad_wrt_x: np.ndarray) -> np.ndarray:
        x_t, _ = self._solve_x_numpy(coeffs)
        x_flat = x_t.reshape(-1)
        grad_flat = np.asarray(grad_wrt_x, dtype=float).reshape(-1)
        weighted_grad = x_flat * grad_flat

        weighted_flow = self._flow_matrix * x_flat[np.newaxis, :]
        system_matrix = weighted_flow.dot(self._flow_matrix_T) + LINEAR_SOLVE_EPS * np.eye(
            self._flow_matrix.shape[0]
        )
        rhs = self._flow_matrix.dot(weighted_grad)
        correction = np.linalg.solve(system_matrix, rhs)
        grad_coeff = (weighted_grad - x_flat * self._flow_matrix_T.dot(correction)) / self.alpha
        return grad_coeff.reshape(self.num_states, self.num_actions)


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _calcrevenue_diff(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> float:
    _validate_diff_alpha(diff_lp, alpha)
    main_x, _ = diff_lp.solve_types(types, ama)
    main_asw = asw(lp.mdp, main_x, types, ama)
    revenue = sw(lp.mdp, main_x, types)

    for i in range(lp.mdp.n_agents):
        cf_types = lp.mdp.counterfactualtype(types, i)
        cf_x, _ = diff_lp.solve_types(cf_types, ama)
        revenue += (asw(lp.mdp, cf_x, cf_types, ama) - main_asw) / ama.weights[i]
    return revenue


def _expectedrevenue_diff(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> float:
    samples = lp.mdp.sampletypes(num_samples)
    revenues = [_calcrevenue_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return float(np.mean(revenues))


def _expectedrevenue_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    samples = lp.mdp.sampletypes(num_samples)
    revenues = [_calcrevenue_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return _mean_and_std(revenues)


def _calcmakespan_diff(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> float:
    _validate_diff_alpha(diff_lp, alpha)
    makespan_fn = getattr(lp.mdp, "makespan_from_sa", None)
    if makespan_fn is None:
        raise TypeError("Regularized makespan evaluation requires makespan_from_sa.")

    x, _ = diff_lp.solve_types(types, ama)
    total = 0.0
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            total += float(x[state_idx, action_idx]) * float(makespan_fn(state, action, types))
    return total


def _expectedmakespan_diff(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> float:
    samples = lp.mdp.sampletypes(num_samples)
    makespans = [_calcmakespan_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return float(np.mean(makespans))


def _expectedperformance_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    samples = lp.mdp.sampletypes(num_samples)
    if hasattr(lp.mdp, "makespan_from_sa"):
        performances = [-_calcmakespan_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    else:
        performances = [_calcrevenue_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return _mean_and_std(performances)


def makespangrad(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    main_x, coeffs = diff_lp.solve_types(types, ama)
    grad_wrt_x = _dmakespan_dx(lp, main_x, types)
    return diff_lp.reverse_objective_gradient(coeffs, grad_wrt_x)


def makespangrad_wb(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    b_grad = makespangrad(lp, diff_lp, types, ama, alpha)
    rewards = _reward_matrix(lp, types)
    w_grad = np.tensordot(b_grad, rewards, axes=([0, 1], [0, 1]))
    return w_grad, b_grad


def revenuegradb_asw_envelope(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    main_x, coeffs = diff_lp.solve_types(types, ama)
    grad_sw = dsw_dx(lp.mdp, main_x, types)
    rev_grad_b = diff_lp.reverse_objective_gradient(coeffs, grad_sw)

    for i in range(lp.mdp.n_agents):
        cf_types = lp.mdp.counterfactualtype(types, i)
        cf_x, _ = diff_lp.solve_types(cf_types, ama)
        rev_grad_b += (cf_x - main_x) / ama.weights[i]
    return rev_grad_b


def revenuegrad_asw_envelope_wb(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    main_x, coeffs = diff_lp.solve_types(types, ama)
    main_asw = asw(lp.mdp, main_x, types, ama)

    grad_sw = dsw_dx(lp.mdp, main_x, types)
    rev_grad_b = diff_lp.reverse_objective_gradient(coeffs, grad_sw)

    counterfactual_asws = np.zeros(lp.mdp.n_agents, dtype=float)
    factual_sw_by_agent = np.zeros(lp.mdp.n_agents, dtype=float)
    counterfactual_sw_by_agent = np.zeros((lp.mdp.n_agents, lp.mdp.n_agents), dtype=float)
    rewards = _reward_matrix(lp, types)
    factual_sw_by_agent = np.tensordot(main_x, rewards, axes=([0, 1], [0, 1]))

    for i in range(lp.mdp.n_agents):
        cf_types = lp.mdp.counterfactualtype(types, i)
        cf_x, _ = diff_lp.solve_types(cf_types, ama)
        counterfactual_asws[i] = asw(lp.mdp, cf_x, cf_types, ama)
        rev_grad_b += (cf_x - main_x) / ama.weights[i]
        cf_rewards = _reward_matrix(lp, cf_types)
        counterfactual_sw_by_agent[i, :] = np.tensordot(cf_x, cf_rewards, axes=([0, 1], [0, 1]))

    w_grads = np.tensordot(rev_grad_b, rewards, axes=([0, 1], [0, 1]))
    w_grads -= np.sum(1.0 / ama.weights) * factual_sw_by_agent
    w_grads += (main_asw / (ama.weights**2)) - (counterfactual_asws / (ama.weights**2))
    for j in range(lp.mdp.n_agents):
        for i in range(lp.mdp.n_agents):
            if i != j:
                w_grads[j] += counterfactual_sw_by_agent[i, j] / ama.weights[i]
    return w_grads, rev_grad_b


def revenuegradb(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    main_x, coeffs = diff_lp.solve_types(types, ama)
    grad_sw = dsw_dx(lp.mdp, main_x, types)
    rev_grad_b = diff_lp.reverse_objective_gradient(coeffs, grad_sw)

    main_dasw = dasw_dx(lp.mdp, main_x, types, ama)
    asw_grad_b = diff_lp.reverse_objective_gradient(coeffs, main_dasw) + main_x

    for i in range(lp.mdp.n_agents):
        cf_types = lp.mdp.counterfactualtype(types, i)
        cf_x, cf_coeffs = diff_lp.solve_types(cf_types, ama)
        deriv_asw = dasw_dx(lp.mdp, cf_x, cf_types, ama)
        counterfactual_grad = diff_lp.reverse_objective_gradient(cf_coeffs, deriv_asw) + cf_x
        rev_grad_b += (counterfactual_grad - asw_grad_b) / ama.weights[i]
    return rev_grad_b


def expectedrevenuegrad(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    samples = lp.mdp.sampletypes(num_samples)
    grad = np.zeros_like(ama.boosts)
    for types in samples:
        grad += revenuegradb_asw_envelope(lp, diff_lp, types, ama, alpha)
    return grad / num_samples


def expectedrevenuegrad_wb(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    samples = lp.mdp.sampletypes(num_samples)
    grad_b = np.zeros_like(ama.boosts)
    grad_w = np.zeros(lp.mdp.n_agents, dtype=float)
    for types in samples:
        grad_w_i, grad_b_i = revenuegrad_asw_envelope_wb(lp, diff_lp, types, ama, alpha)
        grad_b += grad_b_i
        grad_w += grad_w_i
    return grad_w / num_samples, grad_b / num_samples


def expectedmakespangrad(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    samples = lp.mdp.sampletypes(num_samples)
    grad = np.zeros_like(ama.boosts)
    for types in samples:
        grad += makespangrad(lp, diff_lp, types, ama, alpha)
    return grad / num_samples


def expectedmakespangrad_wb(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    samples = lp.mdp.sampletypes(num_samples)
    grad_b = np.zeros_like(ama.boosts)
    grad_w = np.zeros(lp.mdp.n_agents, dtype=float)
    for types in samples:
        grad_w_i, grad_b_i = makespangrad_wb(lp, diff_lp, types, ama, alpha)
        grad_b += grad_b_i
        grad_w += grad_w_i
    return grad_w / num_samples, grad_b / num_samples


def optimize_boosts(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    objective_kind: str,
    num_samples: int = 10,
    alpha: float = 0.01,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    vals = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        if objective_kind == "makespan":
            vals[i] = _expectedmakespan_diff(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            grad = expectedmakespangrad(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            ama.boosts -= lr * grad
        else:
            vals[i] = _expectedrevenue_diff(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            grad = expectedrevenuegrad(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            ama.boosts += lr * grad
    return ama, vals


def optimize_weights_and_boosts(
    lp: MDPLinearProgram,
    diff_lp: DifferentiableMDPLinearProgram,
    ama: AMAParams,
    objective_kind: str,
    num_samples: int = 10,
    alpha: float = 0.01,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    _validate_diff_alpha(diff_lp, alpha)
    vals = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        if objective_kind == "makespan":
            vals[i] = _expectedmakespan_diff(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            w_grad, b_grad = expectedmakespangrad_wb(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            ama.boosts -= lr * b_grad
            ama.weights = _project_positive_weights(ama.weights - lr * w_grad)
        else:
            vals[i] = _expectedrevenue_diff(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            w_grad, b_grad = expectedrevenuegrad_wb(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
            ama.boosts += lr * b_grad
            ama.weights = _project_positive_weights(ama.weights + lr * w_grad)
    return ama, vals


def runtrial(
    num_agents: int,
    num_items: int,
    num_samples: int,
    num_training_iters: int,
    seed: int,
    mdp_factory: Callable[..., Any],
    lr: float,
    reg_strength: float,
    dist_type: str,
    optimize_weights: bool = False,
    gamma: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    lp = MDPLinearProgram(mdp)
    diff_lp = DifferentiableMDPLinearProgram(lp, alpha=reg_strength)

    boosts = np.random.rand(*lp.x.shape)
    ama = AMAParams(np.ones(lp.mdp.n_agents, dtype=float), boosts.copy())
    vcg_ama = AMAParams(np.ones(lp.mdp.n_agents, dtype=float), np.zeros_like(boosts))

    objective_kind = "makespan" if hasattr(mdp, "makespan_from_sa") else "revenue"

    start_time = datetime.now()
    if optimize_weights:
        ama, vals = optimize_weights_and_boosts(
            lp,
            diff_lp,
            ama,
            objective_kind=objective_kind,
            num_samples=num_samples,
            alpha=reg_strength,
            lr=lr,
            num_iters=num_training_iters,
        )
    else:
        ama, vals = optimize_boosts(
            lp,
            diff_lp,
            ama,
            objective_kind=objective_kind,
            num_samples=num_samples,
            alpha=reg_strength,
            lr=lr,
            num_iters=num_training_iters,
        )
    end_time = datetime.now()

    # Keep the full reglp path independent of cvxpy by evaluating with the same
    # regularized inner oracle used during training.
    eval_alpha = reg_strength
    vcg_revenue, vcg_std = _expectedrevenue_diff_stats(
        lp, diff_lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )
    vcg_performance, vcg_performance_std = _expectedperformance_diff_stats(
        lp, diff_lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )
    ama_revenue, ama_std = _expectedrevenue_diff_stats(
        lp, diff_lp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )
    ama_performance, ama_performance_std = _expectedperformance_diff_stats(
        lp, diff_lp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )

    result = {
        "method": "reglp",
        "mdp": getattr(mdp_factory, "__name__", str(mdp_factory)),
        "dist": dist_type,
        "num_agents": num_agents,
        "num_items": num_items,
        "seed": seed,
        "num_samples": num_samples,
        "test_samples": TEST_SAMPLES,
        "training_iters": num_training_iters,
        "lr": lr,
        "reg_strength": reg_strength,
        "eval_alpha": eval_alpha,
        "vcg_performance": vcg_performance,
        "vcg_performance_std": vcg_performance_std,
        "ama_performance": ama_performance,
        "ama_performance_std": ama_performance_std,
        "vcg_revenue": vcg_revenue,
        "vcg_std": vcg_std,
        "ama_revenue": ama_revenue,
        "ama_std": ama_std,
        "runtime": end_time - start_time,
    }
    return result, {"ama": ama, "vals": vals}


def namedtuple_to_csv_line_str(result: dict[str, Any]) -> str:
    return "\t".join(str(v) for v in result.values())


def _dmakespan_dx(lp: MDPLinearProgram, x: np.ndarray, types: Any) -> np.ndarray:
    del x
    makespan_fn = getattr(lp.mdp, "makespan_from_sa", None)
    if makespan_fn is None:
        raise TypeError("makespangrad requires an MDP implementing makespan_from_sa.")
    grad = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            grad[state_idx, action_idx] = float(makespan_fn(state, action, types))
    return grad
