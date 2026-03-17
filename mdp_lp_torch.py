from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
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
LOG_X_CLIP = 80.0
LINEAR_SOLVE_EPS = 1e-9
FLOW_RESIDUAL_TOL = 1e-6
DEFAULT_MAX_ITERS = 250
DUAL_CLIP = 1e4
FAILED_RESIDUAL = 1e308
LBFGS_FALLBACK_LR = 0.1
ADAM_FALLBACK_LR = 0.02
GD_FALLBACK_LR = 1.0
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


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _validate_diff_alpha(diff_lp: "TorchDifferentiableMDPLinearProgram", alpha: float) -> None:
    if alpha <= 0.0:
        raise ValueError("Differentiable LP gradients require alpha > 0.")
    if not np.isclose(diff_lp.alpha, alpha, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"alpha mismatch: diff_lp.alpha={diff_lp.alpha} but function was called with alpha={alpha}."
        )


@dataclass
class TorchDifferentiableMDPLinearProgram:
    lp: MDPLinearProgram
    alpha: float = 0.01
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float64
    optimizer_name: str = "lbfgs"
    optimizer_lr: float = 1.0
    max_iters: int = DEFAULT_MAX_ITERS
    grad_tolerance: float = 1e-12
    change_tolerance: float = 1e-15
    adam_patience: int = 20
    dual_clip: float = DUAL_CLIP

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("TorchDifferentiableMDPLinearProgram requires alpha > 0.")
        self.device = self._resolve_device(self.device)
        self.num_states = len(self.lp.mdp.state_list)
        self.num_actions = len(self.lp.mdp.action_list)
        self.num_variables = self.num_states * self.num_actions

        flow_matrix_np, rhs_np = self._build_flow_matrix()
        self._flow_matrix = torch.as_tensor(flow_matrix_np, dtype=self.dtype, device=self.device)
        self._flow_matrix_T = self._flow_matrix.transpose(0, 1).contiguous()
        self._rhs = torch.as_tensor(rhs_np, dtype=self.dtype, device=self.device)
        self._identity = torch.eye(self._flow_matrix.shape[0], dtype=self.dtype, device=self.device)
        if self._flow_matrix.numel():
            gram = self._flow_matrix @ self._flow_matrix_T + LINEAR_SOLVE_EPS * self._identity
            self._dual_init = torch.linalg.solve(gram, self._flow_matrix)
        else:
            self._dual_init = torch.zeros((0, self.num_variables), dtype=self.dtype, device=self.device)
        self._dual_warm_start = torch.zeros(self._flow_matrix.shape[0], dtype=self.dtype, device=self.device)
        self.last_solve_info: dict[str, Any] = {}

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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

    def _coeffs_to_tensor(self, coeffs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(coeffs, dtype=float).reshape(-1), dtype=self.dtype, device=self.device)

    def _initial_dual(self, coeffs_flat: torch.Tensor) -> torch.Tensor:
        if self._dual_warm_start.numel() == 0:
            return self._dual_warm_start
        if torch.count_nonzero(self._dual_warm_start).item() == 0 and self._dual_init.numel():
            return self._dual_init @ coeffs_flat
        return self._dual_warm_start.clone()

    def _primal_from_dual(
        self, dual: torch.Tensor, coeffs_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced_cost = coeffs_flat - self._flow_matrix_T @ dual
        log_x = torch.clamp((reduced_cost / self.alpha) - 1.0, min=-LOG_X_CLIP, max=LOG_X_CLIP)
        x_flat = torch.exp(log_x)
        x_flat = torch.clamp(x_flat, min=MIN_EFFECTIVE_X)
        return x_flat, log_x

    def _clip_dual_(self, dual: torch.Tensor) -> None:
        if self.dual_clip > 0.0:
            dual.clamp_(min=-self.dual_clip, max=self.dual_clip)

    def _evaluate_candidate(
        self, dual: torch.Tensor, coeffs_flat: torch.Tensor
    ) -> tuple[torch.Tensor | None, float, float, bool]:
        with torch.no_grad():
            dual_eval = dual.detach().clone()
            self._clip_dual_(dual_eval)
            x_flat, _ = self._primal_from_dual(dual_eval, coeffs_flat)
            flow = self._flow_matrix @ x_flat
            residual_tensor = torch.max(torch.abs(self._rhs - flow))
            objective_tensor = torch.dot(self._rhs, dual_eval) + self.alpha * torch.sum(x_flat)
            finite = bool(
                torch.isfinite(x_flat).all()
                and torch.isfinite(flow).all()
                and torch.isfinite(residual_tensor)
                and torch.isfinite(objective_tensor)
            )
            if not finite:
                return None, FAILED_RESIDUAL, float("inf"), False
            return x_flat.detach(), float(residual_tensor.item()), float(objective_tensor.item()), True

    def _dual_objective(self, dual: torch.Tensor, coeffs_flat: torch.Tensor) -> torch.Tensor:
        x_flat, _ = self._primal_from_dual(dual, coeffs_flat)
        return torch.dot(self._rhs, dual) + self.alpha * torch.sum(x_flat)

    def _dual_gradient(self, dual: torch.Tensor, coeffs_flat: torch.Tensor) -> torch.Tensor:
        x_flat, _ = self._primal_from_dual(dual, coeffs_flat)
        return self._rhs - self._flow_matrix @ x_flat

    def _optimize_dual_lbfgs(
        self, coeffs_flat: torch.Tensor, dual0: torch.Tensor, optimizer_lr: float
    ) -> tuple[torch.Tensor | None, torch.Tensor, float, int, dict[str, Any]]:
        dual = torch.nn.Parameter(dual0.clone())
        optimizer = torch.optim.LBFGS(
            [dual],
            lr=optimizer_lr,
            max_iter=self.max_iters,
            tolerance_grad=self.grad_tolerance,
            tolerance_change=self.change_tolerance,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        step_counter = {"count": 0}
        failure_reason = None

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            with torch.no_grad():
                self._clip_dual_(dual.data)
            objective = self._dual_objective(dual, coeffs_flat)
            if not torch.isfinite(objective):
                raise RuntimeError("non_finite_objective")
            objective.backward()
            if dual.grad is None or not torch.isfinite(dual.grad).all():
                raise RuntimeError("non_finite_gradient")
            step_counter["count"] += 1
            return objective

        try:
            optimizer.step(closure)
        except RuntimeError as exc:
            failure_reason = str(exc)

        with torch.no_grad():
            self._clip_dual_(dual.data)
        x_flat, residual, objective_value, finite = self._evaluate_candidate(dual, coeffs_flat)
        debug = {
            "failure_reason": failure_reason,
            "objective_value": objective_value,
            "finite": finite,
            "optimizer_lr": optimizer_lr,
        }
        return x_flat, dual.detach(), float(residual), step_counter["count"], debug

    def _optimize_dual_adam(
        self, coeffs_flat: torch.Tensor, dual0: torch.Tensor, optimizer_lr: float
    ) -> tuple[torch.Tensor | None, torch.Tensor, float, int, dict[str, Any]]:
        dual = torch.nn.Parameter(dual0.clone())
        optimizer = torch.optim.Adam([dual], lr=optimizer_lr)
        best_dual = dual.detach().clone()
        best_x_flat, best_residual, best_objective_value, finite = self._evaluate_candidate(best_dual, coeffs_flat)
        if not finite or best_x_flat is None:
            best_residual = FAILED_RESIDUAL
            best_objective_value = float("inf")
        patience = 0
        failure_reason = None

        for step in range(self.max_iters):
            optimizer.zero_grad()
            objective = self._dual_objective(dual, coeffs_flat)
            if not torch.isfinite(objective):
                failure_reason = "non_finite_objective"
                break
            objective.backward()
            if dual.grad is None or not torch.isfinite(dual.grad).all():
                failure_reason = "non_finite_gradient"
                break
            optimizer.step()
            with torch.no_grad():
                self._clip_dual_(dual.data)
            x_flat, residual, objective_value, finite = self._evaluate_candidate(dual, coeffs_flat)
            if not finite or x_flat is None:
                failure_reason = "non_finite_candidate"
                break
            if residual < best_residual:
                best_residual = residual
                best_dual = dual.detach().clone()
                best_x_flat = x_flat.clone()
                best_objective_value = objective_value
                patience = 0
            else:
                patience += 1
            if residual <= FLOW_RESIDUAL_TOL or patience >= self.adam_patience:
                break
        debug = {
            "failure_reason": failure_reason,
            "objective_value": best_objective_value,
            "finite": best_x_flat is not None and np.isfinite(best_residual),
            "optimizer_lr": optimizer_lr,
        }
        return best_x_flat, best_dual, float(best_residual), step + 1, debug

    def _optimize_dual_gd(
        self, coeffs_flat: torch.Tensor, dual0: torch.Tensor, optimizer_lr: float
    ) -> tuple[torch.Tensor | None, torch.Tensor, float, int, dict[str, Any]]:
        with torch.no_grad():
            dual = dual0.detach().clone()
            self._clip_dual_(dual)
        best_x_flat, best_residual, best_objective_value, finite = self._evaluate_candidate(dual, coeffs_flat)
        if not finite or best_x_flat is None:
            best_residual = FAILED_RESIDUAL
            best_objective_value = float("inf")
        best_dual = dual.clone()
        failure_reason = None
        step_lr = optimizer_lr
        min_step_lr = 1e-8

        for step in range(self.max_iters):
            grad = self._dual_gradient(dual, coeffs_flat)
            if not torch.isfinite(grad).all():
                failure_reason = "non_finite_gradient"
                break
            grad_norm = float(torch.linalg.norm(grad).item())
            if grad_norm <= self.grad_tolerance:
                break
            grad_scale = max(float(torch.max(torch.abs(grad)).item()), 1.0)
            direction = grad / grad_scale

            _, _, current_objective, current_finite = self._evaluate_candidate(dual, coeffs_flat)
            if not current_finite:
                failure_reason = "non_finite_current_candidate"
                break
            accepted = False
            trial_lr = step_lr
            while trial_lr >= min_step_lr:
                candidate_dual = dual - trial_lr * direction
                self._clip_dual_(candidate_dual)
                candidate_x, candidate_residual, candidate_objective, candidate_finite = self._evaluate_candidate(
                    candidate_dual, coeffs_flat
                )
                if candidate_finite and np.isfinite(candidate_objective) and candidate_objective <= current_objective:
                    dual = candidate_dual
                    accepted = True
                    if candidate_x is not None and candidate_residual < best_residual:
                        best_residual = candidate_residual
                        best_objective_value = candidate_objective
                        best_dual = candidate_dual.clone()
                        best_x_flat = candidate_x.clone()
                    step_lr = min(trial_lr * 1.1, optimizer_lr)
                    if candidate_residual <= FLOW_RESIDUAL_TOL:
                        debug = {
                            "failure_reason": None,
                            "objective_value": best_objective_value,
                            "finite": True,
                            "optimizer_lr": optimizer_lr,
                            "final_step_lr": step_lr,
                            "grad_scale": grad_scale,
                        }
                        return best_x_flat, best_dual, float(best_residual), step + 1, debug
                    break
                trial_lr *= 0.5

            if not accepted:
                failure_reason = "line_search_failed"
                break

        debug = {
            "failure_reason": failure_reason,
            "objective_value": best_objective_value,
            "finite": best_x_flat is not None and np.isfinite(best_residual),
            "optimizer_lr": optimizer_lr,
            "final_step_lr": step_lr,
            "grad_scale": grad_scale if 'grad_scale' in locals() else None,
        }
        return best_x_flat, best_dual, float(best_residual), step + 1, debug

    def _solve_from_initial_dual(
        self, coeffs_flat: torch.Tensor, dual0: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, float, int, dict[str, Any]]:
        name = self.optimizer_name.lower()
        if name == "lbfgs":
            x_flat, dual, residual, steps, debug = self._optimize_dual_lbfgs(
                coeffs_flat, dual0, self.optimizer_lr
            )
            if x_flat is not None and residual <= FLOW_RESIDUAL_TOL:
                debug["optimizer_name"] = "lbfgs"
                return x_flat, dual, residual, steps, debug
            fallback_x, fallback_dual, fallback_residual, fallback_steps, fallback_debug = self._optimize_dual_adam(
                coeffs_flat, dual.detach().clone(), min(ADAM_FALLBACK_LR, self.optimizer_lr)
            )
            if fallback_x is None or fallback_residual >= residual:
                gd_x, gd_dual, gd_residual, gd_steps, gd_debug = self._optimize_dual_gd(
                    coeffs_flat, dual0, GD_FALLBACK_LR
                )
                if gd_residual < min(residual, fallback_residual):
                    gd_debug["optimizer_name"] = "gd_fallback"
                    gd_debug["previous_debug"] = {"lbfgs": debug, "adam": fallback_debug}
                    return gd_x, gd_dual, gd_residual, gd_steps, gd_debug
            if fallback_residual < residual:
                fallback_debug["optimizer_name"] = "adam_fallback"
                fallback_debug["previous_debug"] = debug
                return fallback_x, fallback_dual, fallback_residual, fallback_steps, fallback_debug
            debug["optimizer_name"] = "lbfgs"
            return x_flat, dual, residual, steps, debug
        if name == "adam":
            x_flat, dual, residual, steps, debug = self._optimize_dual_adam(
                coeffs_flat, dual0, self.optimizer_lr
            )
            if x_flat is not None and residual <= FLOW_RESIDUAL_TOL:
                debug["optimizer_name"] = "adam"
                return x_flat, dual, residual, steps, debug
            fallback_x, fallback_dual, fallback_residual, fallback_steps, fallback_debug = self._optimize_dual_adam(
                coeffs_flat, torch.zeros_like(dual0), ADAM_FALLBACK_LR
            )
            if fallback_x is None or fallback_residual >= residual:
                gd_x, gd_dual, gd_residual, gd_steps, gd_debug = self._optimize_dual_gd(
                    coeffs_flat, dual0, GD_FALLBACK_LR
                )
                if gd_residual < min(residual, fallback_residual):
                    gd_debug["optimizer_name"] = "gd_fallback"
                    gd_debug["previous_debug"] = {"adam": debug, "adam_fresh": fallback_debug}
                    return gd_x, gd_dual, gd_residual, gd_steps, gd_debug
            if fallback_residual < residual:
                fallback_debug["optimizer_name"] = "adam_fresh_fallback"
                fallback_debug["previous_debug"] = debug
                return fallback_x, fallback_dual, fallback_residual, fallback_steps, fallback_debug
            debug["optimizer_name"] = "adam"
            return x_flat, dual, residual, steps, debug
        if name == "gd":
            x_flat, dual, residual, steps, debug = self._optimize_dual_gd(coeffs_flat, dual0, self.optimizer_lr)
            debug["optimizer_name"] = "gd"
            return x_flat, dual, residual, steps, debug
        raise ValueError(f"Unknown optimizer_name={self.optimizer_name!r}")

    def _solve_x_torch(self, coeffs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        coeffs_flat = self._coeffs_to_tensor(coeffs)
        initializations = [self._initial_dual(coeffs_flat)]
        if self._dual_init.numel():
            initializations.append(self._dual_init @ coeffs_flat)
        initializations.append(torch.zeros_like(self._dual_warm_start))

        best_x_flat: torch.Tensor | None = None
        best_dual: torch.Tensor | None = None
        best_residual = float("inf")
        best_steps = 0
        best_debug: dict[str, Any] = {}
        candidate_debugs: list[dict[str, Any]] = []
        for dual0 in initializations:
            x_flat, dual, residual, steps, debug = self._solve_from_initial_dual(coeffs_flat, dual0)
            debug = dict(debug)
            debug["residual"] = residual
            debug["steps"] = steps
            candidate_debugs.append(debug)
            if x_flat is not None and residual < best_residual:
                best_x_flat = x_flat
                best_dual = dual
                best_residual = residual
                best_steps = steps
                best_debug = debug
            if residual <= FLOW_RESIDUAL_TOL:
                break

        if best_x_flat is None or best_dual is None:
            self.last_solve_info = {
                "device": str(self.device),
                "optimizer_name": self.optimizer_name,
                "best_residual": None,
                "optimizer_steps": None,
                "candidate_debugs": candidate_debugs,
                "status": "failed",
            }
            raise RuntimeError("Torch dual gradient solver failed to produce a candidate solution.")

        self._dual_warm_start = best_dual.clone()
        self.last_solve_info = {
            "device": str(self.device),
            "optimizer_name": best_debug.get("optimizer_name", self.optimizer_name),
            "best_residual": best_residual,
            "optimizer_steps": best_steps,
            "objective_value": best_debug.get("objective_value"),
            "status": "ok" if best_residual <= FLOW_RESIDUAL_TOL else "inaccurate",
            "candidate_debugs": candidate_debugs,
        }
        return best_x_flat.reshape(self.num_states, self.num_actions), best_dual

    def solve_x(self, coeffs: np.ndarray) -> np.ndarray:
        x_t, _ = self._solve_x_torch(coeffs)
        return x_t.detach().cpu().numpy()

    def solve_types(self, types: Any, ama: AMAParams) -> tuple[np.ndarray, np.ndarray]:
        coeffs = _objective_coeffs(self.lp, types, ama)
        return self.solve_x(coeffs), coeffs

    def reverse_objective_gradient(self, coeffs: np.ndarray, grad_wrt_x: np.ndarray) -> np.ndarray:
        x_t, _ = self._solve_x_torch(coeffs)
        x_flat = x_t.reshape(-1)
        grad_flat = torch.as_tensor(
            np.asarray(grad_wrt_x, dtype=float).reshape(-1),
            dtype=self.dtype,
            device=self.device,
        )
        weighted_grad = x_flat * grad_flat
        weighted_flow = self._flow_matrix * x_flat.unsqueeze(0)
        system_matrix = weighted_flow @ self._flow_matrix_T + LINEAR_SOLVE_EPS * self._identity
        rhs = self._flow_matrix @ weighted_grad
        if not torch.isfinite(system_matrix).all() or not torch.isfinite(rhs).all():
            raise RuntimeError("Torch reverse_objective_gradient encountered non-finite linear system.")
        correction = torch.linalg.solve(system_matrix, rhs)
        grad_coeff = (weighted_grad - x_flat * (self._flow_matrix_T @ correction)) / self.alpha
        if not torch.isfinite(grad_coeff).all():
            raise RuntimeError("Torch reverse_objective_gradient produced non-finite coefficients.")
        return grad_coeff.reshape(self.num_states, self.num_actions).detach().cpu().numpy()


DifferentiableMDPLinearProgram = TorchDifferentiableMDPLinearProgram


def _calcrevenue_diff(
    lp: MDPLinearProgram,
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> float:
    samples = lp.mdp.sampletypes(num_samples)
    revenues = [_calcrevenue_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return float(np.mean(revenues))


def _expectedrevenue_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: TorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    samples = lp.mdp.sampletypes(num_samples)
    revenues = [_calcrevenue_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return _mean_and_std(revenues)


def _calcmakespan_diff(
    lp: MDPLinearProgram,
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> float:
    samples = lp.mdp.sampletypes(num_samples)
    makespans = [_calcmakespan_diff(lp, diff_lp, types, ama, alpha) for types in samples]
    return float(np.mean(makespans))


def _expectedperformance_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    diff_lp: TorchDifferentiableMDPLinearProgram,
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
    device: str | torch.device | None = None,
    optimizer_name: str = "lbfgs",
    optimizer_lr: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    lp = MDPLinearProgram(mdp)
    diff_lp = TorchDifferentiableMDPLinearProgram(
        lp,
        alpha=reg_strength,
        device=device,
        optimizer_name=optimizer_name,
        optimizer_lr=optimizer_lr,
    )

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
        "method": "reglp_torch",
        "backend": "torch",
        "device": str(diff_lp.device),
        "optimizer_name": optimizer_name,
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
        "last_solver_residual": diff_lp.last_solve_info.get("best_residual"),
        "last_solver_steps": diff_lp.last_solve_info.get("optimizer_steps"),
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
