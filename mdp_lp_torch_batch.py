from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy as np
import torch

from lp_and_ama import AMAParams, MDPLinearProgram, asw, dsw_dx, sw
import mdp_lp_torch


TEST_SAMPLES = 10000
MIN_WEIGHT = 1e-6
LOG_X_CLIP = 80.0
LINEAR_SOLVE_EPS = 1e-9
FLOW_RESIDUAL_TOL = 1e-6
DEFAULT_MAX_ITERS = 250
DUAL_CLIP = 1e4
MIN_EFFECTIVE_X = 1e-300
GD_STEP_LR = 1.0


def _objective_coeffs(lp: MDPLinearProgram, types: Any, ama: AMAParams) -> np.ndarray:
    coeffs = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            reward = np.asarray(lp.mdp.reward_from_alloc(state, action, types), dtype=float)
            coeffs[state_idx, action_idx] = float(reward @ ama.weights) + float(
                ama.boosts[state_idx, action_idx]
            )
    return coeffs


def _objective_coeffs_batch(lp: MDPLinearProgram, types_batch: Sequence[Any], ama: AMAParams) -> np.ndarray:
    coeffs = np.zeros(
        (len(types_batch), len(lp.mdp.state_list), len(lp.mdp.action_list)),
        dtype=float,
    )
    for batch_idx, types in enumerate(types_batch):
        coeffs[batch_idx] = _objective_coeffs(lp, types, ama)
    return coeffs


def _reward_matrix(lp: MDPLinearProgram, types: Any) -> np.ndarray:
    rewards = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list), lp.mdp.n_agents), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            rewards[state_idx, action_idx, :] = np.asarray(
                lp.mdp.reward_from_alloc(state, action, types), dtype=float
            )
    return rewards


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _validate_diff_alpha(diff_lp: "BatchedTorchDifferentiableMDPLinearProgram", alpha: float) -> None:
    if alpha <= 0.0:
        raise ValueError("Differentiable LP gradients require alpha > 0.")
    if not np.isclose(diff_lp.alpha, alpha, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"alpha mismatch: diff_lp.alpha={diff_lp.alpha} but function was called with alpha={alpha}."
        )


@dataclass
class BatchedTorchDifferentiableMDPLinearProgram:
    lp: MDPLinearProgram
    alpha: float = 0.01
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float64
    max_iters: int = DEFAULT_MAX_ITERS
    grad_tolerance: float = 1e-12
    dual_clip: float = DUAL_CLIP
    step_lr: float = GD_STEP_LR

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("BatchedTorchDifferentiableMDPLinearProgram requires alpha > 0.")
        self.device = self._resolve_device(self.device)
        self.num_states = len(self.lp.mdp.state_list)
        self.num_actions = len(self.lp.mdp.action_list)
        self.num_variables = self.num_states * self.num_actions
        flow_matrix_np, rhs_np = self._build_flow_matrix()
        self._flow_matrix = torch.as_tensor(flow_matrix_np, dtype=self.dtype, device=self.device)
        self._flow_matrix_T = self._flow_matrix.transpose(0, 1).contiguous()
        self._rhs = torch.as_tensor(rhs_np, dtype=self.dtype, device=self.device)
        self._identity = torch.eye(self._flow_matrix.shape[0], dtype=self.dtype, device=self.device)
        self._dual_warm_start: dict[int, torch.Tensor] = {}
        self.last_solve_info: dict[str, Any] = {}
        self._single_solver = mdp_lp_torch.TorchDifferentiableMDPLinearProgram(
            self.lp,
            alpha=self.alpha,
            device=self.device,
            optimizer_name="lbfgs",
            optimizer_lr=1.0,
            max_iters=self.max_iters,
        )

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

    def _clip_dual_(self, dual: torch.Tensor) -> None:
        if self.dual_clip > 0.0:
            dual.clamp_(min=-self.dual_clip, max=self.dual_clip)

    def _coeffs_to_tensor(self, coeffs_batch: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(coeffs_batch, dtype=float).reshape(len(coeffs_batch), -1),
            dtype=self.dtype,
            device=self.device,
        )

    def _initial_dual(self, batch_size: int) -> torch.Tensor:
        warm = self._dual_warm_start.get(batch_size)
        if warm is not None:
            return warm.clone()
        return torch.zeros((batch_size, self._flow_matrix.shape[0]), dtype=self.dtype, device=self.device)

    def _primal_from_dual(
        self, dual_batch: torch.Tensor, coeffs_flat_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduced_cost = coeffs_flat_batch - dual_batch @ self._flow_matrix
        log_x = torch.clamp((reduced_cost / self.alpha) - 1.0, min=-LOG_X_CLIP, max=LOG_X_CLIP)
        x_flat = torch.exp(log_x)
        x_flat = torch.clamp(x_flat, min=MIN_EFFECTIVE_X)
        return x_flat, log_x

    def _dual_gradient(self, dual_batch: torch.Tensor, coeffs_flat_batch: torch.Tensor) -> torch.Tensor:
        x_flat, _ = self._primal_from_dual(dual_batch, coeffs_flat_batch)
        flow = x_flat @ self._flow_matrix_T
        return self._rhs.unsqueeze(0) - flow

    def _evaluate_candidates(
        self, dual_batch: torch.Tensor, coeffs_flat_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dual_eval = dual_batch.detach().clone()
        self._clip_dual_(dual_eval)
        x_flat, _ = self._primal_from_dual(dual_eval, coeffs_flat_batch)
        flow = x_flat @ self._flow_matrix_T
        residual = torch.max(torch.abs(self._rhs.unsqueeze(0) - flow), dim=1).values
        objective = torch.sum(dual_eval * self._rhs.unsqueeze(0), dim=1) + self.alpha * torch.sum(x_flat, dim=1)
        finite = (
            torch.isfinite(x_flat).all(dim=1)
            & torch.isfinite(flow).all(dim=1)
            & torch.isfinite(residual)
            & torch.isfinite(objective)
        )
        return x_flat, residual, objective, finite

    def solve_coeffs_batch(self, coeffs_batch: np.ndarray) -> np.ndarray:
        coeffs_flat_batch = self._coeffs_to_tensor(coeffs_batch)
        batch_size = coeffs_flat_batch.shape[0]
        dual = self._initial_dual(batch_size)
        best_dual = dual.clone()
        best_x_flat, best_residual, best_objective, finite = self._evaluate_candidates(dual, coeffs_flat_batch)
        best_residual = torch.where(finite, best_residual, torch.full_like(best_residual, float("inf")))
        active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        step_lr = self.step_lr
        steps_taken = 0

        for step in range(self.max_iters):
            _, current_residual, current_objective, current_finite = self._evaluate_candidates(dual, coeffs_flat_batch)
            grad = self._dual_gradient(dual, coeffs_flat_batch)
            grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
            grad_inf = torch.max(torch.abs(grad), dim=1, keepdim=True).values
            grad_scale = torch.clamp(grad_inf, min=1.0)
            direction = grad / grad_scale
            grad_norm = torch.linalg.norm(grad, dim=1)
            active = active & current_finite & (grad_norm > self.grad_tolerance)
            if not torch.any(active):
                steps_taken = step + 1
                break

            trial_lr = step_lr
            accepted_any = False
            for _ in range(20):
                candidate_dual = dual.clone()
                candidate_dual[active] = dual[active] - trial_lr * direction[active]
                self._clip_dual_(candidate_dual)
                cand_x, cand_residual, cand_objective, cand_finite = self._evaluate_candidates(
                    candidate_dual, coeffs_flat_batch
                )
                improve = active & cand_finite & (cand_objective <= current_objective + 1e-12)
                if torch.any(improve):
                    dual[improve] = candidate_dual[improve]
                    better = improve & (cand_residual < best_residual)
                    if torch.any(better):
                        best_dual[better] = candidate_dual[better]
                        best_x_flat[better] = cand_x[better]
                        best_residual[better] = cand_residual[better]
                        best_objective[better] = cand_objective[better]
                    active = active & ~(best_residual <= FLOW_RESIDUAL_TOL)
                    accepted_any = True
                    break
                trial_lr *= 0.5
                if trial_lr < 1e-8:
                    break
            if not accepted_any:
                steps_taken = step + 1
                break
            step_lr = min(trial_lr * 1.1, self.step_lr)
            steps_taken = step + 1

        fallback_count = 0
        bad_mask = best_residual > FLOW_RESIDUAL_TOL
        if torch.any(bad_mask):
            bad_indices = torch.nonzero(bad_mask, as_tuple=False).flatten().tolist()
            for batch_idx in bad_indices:
                x_single = self._single_solver.solve_x(coeffs_batch[batch_idx])
                residual_single = self._single_solver.last_solve_info.get("best_residual", float("inf"))
                if np.isfinite(residual_single) and residual_single < float(best_residual[batch_idx].item()):
                    best_x_flat[batch_idx] = torch.as_tensor(
                        x_single.reshape(-1), dtype=self.dtype, device=self.device
                    )
                    best_residual[batch_idx] = residual_single
                    fallback_count += 1

        self._dual_warm_start[batch_size] = best_dual.detach().clone()
        self.last_solve_info = {
            "device": str(self.device),
            "best_residual": float(torch.max(best_residual).item()),
            "mean_residual": float(torch.mean(best_residual).item()),
            "optimizer_steps": steps_taken,
            "batch_size": batch_size,
            "fallback_count": fallback_count,
            "bad_after_fallback": int(torch.sum(best_residual > FLOW_RESIDUAL_TOL).item()),
        }
        return best_x_flat.reshape(batch_size, self.num_states, self.num_actions).detach().cpu().numpy()

    def solve_types(self, types: Any, ama: AMAParams) -> tuple[np.ndarray, np.ndarray]:
        coeffs = _objective_coeffs(self.lp, types, ama)
        x = self.solve_coeffs_batch(coeffs[np.newaxis, :, :])[0]
        return x, coeffs

    def solve_types_batch(self, types_batch: Sequence[Any], ama: AMAParams) -> tuple[np.ndarray, np.ndarray]:
        coeffs_batch = _objective_coeffs_batch(self.lp, types_batch, ama)
        x_batch = self.solve_coeffs_batch(coeffs_batch)
        return x_batch, coeffs_batch

    def reverse_objective_gradient_batch(
        self, coeffs_batch: np.ndarray, grad_wrt_x_batch: np.ndarray
    ) -> np.ndarray:
        x_batch = self.solve_coeffs_batch(coeffs_batch)
        x_flat = torch.as_tensor(x_batch.reshape(len(x_batch), -1), dtype=self.dtype, device=self.device)
        grad_flat = torch.as_tensor(
            np.asarray(grad_wrt_x_batch, dtype=float).reshape(len(coeffs_batch), -1),
            dtype=self.dtype,
            device=self.device,
        )
        weighted_grad = x_flat * grad_flat
        weighted_flow = self._flow_matrix.unsqueeze(0) * x_flat.unsqueeze(1)
        system_matrix = weighted_flow @ self._flow_matrix_T.unsqueeze(0) + LINEAR_SOLVE_EPS * self._identity.unsqueeze(0)
        rhs = weighted_grad @ self._flow_matrix_T
        correction = torch.linalg.solve(system_matrix, rhs.unsqueeze(-1)).squeeze(-1)
        grad_coeff = (weighted_grad - x_flat * (correction @ self._flow_matrix)) / self.alpha
        return grad_coeff.reshape(len(coeffs_batch), self.num_states, self.num_actions).detach().cpu().numpy()

    def reverse_objective_gradient(self, coeffs: np.ndarray, grad_wrt_x: np.ndarray) -> np.ndarray:
        return self.reverse_objective_gradient_batch(coeffs[np.newaxis, :, :], grad_wrt_x[np.newaxis, :, :])[0]


DifferentiableMDPLinearProgram = BatchedTorchDifferentiableMDPLinearProgram


def _build_revenue_problem_batch(lp: MDPLinearProgram, types_list: Sequence[Any]) -> tuple[list[Any], list[tuple[int, int]]]:
    expanded: list[Any] = []
    mapping: list[tuple[int, int]] = []
    for sample_idx, types in enumerate(types_list):
        expanded.append(types)
        mapping.append((sample_idx, -1))
        for agent_idx in range(lp.mdp.n_agents):
            expanded.append(lp.mdp.counterfactualtype(types, agent_idx))
            mapping.append((sample_idx, agent_idx))
    return expanded, mapping


def _calcrevenue_diff(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> float:
    _validate_diff_alpha(diff_lp, alpha)
    x_batch, _ = diff_lp.solve_types_batch(
        [types] + [lp.mdp.counterfactualtype(types, i) for i in range(lp.mdp.n_agents)],
        ama,
    )
    main_x = x_batch[0]
    main_asw = asw(lp.mdp, main_x, types, ama)
    revenue = sw(lp.mdp, main_x, types)
    for i in range(lp.mdp.n_agents):
        cf_types = lp.mdp.counterfactualtype(types, i)
        cf_x = x_batch[i + 1]
        revenue += (asw(lp.mdp, cf_x, cf_types, ama) - main_asw) / ama.weights[i]
    return revenue


def _expectedrevenue_diff(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> float:
    samples = lp.mdp.sampletypes(num_samples)
    revenues = _expectedrevenue_diff_values(lp, diff_lp, samples, ama, alpha)
    return float(np.mean(revenues))


def _expectedrevenue_diff_values(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    samples: Sequence[Any],
    ama: AMAParams,
    alpha: float,
) -> list[float]:
    _validate_diff_alpha(diff_lp, alpha)
    expanded_types, mapping = _build_revenue_problem_batch(lp, samples)
    x_batch, _ = diff_lp.solve_types_batch(expanded_types, ama)
    main_x = {}
    main_asw = {}
    revenues = [0.0 for _ in samples]
    for batch_idx, (sample_idx, agent_idx) in enumerate(mapping):
        types = expanded_types[batch_idx]
        x = x_batch[batch_idx]
        if agent_idx == -1:
            main_x[sample_idx] = x
            main_asw[sample_idx] = asw(lp.mdp, x, types, ama)
            revenues[sample_idx] = sw(lp.mdp, x, types)
        else:
            revenues[sample_idx] += (asw(lp.mdp, x, types, ama) - main_asw[sample_idx]) / ama.weights[agent_idx]
    return revenues


def _expectedrevenue_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    revenues = _expectedrevenue_diff_values(lp, diff_lp, lp.mdp.sampletypes(num_samples), ama, alpha)
    return _mean_and_std(revenues)


def _expectedperformance_diff_stats(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    return _expectedrevenue_diff_stats(lp, diff_lp, ama, num_samples, alpha)


def revenuegradb_asw_envelope(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> np.ndarray:
    _validate_diff_alpha(diff_lp, alpha)
    type_batch = [types] + [lp.mdp.counterfactualtype(types, i) for i in range(lp.mdp.n_agents)]
    x_batch, coeffs_batch = diff_lp.solve_types_batch(type_batch, ama)
    main_x = x_batch[0]
    grad_sw = dsw_dx(lp.mdp, main_x, types)
    grad_x_batch = np.zeros_like(x_batch)
    grad_x_batch[0] = grad_sw
    rev_grad_batch = diff_lp.reverse_objective_gradient_batch(coeffs_batch[:1], grad_x_batch[:1])[0]

    rev_grad_b = rev_grad_batch
    for i in range(lp.mdp.n_agents):
        rev_grad_b += (x_batch[i + 1] - main_x) / ama.weights[i]
    return rev_grad_b


def expectedrevenuegrad(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
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


def optimize_boosts(
    lp: MDPLinearProgram,
    diff_lp: BatchedTorchDifferentiableMDPLinearProgram,
    ama: AMAParams,
    objective_kind: str,
    num_samples: int = 10,
    alpha: float = 0.01,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    if objective_kind != "revenue":
        raise NotImplementedError("The batched torch backend currently supports revenue objectives only.")
    _validate_diff_alpha(diff_lp, alpha)
    vals = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        vals[i] = _expectedrevenue_diff(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
        grad = expectedrevenuegrad(lp, diff_lp, ama, num_samples=num_samples, alpha=alpha)
        ama.boosts += lr * grad
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    if optimize_weights:
        raise NotImplementedError("The batched torch backend currently supports optimize_weights=False only.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    lp = MDPLinearProgram(mdp)
    diff_lp = BatchedTorchDifferentiableMDPLinearProgram(lp, alpha=reg_strength, device=device)
    boosts = np.random.rand(*lp.x.shape)
    ama = AMAParams(np.ones(lp.mdp.n_agents, dtype=float), boosts.copy())
    vcg_ama = AMAParams(np.ones(lp.mdp.n_agents, dtype=float), np.zeros_like(boosts))

    if hasattr(mdp, "makespan_from_sa"):
        raise NotImplementedError("The batched torch backend currently supports auction-style revenue only.")

    start_time = datetime.now()
    ama, vals = optimize_boosts(
        lp,
        diff_lp,
        ama,
        objective_kind="revenue",
        num_samples=num_samples,
        alpha=reg_strength,
        lr=lr,
        num_iters=num_training_iters,
    )
    end_time = datetime.now()

    eval_alpha = reg_strength
    vcg_revenue, vcg_std = _expectedrevenue_diff_stats(lp, diff_lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)
    vcg_performance, vcg_performance_std = _expectedperformance_diff_stats(
        lp, diff_lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )
    ama_revenue, ama_std = _expectedrevenue_diff_stats(lp, diff_lp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)
    ama_performance, ama_performance_std = _expectedperformance_diff_stats(
        lp, diff_lp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha
    )

    result = {
        "method": "reglp_torch_batch",
        "backend": "torch_batch",
        "device": str(diff_lp.device),
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
        "last_solver_mean_residual": diff_lp.last_solve_info.get("mean_residual"),
        "last_solver_steps": diff_lp.last_solve_info.get("optimizer_steps"),
        "last_batch_size": diff_lp.last_solve_info.get("batch_size"),
        "last_fallback_count": diff_lp.last_solve_info.get("fallback_count"),
        "last_bad_after_fallback": diff_lp.last_solve_info.get("bad_after_fallback"),
    }
    return result, {"ama": ama, "vals": vals}


def namedtuple_to_csv_line_str(result: dict[str, Any]) -> str:
    return "\t".join(str(v) for v in result.values())
