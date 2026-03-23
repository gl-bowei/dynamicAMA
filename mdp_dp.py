from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import torch

from auction import AuctionMDP
from lp_and_ama import AMAParams
from sequential_auction_dp import SequentialAuctionDPSolver, calcrevenue_dp


TEST_SAMPLES = 10000
MIN_WEIGHT = 1e-6


def _project_positive_weights(weights: torch.Tensor, min_weight: float = MIN_WEIGHT) -> torch.Tensor:
    return torch.clamp(weights, min=min_weight)


@dataclass
class DifferentiableSequentialAuctionDP:
    mdp: AuctionMDP
    alpha: float = 0.01
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.mdp, AuctionMDP):
            raise TypeError("DifferentiableSequentialAuctionDP only supports AuctionMDP.")
        if self.alpha <= 0.0:
            raise ValueError("DifferentiableSequentialAuctionDP requires alpha > 0.")
        if not np.isclose(self.mdp.gamma, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("DifferentiableSequentialAuctionDP currently assumes gamma=1.")

        self.device_obj = torch.device(self.device)
        self.num_states = len(self.mdp.state_list)
        self.num_actions = len(self.mdp.action_list)
        self.num_agents = self.mdp.n_agents

        self.state_to_index = {state: idx for idx, state in enumerate(self.mdp.state_list)}
        self.states_by_depth = self._build_states_by_depth()
        self.child_index = self._build_child_index()
        self.reward_mask = self._build_reward_mask()

    def _build_states_by_depth(self) -> list[list[int]]:
        states_by_depth: list[list[int]] = [[] for _ in range(self.mdp.n_items)]
        for state_idx, state in enumerate(self.mdp.state_list):
            states_by_depth[int(state[0])].append(state_idx)
        return states_by_depth

    def _next_state(self, state: tuple[int, ...], action: int) -> tuple[int, ...] | None:
        depth = int(state[0])
        if depth >= self.mdp.n_items - 1:
            return None
        new_state = list(state)
        new_state[0] = depth + 1
        new_state[depth + 1] = action
        return tuple(new_state)

    def _build_child_index(self) -> np.ndarray:
        child_index = np.full((self.num_states, self.num_actions), -1, dtype=int)
        for state_idx, state in enumerate(self.mdp.state_list):
            for action_idx, action in enumerate(self.mdp.action_list):
                next_state = self._next_state(state, action)
                if next_state is not None:
                    child_index[state_idx, action_idx] = self.state_to_index[next_state]
        return child_index

    def _build_reward_mask(self) -> torch.Tensor:
        reward_mask = np.zeros((self.num_states, self.num_actions, self.num_agents), dtype=float)
        ones = np.ones(self.num_agents, dtype=float)
        for state_idx, state in enumerate(self.mdp.state_list):
            for action_idx, action in enumerate(self.mdp.action_list):
                reward_mask[state_idx, action_idx, :] = np.asarray(
                    self.mdp.reward_from_alloc(state, action, ones), dtype=float
                )
        return torch.tensor(reward_mask, dtype=self.dtype, device=self.device_obj)

    def _coeffs(
        self,
        types: torch.Tensor,
        weights: torch.Tensor,
        boosts: torch.Tensor,
    ) -> torch.Tensor:
        weighted_types = types * weights
        return torch.sum(self.reward_mask * weighted_types.view(1, 1, -1), dim=-1) + boosts

    def solve_torch(
        self,
        types: torch.Tensor,
        weights: torch.Tensor,
        boosts: torch.Tensor,
    ) -> torch.Tensor:
        coeffs = self._coeffs(types, weights, boosts)
        zero_scalar = torch.zeros((), dtype=self.dtype, device=self.device_obj)
        zero_row = torch.zeros(self.num_actions, dtype=self.dtype, device=self.device_obj)

        state_values: list[torch.Tensor] = [zero_scalar for _ in range(self.num_states)]
        policy_rows: list[torch.Tensor] = [zero_row for _ in range(self.num_states)]

        for depth in range(self.mdp.n_items - 1, -1, -1):
            beta = self.alpha * float(self.mdp.n_items - depth)
            for state_idx in self.states_by_depth[depth]:
                child_bonus = []
                for child_idx in self.child_index[state_idx]:
                    child_bonus.append(state_values[child_idx] if child_idx >= 0 else zero_scalar)
                q_values = coeffs[state_idx] + torch.stack(child_bonus)
                logits = q_values / beta
                policy = torch.softmax(logits, dim=0)
                value = beta * torch.logsumexp(logits, dim=0)
                policy_rows[state_idx] = policy
                state_values[state_idx] = value

        state_occ: list[torch.Tensor] = [zero_scalar for _ in range(self.num_states)]
        x_rows: list[torch.Tensor] = [zero_row for _ in range(self.num_states)]
        root_idx = self.state_to_index[self.mdp.startstate()]
        state_occ[root_idx] = torch.ones((), dtype=self.dtype, device=self.device_obj)

        for depth in range(self.mdp.n_items):
            for state_idx in self.states_by_depth[depth]:
                occ = state_occ[state_idx]
                row = occ * policy_rows[state_idx]
                x_rows[state_idx] = row
                for action_idx, child_idx in enumerate(self.child_index[state_idx]):
                    if child_idx >= 0:
                        state_occ[child_idx] = state_occ[child_idx] + row[action_idx]

        return torch.stack(x_rows, dim=0)

    def asw(self, x: torch.Tensor, types: torch.Tensor, weights: torch.Tensor, boosts: torch.Tensor) -> torch.Tensor:
        coeffs = self._coeffs(types, weights, boosts)
        return torch.sum(coeffs * x)

    def sw(self, x: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        reward_values = torch.sum(self.reward_mask * types.view(1, 1, -1), dim=-1)
        return torch.sum(reward_values * x)

    def revenue(self, types: torch.Tensor, weights: torch.Tensor, boosts: torch.Tensor) -> torch.Tensor:
        main_x = self.solve_torch(types, weights, boosts)
        main_asw = self.asw(main_x, types, weights, boosts)
        revenue = self.sw(main_x, types)

        for agent_idx in range(self.num_agents):
            cf_types = types.clone()
            cf_types[agent_idx] = 0.0
            cf_x = self.solve_torch(cf_types, weights, boosts)
            cf_asw = self.asw(cf_x, cf_types, weights, boosts)
            revenue = revenue + (cf_asw - main_asw) / weights[agent_idx]
        return revenue


def _expectedrevenue_dp_autograd(
    diff_dp: DifferentiableSequentialAuctionDP,
    ama_weights: torch.Tensor,
    ama_boosts: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    sample_list = diff_dp.mdp.sampletypes(num_samples)
    total = torch.zeros((), dtype=diff_dp.dtype, device=diff_dp.device_obj)
    for sample in sample_list:
        types_t = torch.tensor(np.asarray(sample, dtype=float), dtype=diff_dp.dtype, device=diff_dp.device_obj)
        total = total + diff_dp.revenue(types_t, ama_weights, ama_boosts)
    return total / float(num_samples)


def _expectedrevenue_dp_stats(
    solver: SequentialAuctionDPSolver,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    revenues = [calcrevenue_dp(solver, types, ama, alpha) for types in solver.mdp.sampletypes(num_samples)]
    mean = float(np.mean(revenues))
    std = float(np.std(revenues, ddof=1)) if len(revenues) > 1 else 0.0
    return mean, std


def optimize_boosts(
    diff_dp: DifferentiableSequentialAuctionDP,
    ama: AMAParams,
    num_samples: int = 10,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    boosts = torch.tensor(ama.boosts, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    weights = torch.tensor(ama.weights, dtype=diff_dp.dtype, device=diff_dp.device_obj)
    vals = np.zeros(num_iters, dtype=float)

    for i in range(num_iters):
        objective = _expectedrevenue_dp_autograd(diff_dp, weights, boosts, num_samples=num_samples)
        vals[i] = float(objective.item())
        grad_boosts = torch.autograd.grad(objective, boosts)[0]
        with torch.no_grad():
            boosts += lr * grad_boosts

    return AMAParams(weights.detach().cpu().numpy(), boosts.detach().cpu().numpy()), vals


def optimize_weights_and_boosts(
    diff_dp: DifferentiableSequentialAuctionDP,
    ama: AMAParams,
    num_samples: int = 10,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    boosts = torch.tensor(ama.boosts, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    weights = torch.tensor(ama.weights, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    vals = np.zeros(num_iters, dtype=float)

    for i in range(num_iters):
        objective = _expectedrevenue_dp_autograd(diff_dp, weights, boosts, num_samples=num_samples)
        vals[i] = float(objective.item())
        grad_weights, grad_boosts = torch.autograd.grad(objective, (weights, boosts))
        with torch.no_grad():
            boosts += lr * grad_boosts
            weights += lr * grad_weights
            weights.copy_(_project_positive_weights(weights))

    return AMAParams(weights.detach().cpu().numpy(), boosts.detach().cpu().numpy()), vals


def optimize_weights_only(
    diff_dp: DifferentiableSequentialAuctionDP,
    ama: AMAParams,
    num_samples: int = 10,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    boosts = torch.tensor(ama.boosts, dtype=diff_dp.dtype, device=diff_dp.device_obj)
    weights = torch.tensor(ama.weights, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    vals = np.zeros(num_iters, dtype=float)

    for i in range(num_iters):
        objective = _expectedrevenue_dp_autograd(diff_dp, weights, boosts, num_samples=num_samples)
        vals[i] = float(objective.item())
        grad_weights = torch.autograd.grad(objective, weights)[0]
        with torch.no_grad():
            weights += lr * grad_weights
            weights.copy_(_project_positive_weights(weights))

    return AMAParams(weights.detach().cpu().numpy(), boosts.detach().cpu().numpy()), vals


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
    optimize_mode: str | None = None,
    gamma: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mdp_factory is not AuctionMDP:
        raise TypeError("mdp_dp.runtrial currently only supports AuctionMDP.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    diff_dp = DifferentiableSequentialAuctionDP(mdp, alpha=reg_strength)
    eval_dp = SequentialAuctionDPSolver(mdp)

    if optimize_mode is None:
        optimize_mode = "weights_and_boosts" if optimize_weights else "boosts_only"

    if optimize_mode not in {"boosts_only", "weights_and_boosts", "weights_only"}:
        raise ValueError("optimize_mode must be one of boosts_only, weights_and_boosts, weights_only")

    if optimize_mode == "weights_only":
        boosts = np.zeros((len(mdp.state_list), len(mdp.action_list)), dtype=float)
    else:
        boosts = np.random.rand(len(mdp.state_list), len(mdp.action_list))
    ama = AMAParams(np.ones(mdp.n_agents, dtype=float), boosts.copy())
    vcg_ama = AMAParams(np.ones(mdp.n_agents, dtype=float), np.zeros_like(boosts))

    start_time = datetime.now()
    if optimize_mode == "weights_only":
        ama, vals = optimize_weights_only(
            diff_dp,
            ama,
            num_samples=num_samples,
            lr=lr,
            num_iters=num_training_iters,
        )
    elif optimize_mode == "weights_and_boosts":
        ama, vals = optimize_weights_and_boosts(
            diff_dp,
            ama,
            num_samples=num_samples,
            lr=lr,
            num_iters=num_training_iters,
        )
    else:
        ama, vals = optimize_boosts(
            diff_dp,
            ama,
            num_samples=num_samples,
            lr=lr,
            num_iters=num_training_iters,
        )
    end_time = datetime.now()

    eval_alpha = 0.0
    vcg_revenue, vcg_std = _expectedrevenue_dp_stats(eval_dp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)
    ama_revenue, ama_std = _expectedrevenue_dp_stats(eval_dp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)

    result = {
        "method": "regdp" if optimize_mode != "weights_only" else "regdp_weights_only",
        "mdp": getattr(mdp_factory, "__name__", str(mdp_factory)),
        "dist": dist_type,
        "optimize_mode": optimize_mode,
        "num_agents": num_agents,
        "num_items": num_items,
        "seed": seed,
        "num_samples": num_samples,
        "test_samples": TEST_SAMPLES,
        "training_iters": num_training_iters,
        "lr": lr,
        "reg_strength": reg_strength,
        "eval_alpha": eval_alpha,
        "vcg_performance": vcg_revenue,
        "vcg_performance_std": vcg_std,
        "ama_performance": ama_revenue,
        "ama_performance_std": ama_std,
        "vcg_revenue": vcg_revenue,
        "vcg_std": vcg_std,
        "ama_revenue": ama_revenue,
        "ama_std": ama_std,
        "runtime": end_time - start_time,
    }
    return result, {"ama": ama, "vals": vals}
