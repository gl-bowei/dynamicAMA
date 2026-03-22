from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import torch

from lp_and_ama import AMAParams
from schedule import ScheduleMDP
from sequential_schedule_dp import SequentialScheduleDPSolver, calcmakespan_dp


TEST_SAMPLES = 10000
MIN_WEIGHT = 1e-6


def _project_positive_weights(weights: torch.Tensor, min_weight: float = MIN_WEIGHT) -> torch.Tensor:
    return torch.clamp(weights, min=min_weight)


@dataclass
class DifferentiableSequentialScheduleDP:
    mdp: ScheduleMDP
    alpha: float = 0.01
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.mdp, ScheduleMDP):
            raise TypeError("DifferentiableSequentialScheduleDP only supports ScheduleMDP.")
        if self.alpha <= 0.0:
            raise ValueError("DifferentiableSequentialScheduleDP requires alpha > 0.")
        if not np.isclose(self.mdp.gamma, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("DifferentiableSequentialScheduleDP currently assumes gamma=1.")

        self.device_obj = torch.device(self.device)
        self.num_states = len(self.mdp.state_list)
        self.num_actions = len(self.mdp.action_list)
        self.num_agents = self.mdp.n_agents

        self.state_to_index = {state: idx for idx, state in enumerate(self.mdp.state_list)}
        self.states_by_stage = self._build_states_by_stage()
        self.child_index = self._build_child_index()
        self.stage_index = self._build_stage_index()
        self.action_agent_index = torch.tensor(
            [int(action) - 1 for action in self.mdp.action_list],
            dtype=torch.long,
            device=self.device_obj,
        )

    def _build_states_by_stage(self) -> list[list[int]]:
        states_by_stage: list[list[int]] = [[] for _ in range(self.mdp.n_items)]
        for state_idx, state in enumerate(self.mdp.state_list):
            stage = int(state[0]) - 1
            states_by_stage[stage].append(state_idx)
        return states_by_stage

    def _next_state(self, state: tuple[int, ...], action: int) -> tuple[int, ...] | None:
        if int(state[0]) >= self.mdp.n_items:
            return None
        new_state = list(state)
        new_state[new_state[0]] = action
        new_state[0] += 1
        return tuple(new_state)

    def _build_child_index(self) -> np.ndarray:
        child_index = np.full((self.num_states, self.num_actions), -1, dtype=int)
        for state_idx, state in enumerate(self.mdp.state_list):
            for action_idx, action in enumerate(self.mdp.action_list):
                next_state = self._next_state(state, action)
                if next_state is not None:
                    child_index[state_idx, action_idx] = self.state_to_index[next_state]
        return child_index

    def _build_stage_index(self) -> torch.Tensor:
        return torch.tensor(
            [int(state[0]) - 1 for state in self.mdp.state_list],
            dtype=torch.long,
            device=self.device_obj,
        )

    def _coeffs(
        self,
        types: torch.Tensor,
        weights: torch.Tensor,
        boosts: torch.Tensor,
    ) -> torch.Tensor:
        task_costs = types[
            self.action_agent_index.unsqueeze(0),
            self.stage_index.unsqueeze(1),
        ]
        return -(task_costs * weights[self.action_agent_index].unsqueeze(0)) + boosts

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

        for stage in range(self.mdp.n_items - 1, -1, -1):
            beta = self.alpha * float(self.mdp.n_items - stage)
            for state_idx in self.states_by_stage[stage]:
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

        for stage in range(self.mdp.n_items):
            for state_idx in self.states_by_stage[stage]:
                occ = state_occ[state_idx]
                row = occ * policy_rows[state_idx]
                x_rows[state_idx] = row
                for action_idx, child_idx in enumerate(self.child_index[state_idx]):
                    if child_idx >= 0:
                        state_occ[child_idx] = state_occ[child_idx] + row[action_idx]

        return torch.stack(x_rows, dim=0)

    def makespan(self, x: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        makespan_values = torch.zeros((self.num_states, self.num_actions), dtype=self.dtype, device=self.device_obj)
        types_np = types.detach().cpu().numpy()
        for state_idx, state in enumerate(self.mdp.state_list):
            for action_idx, action in enumerate(self.mdp.action_list):
                makespan_values[state_idx, action_idx] = float(self.mdp.makespan_from_sa(state, action, types_np))
        return torch.sum(makespan_values * x)


def _expected_neg_makespan_autograd(
    diff_dp: DifferentiableSequentialScheduleDP,
    ama_weights: torch.Tensor,
    ama_boosts: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    sample_list = diff_dp.mdp.sampletypes(num_samples)
    total = torch.zeros((), dtype=diff_dp.dtype, device=diff_dp.device_obj)
    for sample in sample_list:
        types_t = torch.tensor(np.asarray(sample, dtype=float), dtype=diff_dp.dtype, device=diff_dp.device_obj)
        x = diff_dp.solve_torch(types_t, ama_weights, ama_boosts)
        total = total - diff_dp.makespan(x, types_t)
    return total / float(num_samples)


def _expectedmakespan_dp_stats(
    solver: SequentialScheduleDPSolver,
    ama: AMAParams,
    num_samples: int,
    alpha: float,
) -> tuple[float, float]:
    makespans = [calcmakespan_dp(solver, types, ama, alpha) for types in solver.mdp.sampletypes(num_samples)]
    mean = float(np.mean(makespans))
    std = float(np.std(makespans, ddof=1)) if len(makespans) > 1 else 0.0
    return mean, std


def optimize_boosts(
    diff_dp: DifferentiableSequentialScheduleDP,
    ama: AMAParams,
    num_samples: int = 10,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    boosts = torch.tensor(ama.boosts, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    weights = torch.tensor(ama.weights, dtype=diff_dp.dtype, device=diff_dp.device_obj)
    vals = np.zeros(num_iters, dtype=float)

    for i in range(num_iters):
        objective = _expected_neg_makespan_autograd(diff_dp, weights, boosts, num_samples=num_samples)
        vals[i] = float(objective.item())
        grad_boosts = torch.autograd.grad(objective, boosts)[0]
        with torch.no_grad():
            boosts += lr * grad_boosts

    return AMAParams(weights.detach().cpu().numpy(), boosts.detach().cpu().numpy()), vals


def optimize_weights_and_boosts(
    diff_dp: DifferentiableSequentialScheduleDP,
    ama: AMAParams,
    num_samples: int = 10,
    lr: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    boosts = torch.tensor(ama.boosts, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    weights = torch.tensor(ama.weights, dtype=diff_dp.dtype, device=diff_dp.device_obj, requires_grad=True)
    vals = np.zeros(num_iters, dtype=float)

    for i in range(num_iters):
        objective = _expected_neg_makespan_autograd(diff_dp, weights, boosts, num_samples=num_samples)
        vals[i] = float(objective.item())
        grad_weights, grad_boosts = torch.autograd.grad(objective, (weights, boosts))
        with torch.no_grad():
            boosts += lr * grad_boosts
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
    gamma: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mdp_factory is not ScheduleMDP:
        raise TypeError("mdp_schedule_dp.runtrial currently only supports ScheduleMDP.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    diff_dp = DifferentiableSequentialScheduleDP(mdp, alpha=reg_strength)
    eval_dp = SequentialScheduleDPSolver(mdp)

    boosts = np.random.rand(len(mdp.state_list), len(mdp.action_list))
    ama = AMAParams(np.ones(mdp.n_agents, dtype=float), boosts.copy())
    vcg_ama = AMAParams(np.ones(mdp.n_agents, dtype=float), np.zeros_like(boosts))

    start_time = datetime.now()
    if optimize_weights:
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
    vcg_makespan, vcg_std = _expectedmakespan_dp_stats(eval_dp, vcg_ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)
    ama_makespan, ama_std = _expectedmakespan_dp_stats(eval_dp, ama, num_samples=TEST_SAMPLES, alpha=eval_alpha)

    result = {
        "method": "regdp",
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
        "vcg_performance": -vcg_makespan,
        "vcg_performance_std": vcg_std,
        "ama_performance": -ama_makespan,
        "ama_performance_std": ama_std,
        "vcg_revenue": vcg_makespan,
        "vcg_std": vcg_std,
        "ama_revenue": ama_makespan,
        "ama_std": ama_std,
        "runtime": end_time - start_time,
    }
    return result, {"ama": ama, "vals": vals}
