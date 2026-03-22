from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lp_and_ama import AMAParams, mean_and_std
from schedule import ScheduleMDP, ScheduleState


def _objective_coeffs(mdp: ScheduleMDP, ama: AMAParams, types: Any) -> np.ndarray:
    coeffs = np.zeros((len(mdp.state_list), len(mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(mdp.state_list):
        for action_idx, action in enumerate(mdp.action_list):
            reward = np.asarray(mdp.reward_from_alloc(state, action, types), dtype=float)
            coeffs[state_idx, action_idx] = float(reward @ ama.weights) + float(
                ama.boosts[state_idx, action_idx]
            )
    return coeffs


def _entropy_from_occupancy(x: np.ndarray) -> float:
    positive = x > 0.0
    if not np.any(positive):
        return 0.0
    return float(-np.sum(x[positive] * np.log(x[positive])))


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


@dataclass
class DPSolveResult:
    x: np.ndarray
    policy: np.ndarray
    state_values: np.ndarray
    state_occupancy: np.ndarray
    objective_value: float
    status: str


@dataclass
class SequentialScheduleDPSolver:
    mdp: ScheduleMDP

    def __post_init__(self) -> None:
        if not isinstance(self.mdp, ScheduleMDP):
            raise TypeError("SequentialScheduleDPSolver only supports ScheduleMDP.")
        if not np.isclose(self.mdp.gamma, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("SequentialScheduleDPSolver currently assumes gamma=1.")

        self.state_list = list(self.mdp.state_list)
        self.action_list = list(self.mdp.action_list)
        self.num_states = len(self.state_list)
        self.num_actions = len(self.action_list)
        self.state_to_index = {state: idx for idx, state in enumerate(self.state_list)}
        self.states_by_stage = self._build_states_by_stage()
        self.child_index = self._build_child_index()

    def _build_states_by_stage(self) -> list[list[int]]:
        states_by_stage: list[list[int]] = [[] for _ in range(self.mdp.n_items)]
        for state_idx, state in enumerate(self.state_list):
            stage = int(state[0]) - 1
            if stage < 0 or stage >= self.mdp.n_items:
                raise ValueError(f"Unexpected schedule stage {state[0]} for state {state}.")
            states_by_stage[stage].append(state_idx)
        return states_by_stage

    def _next_state(self, state: ScheduleState, action: int) -> ScheduleState | None:
        if int(state[0]) >= self.mdp.n_items:
            return None
        new_state = list(state)
        new_state[new_state[0]] = action
        new_state[0] += 1
        return tuple(new_state)

    def _build_child_index(self) -> np.ndarray:
        child_index = np.full((self.num_states, self.num_actions), -1, dtype=int)
        for state_idx, state in enumerate(self.state_list):
            for action_idx, action in enumerate(self.action_list):
                next_state = self._next_state(state, action)
                if next_state is None:
                    continue
                child_index[state_idx, action_idx] = self.state_to_index[next_state]
        return child_index

    def solve(self, types: Any, ama: AMAParams, alpha: float = 0.01) -> DPSolveResult:
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative.")

        coeffs = _objective_coeffs(self.mdp, ama, types)
        state_values = np.zeros(self.num_states, dtype=float)
        policy = np.zeros((self.num_states, self.num_actions), dtype=float)

        for stage in range(self.mdp.n_items - 1, -1, -1):
            beta = alpha * float(self.mdp.n_items - stage)
            for state_idx in self.states_by_stage[stage]:
                q_values = coeffs[state_idx].copy()
                child_indices = self.child_index[state_idx]
                has_child = child_indices >= 0
                q_values[has_child] += state_values[child_indices[has_child]]

                if alpha == 0.0:
                    best_action_idx = int(np.argmax(q_values))
                    policy[state_idx, best_action_idx] = 1.0
                    state_values[state_idx] = float(q_values[best_action_idx])
                    continue

                logits = q_values / beta
                log_partition = _logsumexp(logits)
                policy[state_idx] = np.exp(logits - log_partition)
                state_values[state_idx] = beta * log_partition

        state_occupancy = np.zeros(self.num_states, dtype=float)
        x = np.zeros((self.num_states, self.num_actions), dtype=float)
        root_index = self.state_to_index[self.mdp.startstate()]
        state_occupancy[root_index] = 1.0

        for stage in range(self.mdp.n_items):
            for state_idx in self.states_by_stage[stage]:
                occ = state_occupancy[state_idx]
                if occ == 0.0:
                    continue
                x[state_idx] = occ * policy[state_idx]
                for action_idx, child_idx in enumerate(self.child_index[state_idx]):
                    if child_idx >= 0:
                        state_occupancy[child_idx] += x[state_idx, action_idx]

        objective_value = float(np.sum(coeffs * x))
        if alpha > 0.0:
            objective_value += alpha * _entropy_from_occupancy(x)

        return DPSolveResult(
            x=x,
            policy=policy,
            state_values=state_values,
            state_occupancy=state_occupancy,
            objective_value=objective_value,
            status="optimal",
        )


def calcmakespan_dp(
    solver: SequentialScheduleDPSolver,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> float:
    result = solver.solve(types, ama, alpha)
    total_makespan = 0.0
    for state_idx, state in enumerate(solver.mdp.state_list):
        for action_idx, action in enumerate(solver.mdp.action_list):
            total_makespan += float(result.x[state_idx, action_idx]) * float(
                solver.mdp.makespan_from_sa(state, action, types)
            )
    return total_makespan


def expectedmakespan_dp(
    solver: SequentialScheduleDPSolver,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> tuple[float, float]:
    makespans = [calcmakespan_dp(solver, types, ama, alpha) for types in solver.mdp.sampletypes(num_samples)]
    return mean_and_std(makespans)
