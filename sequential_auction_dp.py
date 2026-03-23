from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from auction import AuctionMDP, AuctionState
from lp_and_ama import AMAParams, asw, mean_and_std, sw


def _objective_coeffs(mdp: AuctionMDP, ama: AMAParams, types: Any) -> np.ndarray:
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
class SequentialAuctionDPSolver:
    mdp: AuctionMDP

    def __post_init__(self) -> None:
        if not isinstance(self.mdp, AuctionMDP):
            raise TypeError("SequentialAuctionDPSolver only supports AuctionMDP.")
        if not np.isclose(self.mdp.gamma, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "SequentialAuctionDPSolver currently assumes gamma=1. "
                "The occupancy-entropy to soft-DP reduction used here is implemented only for gamma=1."
            )

        self.state_list = list(self.mdp.state_list)
        self.action_list = list(self.mdp.action_list)
        self.num_states = len(self.state_list)
        self.num_actions = len(self.action_list)
        self.state_to_index = {state: idx for idx, state in enumerate(self.state_list)}
        self.action_to_index = {action: idx for idx, action in enumerate(self.action_list)}
        self.states_by_depth = self._build_states_by_depth()
        self.child_index = self._build_child_index()

    def _build_states_by_depth(self) -> list[list[int]]:
        states_by_depth: list[list[int]] = [[] for _ in range(self.mdp.n_items)]
        for state_idx, state in enumerate(self.state_list):
            depth = int(state[0])
            if depth < 0 or depth >= self.mdp.n_items:
                raise ValueError(f"Unexpected auction depth {depth} for state {state}.")
            states_by_depth[depth].append(state_idx)
        return states_by_depth

    def _next_state(self, state: AuctionState, action: int) -> AuctionState | None:
        depth = int(state[0])
        if depth >= self.mdp.n_items - 1:
            return None
        new_state = list(state)
        new_state[0] = depth + 1
        new_state[depth + 1] = action
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

        for depth in range(self.mdp.n_items - 1, -1, -1):
            beta = alpha * float(self.mdp.n_items - depth)
            for state_idx in self.states_by_depth[depth]:
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

        for depth in range(self.mdp.n_items):
            for state_idx in self.states_by_depth[depth]:
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


def evalauction_dp(
    solver: SequentialAuctionDPSolver,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> DPSolveResult:
    return solver.solve(types, ama, alpha)


def calcrevenue_dp(
    solver: SequentialAuctionDPSolver,
    types: Any,
    ama: AMAParams,
    alpha: float,
) -> float:
    main_result = solver.solve(types, ama, alpha)
    main_asw = asw(solver.mdp, main_result.x, types, ama)
    revenue = sw(solver.mdp, main_result.x, types)

    for agent_idx in range(solver.mdp.n_agents):
        counterfactual_types = solver.mdp.counterfactualtype(types, agent_idx)
        cf_result = solver.solve(counterfactual_types, ama, alpha)
        revenue += (asw(solver.mdp, cf_result.x, counterfactual_types, ama) - main_asw) / float(
            ama.weights[agent_idx]
        )
    return revenue


def expectedrevenue_dp(
    solver: SequentialAuctionDPSolver,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
) -> tuple[float, float]:
    revenues = [
        calcrevenue_dp(solver, types, ama, alpha) for types in solver.mdp.sampletypes(num_samples)
    ]
    return mean_and_std(revenues)

