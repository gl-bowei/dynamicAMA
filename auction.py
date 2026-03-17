from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


AuctionState = Tuple[int, ...]


def auction_state_and_action_list(
    n_agents: int, n_items: int
) -> tuple[list[AuctionState], list[int]]:
    actions = list(range(0, n_agents + 1))
    start_state = tuple([0] + [-1] * n_items)
    state_list: list[AuctionState] = [start_state]
    states_to_process: list[AuctionState] = [start_state]

    while states_to_process:
        state = states_to_process.pop(0)
        for action in actions:
            if (state[0] + 1) == n_items:
                continue
            new_state = list(state)
            new_state[0] += 1
            new_state[new_state[0]] = action
            new_state_tuple = tuple(new_state)
            state_list.append(new_state_tuple)
            states_to_process.append(new_state_tuple)

    return state_list, actions


@dataclass
class AuctionMDP:
    n_agents: int
    n_items: int
    gamma: float
    dist_type: str

    def __post_init__(self) -> None:
        self.state_list, self.action_list = auction_state_and_action_list(
            self.n_agents, self.n_items
        )

    def startstate(self) -> AuctionState:
        return tuple([0] + [-1] * self.n_items)

    def transition_probability(
        self, prev_state: AuctionState, prev_action: int, state: AuctionState
    ) -> float:
        prob = 1.0
        if prev_state[0] + 1 != state[0]:
            prob *= 0.0
        if prev_state[1 : prev_state[0] + 1] != state[1 : prev_state[0] + 1]:
            prob *= 0.0
        if prev_action != state[prev_state[0] + 1]:
            prob *= 0.0
        return prob

    def reward_from_alloc(
        self, state: AuctionState, action: int, type_params: Sequence[float]
    ) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=float)
        if action != 0 and action not in state[1:]:
            rewards[action - 1] = float(type_params[action - 1])
        return rewards

    def nonterminal(self, state: AuctionState) -> bool:
        return state[0] < self.n_items

    def counterfactualtype(
        self, types: Sequence[float], agent_idx: int
    ) -> np.ndarray:
        result = np.asarray(types, dtype=float).copy()
        result[agent_idx] = 0.0
        return result

    def sampletypes(self, num_samples: int) -> List[np.ndarray]:
        if self.dist_type == "uniform":
            return [np.random.rand(self.n_agents) for _ in range(num_samples)]
        if self.dist_type == "asymmetric":
            weights = np.asarray([1.0 / i for i in range(1, self.n_agents + 1)])
            return [weights * np.random.rand(self.n_agents) for _ in range(num_samples)]
        raise ValueError("Unknown distribution type")
