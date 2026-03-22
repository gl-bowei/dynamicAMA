from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


ScheduleState = Tuple[int, ...]
SCHEDULER_SENTINEL = 5.0


def task_state_and_action_list(
    n_agents: int, n_items: int
) -> tuple[list[ScheduleState], list[int]]:
    actions = list(range(1, n_agents + 1))
    start_state = tuple([1] + [-1] * n_items)
    state_list: list[ScheduleState] = [start_state]
    states_to_process: list[ScheduleState] = [start_state]

    while states_to_process:
        state = states_to_process.pop(0)
        for action in actions:
            if state[0] == n_items:
                continue
            new_state = list(state)
            new_state[new_state[0]] = action
            new_state[0] += 1
            new_state_tuple = tuple(new_state)
            state_list.append(new_state_tuple)
            states_to_process.append(new_state_tuple)

    return state_list, actions


@dataclass
class ScheduleMDP:
    n_agents: int
    n_items: int
    gamma: float
    dist_type: str

    def __post_init__(self) -> None:
        self.state_list, self.action_list = task_state_and_action_list(
            self.n_agents, self.n_items
        )

    def startstate(self) -> ScheduleState:
        return tuple([1] + [-1] * self.n_items)

    def transition_probability(
        self, prev_state: ScheduleState, prev_action: int, state: ScheduleState
    ) -> float:
        prob = 1.0
        if prev_state[0] + 1 != state[0]:
            prob *= 0.0
        if prev_state[1 : prev_state[0]] != state[1 : prev_state[0]]:
            prob *= 0.0
        if prev_action != state[prev_state[0]]:
            prob *= 0.0
        return prob

    def reward_from_alloc(
        self, state: ScheduleState, action: int, type_params: np.ndarray
    ) -> np.ndarray:
        rewards = np.zeros(self.n_agents, dtype=float)
        rewards[action - 1] = -float(type_params[action - 1, state[0] - 1])
        return rewards

    def remaining_makespan(
        self, agent_types: np.ndarray, final_assignments: List[int]
    ) -> float:
        rem_makespan = np.zeros(self.n_agents, dtype=float)
        for task in range(self.n_items):
            rem_makespan[rem_makespan > 0] -= 1
            assigned_agent = final_assignments[task]
            rem_makespan[assigned_agent - 1] += float(agent_types[assigned_agent - 1, task])
        return float(max(np.max(rem_makespan), 0.0))

    def makespan_from_sa(
        self, state: ScheduleState, action: int, type_params: np.ndarray
    ) -> float:
        if state[0] < self.n_items:
            return 0.0
        assignments = [*state[1 : len(state) - 1], action]
        return self.remaining_makespan(type_params, assignments)

    def sampletypes(self, num_samples: int) -> List[np.ndarray]:
        if self.dist_type == "uniform":
            return [3.0 * np.random.rand(self.n_agents, self.n_items) for _ in range(num_samples)]
        if self.dist_type == "asymmetric":
            weights = np.asarray([float(i) for i in range(1, self.n_agents + 1)]).reshape(-1, 1)
            return [weights * (3.0 * np.random.rand(self.n_agents, self.n_items)) for _ in range(num_samples)]
        raise ValueError("unknown distribution type")

    def nonterminal(self, state: ScheduleState) -> bool:
        return state[0] <= self.n_items

    def counterfactualtype(self, types: np.ndarray, agent_idx: int) -> np.ndarray:
        result = np.asarray(types, dtype=float).copy()
        result[agent_idx, :] = SCHEDULER_SENTINEL
        return result
