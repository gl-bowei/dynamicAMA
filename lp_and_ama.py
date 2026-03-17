from __future__ import annotations

from dataclasses import dataclass
from statistics import stdev
from typing import Any, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError as exc:
    raise ImportError(
        "lp_and_ama.py requires cvxpy in the active Python environment."
    ) from exc


State = Any
Action = Any
TypeProfile = Any
ArrayLike = np.ndarray


class MDP(Protocol):
    n_agents: int
    n_items: int
    gamma: float
    state_list: Sequence[State]
    action_list: Sequence[Action]

    def startstate(self) -> State:
        ...

    def nonterminal(self, state: State) -> bool:
        ...

    def transition_probability(
        self, prev_state: State, prev_action: Action, state: State
    ) -> float:
        ...

    def reward_from_alloc(
        self, state: State, action: Action, types: TypeProfile
    ) -> Sequence[float]:
        ...

    def counterfactualtype(self, types: TypeProfile, agent_idx: int) -> TypeProfile:
        ...

    def sampletypes(self, num_samples: int) -> Sequence[TypeProfile]:
        ...


@dataclass
class AMAParams:
    weights: ArrayLike
    boosts: ArrayLike

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.boosts = np.asarray(self.boosts, dtype=float)


@dataclass
class SolveResult:
    x: ArrayLike
    status: str
    objective_value: float


@dataclass
class MDPLinearProgram:
    mdp: MDP
    solver: str = "SCS"
    solver_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.num_states = len(self.mdp.state_list)
        self.num_actions = len(self.mdp.action_list)
        self.state_list = list(self.mdp.state_list)
        self.action_list = list(self.mdp.action_list)
        self.gamma = float(self.mdp.gamma)

        self.x = cp.Variable((self.num_states, self.num_actions), nonneg=True)
        self._transition_tensor = self._build_transition_tensor()
        self._constraints = self._build_flow_constraints(self.x)

        self.last_problem: Optional[cp.Problem] = None
        self.last_result: Optional[SolveResult] = None

    def _build_transition_tensor(self) -> ArrayLike:
        tensor = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=float)
        for prev_state_idx, prev_state in enumerate(self.state_list):
            for prev_action_idx, prev_action in enumerate(self.action_list):
                for state_idx, state in enumerate(self.state_list):
                    tensor[prev_state_idx, prev_action_idx, state_idx] = (
                        self.mdp.transition_probability(prev_state, prev_action, state)
                    )
        return tensor

    def _build_flow_constraints(self, x_var: "cp.Variable") -> list["cp.Constraint"]:
        constraints: list[cp.Constraint] = []
        for state_idx, state in enumerate(self.state_list):
            if not self.mdp.nonterminal(state):
                continue

            lhs = cp.sum(x_var[state_idx, :])
            rhs = self.gamma * cp.sum(cp.multiply(self._transition_tensor[:, :, state_idx], x_var))
            if state == self.mdp.startstate():
                rhs += 1.0
            constraints.append(lhs == rhs)
        return constraints

    def objective_expr(
        self,
        x_var: "cp.Expression",
        types: TypeProfile,
        ama: AMAParams,
        alpha: float,
    ) -> "cp.Expression":
        coeffs = np.zeros((self.num_states, self.num_actions), dtype=float)
        for state_idx, state in enumerate(self.state_list):
            for action_idx, action in enumerate(self.action_list):
                reward = np.asarray(self.mdp.reward_from_alloc(state, action, types), dtype=float)
                coeffs[state_idx, action_idx] = float(reward @ ama.weights) + ama.boosts[
                    state_idx, action_idx
                ]

        objective = cp.sum(cp.multiply(coeffs, x_var))
        if alpha != 0.0:
            objective += alpha * cp.sum(cp.entr(x_var))
        return objective

    def solve(self, types: TypeProfile, ama: AMAParams, alpha: float) -> SolveResult:
        problem = cp.Problem(cp.Maximize(self.objective_expr(self.x, types, ama, alpha)), self._constraints)
        kwargs = dict(self.solver_kwargs or {})
        problem.solve(solver=self.solver, verbose=False, **kwargs)
        result = SolveResult(
            x=_as_array(self.x.value, (self.num_states, self.num_actions)),
            status=str(problem.status),
            objective_value=_as_float(problem.value),
        )
        self.last_problem = problem
        self.last_result = result
        return result


@dataclass
class UnregMDP:
    mdp: MDP
    solver: str = "SCS"
    solver_kwargs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.num_states = len(self.mdp.state_list)
        self.num_actions = len(self.mdp.action_list)
        self.state_list = list(self.mdp.state_list)
        self.action_list = list(self.mdp.action_list)
        self.gamma = float(self.mdp.gamma)

        self.x = cp.Variable((self.num_states, self.num_actions), nonneg=True)
        self._transition_tensor = self._build_transition_tensor()
        self._constraints = self._build_flow_constraints(self.x)

        self.last_problem: Optional[cp.Problem] = None
        self.last_result: Optional[SolveResult] = None

    def _build_transition_tensor(self) -> ArrayLike:
        tensor = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=float)
        for prev_state_idx, prev_state in enumerate(self.state_list):
            for prev_action_idx, prev_action in enumerate(self.action_list):
                for state_idx, state in enumerate(self.state_list):
                    tensor[prev_state_idx, prev_action_idx, state_idx] = (
                        self.mdp.transition_probability(prev_state, prev_action, state)
                    )
        return tensor

    def _build_flow_constraints(self, x_var: "cp.Variable") -> list["cp.Constraint"]:
        constraints: list[cp.Constraint] = []
        for state_idx, state in enumerate(self.state_list):
            if not self.mdp.nonterminal(state):
                continue

            lhs = cp.sum(x_var[state_idx, :])
            rhs = self.gamma * cp.sum(cp.multiply(self._transition_tensor[:, :, state_idx], x_var))
            if state == self.mdp.startstate():
                rhs += 1.0
            constraints.append(lhs == rhs)
        return constraints

    def solve(self, types: TypeProfile, ama: AMAParams) -> SolveResult:
        coeffs = np.zeros((self.num_states, self.num_actions), dtype=float)
        for state_idx, state in enumerate(self.state_list):
            for action_idx, action in enumerate(self.action_list):
                reward = np.asarray(self.mdp.reward_from_alloc(state, action, types), dtype=float)
                coeffs[state_idx, action_idx] = float(reward @ ama.weights) + ama.boosts[
                    state_idx, action_idx
                ]

        problem = cp.Problem(cp.Maximize(cp.sum(cp.multiply(coeffs, self.x))), self._constraints)
        kwargs = dict(self.solver_kwargs or {})
        problem.solve(solver=self.solver, verbose=False, **kwargs)
        result = SolveResult(
            x=_as_array(self.x.value, (self.num_states, self.num_actions)),
            status=str(problem.status),
            objective_value=_as_float(problem.value),
        )
        self.last_problem = problem
        self.last_result = result
        return result


def evalauction(
    auctionlp: MDPLinearProgram,
    types: TypeProfile,
    ama: AMAParams,
    alpha: float,
) -> SolveResult:
    return auctionlp.solve(types, ama, alpha)


def evalwithoutreg(auctionlp: UnregMDP, types: TypeProfile, ama: AMAParams) -> SolveResult:
    result = auctionlp.solve(types, ama)
    if result.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Unregularized solve failed with status={result.status}")
    return result


def asw(mdp: MDP, x: ArrayLike, types: TypeProfile, ama: AMAParams) -> float:
    total_asw = 0.0
    for state_idx, state in enumerate(mdp.state_list):
        for action_idx, action in enumerate(mdp.action_list):
            reward = np.asarray(mdp.reward_from_alloc(state, action, types), dtype=float)
            total_asw += float(x[state_idx, action_idx]) * (
                float(reward @ ama.weights) + float(ama.boosts[state_idx, action_idx])
            )
    return total_asw


def sw(mdp: MDP, x: ArrayLike, types: TypeProfile) -> float:
    total_sw = 0.0
    for state_idx, state in enumerate(mdp.state_list):
        for action_idx, action in enumerate(mdp.action_list):
            reward = np.asarray(mdp.reward_from_alloc(state, action, types), dtype=float)
            total_sw += float(x[state_idx, action_idx]) * float(np.sum(reward))
    return total_sw


def calcrevenue(
    lp: MDPLinearProgram,
    types: TypeProfile,
    ama: AMAParams,
    alpha: float,
    require_optimal: bool = False,
) -> float:
    main_result = evalauction(lp, types, ama, alpha)
    _ensure_solved(main_result.status, require_optimal, "main problem", types, ama, alpha)

    asw_main_x = asw(lp.mdp, main_result.x, types, ama)
    sw_main_x = sw(lp.mdp, main_result.x, types)
    revenue = sw_main_x

    for agent_idx in range(lp.mdp.n_agents):
        counterfactual_types = lp.mdp.counterfactualtype(types, agent_idx)
        cf_result = evalauction(lp, counterfactual_types, ama, alpha)
        _ensure_solved(
            cf_result.status,
            require_optimal,
            f"counterfactual problem for agent {agent_idx}",
            types,
            ama,
            alpha,
        )
        asw_counterfactual_x = asw(lp.mdp, cf_result.x, counterfactual_types, ama)
        revenue += (asw_counterfactual_x - asw_main_x) / float(ama.weights[agent_idx])
    return revenue


def calcmakespan(
    lp: MDPLinearProgram,
    types: TypeProfile,
    ama: AMAParams,
    alpha: float,
    require_optimal: bool = False,
) -> float:
    makespan_fn = getattr(lp.mdp, "makespan_from_sa", None)
    if makespan_fn is None:
        raise TypeError("calcmakespan requires an MDP implementing makespan_from_sa.")

    result = evalauction(lp, types, ama, alpha)
    _ensure_solved(result.status, require_optimal, "makespan problem", types, ama, alpha)

    total_makespan = 0.0
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            total_makespan += float(result.x[state_idx, action_idx]) * float(
                makespan_fn(state, action, types)
            )
    return total_makespan


def expectedrevenue(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
    require_optimal: bool = False,
) -> Tuple[float, float]:
    revenues = [
        calcrevenue(lp, types, ama, alpha, require_optimal=require_optimal)
        for types in lp.mdp.sampletypes(num_samples)
    ]
    return mean_and_std(revenues)


def expectedmakespan(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
    require_optimal: bool = False,
) -> Tuple[float, float]:
    makespans = [
        calcmakespan(lp, types, ama, alpha, require_optimal=require_optimal)
        for types in lp.mdp.sampletypes(num_samples)
    ]
    return mean_and_std(makespans)


def expectedperformance(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 1000,
    alpha: float = 0.01,
    require_optimal: bool = False,
) -> Tuple[float, float]:
    if hasattr(lp.mdp, "makespan_from_sa"):
        mkspan, std = expectedmakespan(
            lp,
            ama,
            num_samples=num_samples,
            alpha=alpha,
            require_optimal=require_optimal,
        )
        return -mkspan, std
    return expectedrevenue(
        lp,
        ama,
        num_samples=num_samples,
        alpha=alpha,
        require_optimal=require_optimal,
    )


def dsw_dx(mdp: MDP, x: ArrayLike, types: TypeProfile) -> ArrayLike:
    del x
    deriv = np.zeros((len(mdp.state_list), len(mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(mdp.state_list):
        for action_idx, action in enumerate(mdp.action_list):
            deriv[state_idx, action_idx] = float(
                np.sum(np.asarray(mdp.reward_from_alloc(state, action, types), dtype=float))
            )
    return deriv


def dasw_dx(mdp: MDP, x: ArrayLike, types: TypeProfile, ama: AMAParams) -> ArrayLike:
    del x
    deriv = np.zeros((len(mdp.state_list), len(mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(mdp.state_list):
        for action_idx, action in enumerate(mdp.action_list):
            reward = np.asarray(mdp.reward_from_alloc(state, action, types), dtype=float)
            deriv[state_idx, action_idx] = float(reward @ ama.weights) + float(
                ama.boosts[state_idx, action_idx]
            )
    return deriv


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = float(sum(values) / len(values))
    std = float(stdev(values)) if len(values) > 1 else 0.0
    return mean, std


def _as_array(value: Any, shape: tuple[int, int]) -> ArrayLike:
    if value is None:
        return np.zeros(shape, dtype=float)
    return np.asarray(value, dtype=float)


def _as_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _ensure_solved(
    status: str,
    require_optimal: bool,
    problem_name: str,
    types: TypeProfile,
    ama: AMAParams,
    alpha: float,
) -> None:
    acceptable = {"optimal", "optimal_inaccurate"}
    if status in acceptable:
        return
    if require_optimal:
        raise RuntimeError(
            f"{problem_name} failed with status={status}, alpha={alpha}, "
            f"types={types}, ama={ama}"
        )
    print(f"WARNING: {problem_name} returned status={status}")
