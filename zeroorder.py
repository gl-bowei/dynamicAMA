from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

import numpy as np

from lp_and_ama import AMAParams, MDPLinearProgram, asw, calcrevenue, expectedmakespan, expectedperformance, expectedrevenue, evalauction


TEST_SAMPLES = 10000
MIN_WEIGHT = 1e-6


def namedtuple_to_csv_line_str(result: dict[str, Any]) -> str:
    return "\t".join(str(v) for v in result.values())


def _reward_vector(lp: MDPLinearProgram, state_idx: int, action_idx: int, types: Any) -> np.ndarray:
    state = lp.mdp.state_list[state_idx]
    action = lp.mdp.action_list[action_idx]
    return np.asarray(lp.mdp.reward_from_alloc(state, action, types), dtype=float)


def _dmakespan_dx(lp: MDPLinearProgram, x: np.ndarray, types: Any) -> np.ndarray:
    del x
    makespan_fn = getattr(lp.mdp, "makespan_from_sa", None)
    if makespan_fn is None:
        raise TypeError("Schedule-style zero-order code requires makespan_from_sa.")

    grad = np.zeros((len(lp.mdp.state_list), len(lp.mdp.action_list)), dtype=float)
    for state_idx, state in enumerate(lp.mdp.state_list):
        for action_idx, action in enumerate(lp.mdp.action_list):
            grad[state_idx, action_idx] = float(makespan_fn(state, action, types))
    return grad


def _project_positive_weights(weights: np.ndarray, min_weight: float = MIN_WEIGHT) -> np.ndarray:
    return np.maximum(weights, min_weight)


def zeroorder_jac_makespan(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    normal_samples = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise in normal_samples:
        new_ama = AMAParams(ama.weights.copy(), ama.boosts + noise_magnitude * noise)
        new_x = evalauction(lp, types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term = np.zeros_like(ama.boosts)
    grady = _dmakespan_dx(lp, main_x, types)
    for noise, delta in zip(normal_samples, deltas):
        average_jacobian_term += np.sum(delta * grady) * noise
    return average_jacobian_term / num_perturbations


def zeroorder_jac_makespan_wb(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    normal_samples_b = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    normal_samples_w = [np.random.randn(*ama.weights.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise_w, noise_b in zip(normal_samples_w, normal_samples_b):
        new_ama = AMAParams(
            ama.weights + noise_magnitude * noise_w,
            ama.boosts + noise_magnitude * noise_b,
        )
        new_x = evalauction(lp, types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term_b = np.zeros_like(ama.boosts)
    average_jacobian_term_w = np.zeros_like(ama.weights)
    grady = _dmakespan_dx(lp, main_x, types)
    for noise_w, noise_b, delta in zip(normal_samples_w, normal_samples_b, deltas):
        inner_product_term = np.sum(delta * grady)
        average_jacobian_term_b += inner_product_term * noise_b
        average_jacobian_term_w += inner_product_term * noise_w
    return average_jacobian_term_w / num_perturbations, average_jacobian_term_b / num_perturbations


def zeroorder_jac_factual_rev_b(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    normal_samples = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise in normal_samples:
        new_ama = AMAParams(ama.weights.copy(), ama.boosts + noise_magnitude * noise)
        new_x = evalauction(lp, types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term = np.zeros_like(ama.boosts)
    grady = np.zeros_like(ama.boosts)
    for state_idx in range(len(lp.mdp.state_list)):
        for action_idx in range(len(lp.mdp.action_list)):
            rew = _reward_vector(lp, state_idx, action_idx, types)
            grady[state_idx, action_idx] = (
                sum(
                    (1.0 / ama.weights[i]) * -(float(rew @ ama.weights) + ama.boosts[state_idx, action_idx])
                    for i in range(lp.mdp.n_agents)
                )
                + float(np.sum(rew))
            )
    for noise, delta in zip(normal_samples, deltas):
        average_jacobian_term += np.sum(delta * grady) * noise
    return average_jacobian_term / num_perturbations


def zeroorder_jac_factual_rev_wb(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    normal_samples_b = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    normal_samples_w = [np.random.randn(*ama.weights.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise_w, noise_b in zip(normal_samples_w, normal_samples_b):
        new_ama = AMAParams(
            ama.weights + noise_magnitude * noise_w,
            ama.boosts + noise_magnitude * noise_b,
        )
        new_x = evalauction(lp, types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term_b = np.zeros_like(ama.boosts)
    average_jacobian_term_w = np.zeros_like(ama.weights)
    grady = np.zeros_like(ama.boosts)
    for state_idx in range(len(lp.mdp.state_list)):
        for action_idx in range(len(lp.mdp.action_list)):
            rew = _reward_vector(lp, state_idx, action_idx, types)
            grady[state_idx, action_idx] = (
                sum(
                    (1.0 / ama.weights[i]) * -(float(rew @ ama.weights) + ama.boosts[state_idx, action_idx])
                    for i in range(lp.mdp.n_agents)
                )
                + float(np.sum(rew))
            )
    for noise_w, noise_b, delta in zip(normal_samples_w, normal_samples_b, deltas):
        inner_product_term = np.sum(delta * grady)
        average_jacobian_term_b += inner_product_term * noise_b
        average_jacobian_term_w += inner_product_term * noise_w
    return average_jacobian_term_w / num_perturbations, average_jacobian_term_b / num_perturbations


def zeroorder_jac_counterfactual_rev_wb(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    i: int,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    counterfactual_types = lp.mdp.counterfactualtype(types, i)
    normal_samples_b = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    normal_samples_w = [np.random.randn(*ama.weights.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise_w, noise_b in zip(normal_samples_w, normal_samples_b):
        new_ama = AMAParams(
            ama.weights + noise_magnitude * noise_w,
            ama.boosts + noise_magnitude * noise_b,
        )
        new_x = evalauction(lp, counterfactual_types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term_b = np.zeros_like(ama.boosts)
    average_jacobian_term_w = np.zeros_like(ama.weights)
    grady = np.zeros_like(ama.boosts)
    for state_idx in range(len(lp.mdp.state_list)):
        for action_idx in range(len(lp.mdp.action_list)):
            reward = _reward_vector(lp, state_idx, action_idx, counterfactual_types)
            grady[state_idx, action_idx] = (
                (1.0 / ama.weights[i])
                * (float(reward @ ama.weights) + ama.boosts[state_idx, action_idx])
            )
    for noise_w, noise_b, delta in zip(normal_samples_w, normal_samples_b, deltas):
        inner_product_term = np.sum(delta * grady)
        average_jacobian_term_b += inner_product_term * noise_b
        average_jacobian_term_w += inner_product_term * noise_w
    return average_jacobian_term_w / num_perturbations, average_jacobian_term_b / num_perturbations


def zeroorder_jac_counterfactual_rev_b(
    lp: MDPLinearProgram,
    main_x: np.ndarray,
    types: Any,
    ama: AMAParams,
    i: int,
    num_perturbations: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    counterfactual_types = lp.mdp.counterfactualtype(types, i)
    normal_samples = [np.random.randn(*ama.boosts.shape) for _ in range(num_perturbations)]
    deltas = []
    for noise in normal_samples:
        new_ama = AMAParams(ama.weights.copy(), ama.boosts + noise_magnitude * noise)
        new_x = evalauction(lp, counterfactual_types, new_ama, alpha).x
        deltas.append((new_x - main_x) / noise_magnitude)

    average_jacobian_term = np.zeros_like(ama.boosts)
    grady = np.zeros_like(ama.boosts)
    for state_idx in range(len(lp.mdp.state_list)):
        for action_idx in range(len(lp.mdp.action_list)):
            reward = _reward_vector(lp, state_idx, action_idx, counterfactual_types)
            grady[state_idx, action_idx] = (
                (1.0 / ama.weights[i])
                * (float(reward @ ama.weights) + ama.boosts[state_idx, action_idx])
            )
    for noise, delta in zip(normal_samples, deltas):
        average_jacobian_term += np.sum(delta * grady) * noise
    return average_jacobian_term / num_perturbations


def zeroorder_makespangradb(
    lp: MDPLinearProgram,
    types: Any,
    ama: AMAParams,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    main_x = evalauction(lp, types, ama, alpha).x
    return zeroorder_jac_makespan(
        lp,
        main_x,
        types,
        ama,
        num_perturbations=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )


def zeroorder_makespangradwb(
    lp: MDPLinearProgram,
    types: Any,
    ama: AMAParams,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    main_x = evalauction(lp, types, ama, alpha).x
    return zeroorder_jac_makespan_wb(
        lp,
        main_x,
        types,
        ama,
        num_perturbations=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )


def zeroorder_revenuegradb(
    lp: MDPLinearProgram,
    types: Any,
    ama: AMAParams,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    main_x = evalauction(lp, types, ama, alpha).x
    leader_grad_term = np.zeros_like(ama.boosts)
    counterfactual_solns = []

    for i in range(lp.mdp.n_agents):
        counterfactual_types = lp.mdp.counterfactualtype(types, i)
        counterfactual_x = evalauction(lp, counterfactual_types, ama, alpha).x
        counterfactual_solns.append(counterfactual_x)
        leader_grad_term += (counterfactual_x - main_x) / ama.weights[i]

    average_jacobian_term = zeroorder_jac_factual_rev_b(
        lp,
        main_x,
        types,
        ama,
        num_perturbations=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )

    counterfactual_jacobian_estimates = []
    for i in range(lp.mdp.n_agents):
        counterfactual_types = lp.mdp.counterfactualtype(types, i)
        counterfactual_jacobian_estimates.append(
            zeroorder_jac_counterfactual_rev_b(
                lp,
                counterfactual_solns[i],
                counterfactual_types,
                ama,
                i,
                num_perturbations=num_perturb,
                alpha=alpha,
                noise_magnitude=noise_magnitude,
            )
        )

    return leader_grad_term + average_jacobian_term + sum(counterfactual_jacobian_estimates)


def zeroorder_revenuegrad_wb(
    lp: MDPLinearProgram,
    types: Any,
    ama: AMAParams,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    main_x = evalauction(lp, types, ama, alpha).x
    leader_grad_term_b = np.zeros_like(ama.boosts)
    leader_grad_term_w = np.zeros_like(ama.weights)
    main_asw = asw(lp.mdp, main_x, types, ama)
    counterfactual_solns = []
    counterfactual_asws = np.zeros(lp.mdp.n_agents, dtype=float)

    for i in range(lp.mdp.n_agents):
        counterfactual_types = lp.mdp.counterfactualtype(types, i)
        counterfactual_x = evalauction(lp, counterfactual_types, ama, alpha).x
        counterfactual_solns.append(counterfactual_x)
        counterfactual_asws[i] = asw(lp.mdp, counterfactual_x, counterfactual_types, ama)
        leader_grad_term_b += (counterfactual_x - main_x) / ama.weights[i]

    leader_grad_term_w += main_asw / (ama.weights**2)

    factual_sw_by_agent = np.zeros(lp.mdp.n_agents, dtype=float)
    counterfactual_sw_by_agent = np.zeros((lp.mdp.n_agents, lp.mdp.n_agents), dtype=float)

    for state_idx in range(len(lp.mdp.state_list)):
        for action_idx in range(len(lp.mdp.action_list)):
            factual_sw_by_agent += main_x[state_idx, action_idx] * _reward_vector(
                lp, state_idx, action_idx, types
            )
            for i in range(lp.mdp.n_agents):
                cf_types = lp.mdp.counterfactualtype(types, i)
                counterfactual_sw_by_agent[i, :] += counterfactual_solns[i][state_idx, action_idx] * _reward_vector(
                    lp, state_idx, action_idx, cf_types
                )

    sumrecip = np.sum(1.0 / ama.weights)
    leader_grad_term_w -= sumrecip * factual_sw_by_agent

    for j in range(lp.mdp.n_agents):
        leader_grad_term_w[j] -= counterfactual_asws[j] / (ama.weights[j] ** 2)
        for i in range(lp.mdp.n_agents):
            if i != j:
                leader_grad_term_w[j] += counterfactual_sw_by_agent[i, j] / ama.weights[i]

    average_jacobian_term_w, average_jacobian_term_b = zeroorder_jac_factual_rev_wb(
        lp,
        main_x,
        types,
        ama,
        num_perturbations=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )

    counterfactual_jacobian_estimates_b = []
    counterfactual_jacobian_estimates_w = []
    for i in range(lp.mdp.n_agents):
        counterfactual_types = lp.mdp.counterfactualtype(types, i)
        cf_w, cf_b = zeroorder_jac_counterfactual_rev_wb(
            lp,
            counterfactual_solns[i],
            counterfactual_types,
            ama,
            i,
            num_perturbations=num_perturb,
            alpha=alpha,
            noise_magnitude=noise_magnitude,
        )
        counterfactual_jacobian_estimates_w.append(cf_w)
        counterfactual_jacobian_estimates_b.append(cf_b)

    return (
        leader_grad_term_w + average_jacobian_term_w + sum(counterfactual_jacobian_estimates_w),
        leader_grad_term_b + average_jacobian_term_b + sum(counterfactual_jacobian_estimates_b),
    )


def zeroorder_expectedrevenuegrad(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    types_samples = lp.mdp.sampletypes(num_samples)
    rev_grad_b = np.zeros_like(ama.boosts)
    for types in types_samples:
        rev_grad_b += zeroorder_revenuegradb(
            lp,
            types,
            ama,
            num_perturb=num_perturb,
            alpha=alpha,
            noise_magnitude=noise_magnitude,
        )
    return rev_grad_b / num_samples


def zeroorder_expectedrevenuegrad_wb(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    types_samples = lp.mdp.sampletypes(num_samples)
    rev_grad_w, rev_grad_b = zeroorder_revenuegrad_wb(
        lp,
        types_samples[0],
        ama,
        num_perturb=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )
    for types in types_samples[1:]:
        grad_w_i, grad_b_i = zeroorder_revenuegrad_wb(
            lp,
            types,
            ama,
            num_perturb=num_perturb,
            alpha=alpha,
            noise_magnitude=noise_magnitude,
        )
        rev_grad_w += grad_w_i
        rev_grad_b += grad_b_i
    return rev_grad_w / num_samples, rev_grad_b / num_samples


def zeroorder_expectedmakespan_grad(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> np.ndarray:
    types_samples = lp.mdp.sampletypes(num_samples)
    grad_b = np.zeros_like(ama.boosts)
    for types in types_samples:
        grad_b += zeroorder_makespangradb(
            lp,
            types,
            ama,
            num_perturb=num_perturb,
            alpha=alpha,
            noise_magnitude=noise_magnitude,
        )
    return grad_b / num_samples


def zeroorder_expectedmakespan_grad_wb(
    lp: MDPLinearProgram,
    ama: AMAParams,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    noise_magnitude: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    types_samples = lp.mdp.sampletypes(num_samples)
    grad_w, grad_b = zeroorder_makespangradwb(
        lp,
        types_samples[0],
        ama,
        num_perturb=num_perturb,
        alpha=alpha,
        noise_magnitude=noise_magnitude,
    )
    for types in types_samples[1:]:
        grad_w_i, grad_b_i = zeroorder_makespangradwb(
            lp,
            types,
            ama,
            num_perturb=num_perturb,
            alpha=alpha,
            noise_magnitude=noise_magnitude,
        )
        grad_w += grad_w_i
        grad_b += grad_b_i
    return grad_w / num_samples, grad_b / num_samples


def optimize_boosts(
    lp: MDPLinearProgram,
    ama: AMAParams,
    objective_kind: str,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    lr: float = 0.01,
    noise_magnitude: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    vals = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        if objective_kind == "revenue":
            est_grad = zeroorder_expectedrevenuegrad(
                lp, ama, num_samples=num_samples, num_perturb=num_perturb, alpha=alpha, noise_magnitude=noise_magnitude
            )
            ama.boosts += lr * est_grad
            vals[i] = expectedrevenue(lp, ama, num_samples=num_samples, alpha=alpha)[0]
        elif objective_kind == "makespan":
            est_grad = zeroorder_expectedmakespan_grad(
                lp, ama, num_samples=num_samples, num_perturb=num_perturb, alpha=alpha, noise_magnitude=noise_magnitude
            )
            ama.boosts -= lr * est_grad
            vals[i] = expectedmakespan(lp, ama, num_samples=num_samples, alpha=alpha)[0]
        else:
            raise ValueError("objective_kind must be 'revenue' or 'makespan'")
    return ama, vals


def optimize_weights_and_boosts(
    lp: MDPLinearProgram,
    ama: AMAParams,
    objective_kind: str,
    num_samples: int = 10,
    num_perturb: int = 10,
    alpha: float = 0.0,
    lr: float = 0.01,
    noise_magnitude: float = 0.01,
    num_iters: int = 100,
) -> tuple[AMAParams, np.ndarray]:
    vals = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        if objective_kind == "revenue":
            grad_w, grad_b = zeroorder_expectedrevenuegrad_wb(
                lp, ama, num_samples=num_samples, num_perturb=num_perturb, alpha=alpha, noise_magnitude=noise_magnitude
            )
            ama.weights = _project_positive_weights(ama.weights + lr * grad_w)
            ama.boosts += lr * grad_b
            vals[i] = expectedrevenue(lp, ama, num_samples=num_samples, alpha=alpha)[0]
        elif objective_kind == "makespan":
            grad_w, grad_b = zeroorder_expectedmakespan_grad_wb(
                lp, ama, num_samples=num_samples, num_perturb=num_perturb, alpha=alpha, noise_magnitude=noise_magnitude
            )
            ama.weights = _project_positive_weights(ama.weights - lr * grad_w)
            ama.boosts -= lr * grad_b
            vals[i] = expectedmakespan(lp, ama, num_samples=num_samples, alpha=alpha)[0]
        else:
            raise ValueError("objective_kind must be 'revenue' or 'makespan'")
    return ama, vals


def runtrial(
    num_agents: int,
    num_items: int,
    num_samples: int,
    num_perturb: int,
    num_training_iters: int,
    seed: int,
    mdp_factory: Callable[..., Any],
    lr: float,
    noise_magnitude: float,
    dist_type: str,
    optimize_weights: bool = False,
    gamma: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np.random.seed(seed)
    mdp = mdp_factory(num_agents, num_items, gamma, dist_type)
    lp = MDPLinearProgram(mdp)
    boosts = np.random.rand(*lp.x.shape)
    ama = AMAParams(np.ones(mdp.n_agents, dtype=float), boosts.copy())
    vcg_ama = AMAParams(np.ones(mdp.n_agents, dtype=float), np.zeros_like(boosts))

    objective_kind = "makespan" if hasattr(mdp, "makespan_from_sa") else "revenue"
    start_time = datetime.now()
    if optimize_weights:
        ama, vals = optimize_weights_and_boosts(
            lp,
            ama,
            objective_kind=objective_kind,
            num_samples=num_samples,
            num_perturb=num_perturb,
            alpha=0.0,
            lr=lr,
            noise_magnitude=noise_magnitude,
            num_iters=num_training_iters,
        )
    else:
        ama, vals = optimize_boosts(
            lp,
            ama,
            objective_kind=objective_kind,
            num_samples=num_samples,
            num_perturb=num_perturb,
            alpha=0.0,
            lr=lr,
            noise_magnitude=noise_magnitude,
            num_iters=num_training_iters,
        )
    end_time = datetime.now()

    vcg_revenue, vcg_std = expectedrevenue(lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=0.0, require_optimal=True)
    vcg_performance, vcg_performance_std = expectedperformance(lp, vcg_ama, num_samples=TEST_SAMPLES, alpha=0.0, require_optimal=True)
    ama_revenue, ama_std = expectedrevenue(lp, ama, num_samples=TEST_SAMPLES, alpha=0.0, require_optimal=True)
    ama_performance, ama_performance_std = expectedperformance(lp, ama, num_samples=TEST_SAMPLES, alpha=0.0, require_optimal=True)

    result = {
        "method": "zeroorder",
        "mdp": getattr(mdp_factory, "__name__", str(mdp_factory)),
        "num_agents": num_agents,
        "num_items": num_items,
        "seed": seed,
        "num_samples": num_samples,
        "test_samples": TEST_SAMPLES,
        "num_perturb": num_perturb,
        "lr": lr,
        "noise_magnitude": noise_magnitude,
        "num_training_iters": num_training_iters,
        "dist_type": dist_type,
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
