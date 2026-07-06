"""Portfolio metrics for the paper/poster benchmarks.

Adds five metrics on top of the original repo's profit p.a. / Sharpe pair
(and complementary to the teammate-owned Calmar, max drawdown, Sortino,
and CVaR 5% work):

    1. annualized_return          geometric (CAGR)
    2. annualized_volatility      std * sqrt(252)
    3. value_at_risk_5            historical daily 5% VaR, positive = loss
    4. avg_turnover               mean one-way rebalance turnover
    5. tc_adjusted_annual_return  CAGR net of proportional transaction costs

All functions consume plain daily-return arrays and daily weight matrices,
so classical baselines and RL actors are measured identically.

Weight trajectories: the original repo's RLEvaluator only returns
(profit, sharpe), so turnover and transaction costs cannot be computed
from it. `evaluate_actor_with_weights` mirrors RLEvaluator's SPO/DPO
logic exactly (constant weights within each rebalance interval, portfolio
return = sum(w * r) per day, negatives optionally clamped) but returns the
full daily weight matrix and daily returns. Weight drift between
rebalances is ignored, matching the repo's evaluation semantics.
"""
import numpy as np
import pandas as pd
import torch


TRADING_DAYS = 252
RISK_FREE_RATE = 0.0418  # matches utilities/metrics.py (US 10Y note)
DEFAULT_COST_BPS = 10.0  # one-way proportional transaction cost


# ---------------------------------------------------------------------------
# Core metric functions (the five new metrics)
# ---------------------------------------------------------------------------

def annualized_return(daily_returns) -> float:
    """Geometric annualized return (CAGR).

    Note: the original repo's `calculate_test_performance` uses the
    arithmetic-mean annualization (1 + mean)^252 - 1. CAGR compounds the
    realized path instead, so it is lower when returns are volatile.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_DAYS / len(r)) - 1.0)


def annualized_volatility(daily_returns) -> float:
    """Annualized volatility: population std of daily returns * sqrt(252)."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    return float(np.std(r) * np.sqrt(TRADING_DAYS))


def value_at_risk_5(daily_returns) -> float:
    """Historical one-day 5% Value at Risk.

    Reported as a positive loss fraction: VaR = -5th percentile of daily
    returns. VaR of 0.02 means "5% of days lose more than 2%".
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) == 0:
        return float("nan")
    return float(-np.percentile(r, 5))


def turnover_series(weights_by_day) -> np.ndarray:
    """One-way turnover per day: 0.5 * sum_i |w_t,i - w_{t-1},i|.

    Days inside a hold period contribute 0; rebalance days contribute the
    one-way traded fraction of the portfolio. The first day has no
    predecessor and contributes 0 (initial position build is excluded).
    """
    w = np.asarray(weights_by_day, dtype=np.float64)
    if w.ndim != 2 or len(w) < 2:
        return np.zeros(max(len(w), 0))
    day_turnover = 0.5 * np.abs(np.diff(w, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], day_turnover])


def avg_turnover(weights_by_day) -> float:
    """Mean one-way turnover per rebalance event (not per day).

    Averaging over rebalance events rather than days keeps the number
    comparable between strategies with different rebalance frequencies.
    A static (SPO-style) allocation has turnover 0.
    """
    ts = turnover_series(weights_by_day)
    rebalances = ts[ts > 1e-12]
    if len(rebalances) == 0:
        return 0.0
    return float(np.mean(rebalances))


def tc_adjusted_returns(daily_returns, weights_by_day,
                        cost_bps: float = DEFAULT_COST_BPS) -> np.ndarray:
    """Daily returns net of proportional transaction costs.

    Cost model: each rebalance day pays cost_bps basis points on the
    one-way traded volume, i.e. net_r_t = r_t - (cost_bps / 1e4) *
    turnover_t.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    ts = turnover_series(weights_by_day)
    if len(ts) != len(r):
        raise ValueError(
            f"daily_returns has {len(r)} days but weights imply {len(ts)}"
        )
    return r - (cost_bps / 1e4) * ts


def tc_adjusted_annual_return(daily_returns, weights_by_day,
                              cost_bps: float = DEFAULT_COST_BPS) -> float:
    """Annualized (CAGR) return net of transaction costs."""
    return annualized_return(tc_adjusted_returns(daily_returns,
                                                 weights_by_day, cost_bps))


# ---------------------------------------------------------------------------
# Repo-compatible metrics, reimplemented locally so this module does not
# depend on the original repo's config/tickers globals.
# ---------------------------------------------------------------------------

def repo_profit_pa(daily_returns) -> float:
    """(1 + mean)^252 - 1, as in utilities/metrics.calculate_test_performance."""
    r = np.asarray(daily_returns, dtype=np.float64)
    return float((1.0 + np.mean(r)) ** TRADING_DAYS - 1.0)


def repo_sharpe(daily_returns) -> float:
    """Annualized Sharpe with rf=4.18%, as in utilities/metrics.py."""
    r = np.asarray(daily_returns, dtype=np.float64)
    mean_return = np.mean(r) * TRADING_DAYS
    std = np.std(r) * np.sqrt(TRADING_DAYS)
    eps = 1e-7
    return float((mean_return - RISK_FREE_RATE) / (std + eps)) if std != 0 else 0.0


def cumulative_returns(daily_returns) -> np.ndarray:
    """Cumulative growth path: cumprod(1 + r) - 1."""
    r = np.asarray(daily_returns, dtype=np.float64)
    return np.cumprod(1.0 + r) - 1.0


def compute_all_metrics(daily_returns, weights_by_day,
                        cost_bps: float = DEFAULT_COST_BPS) -> dict:
    """All metrics for one strategy evaluation, as a flat dict."""
    return {
        "profit_pa": repo_profit_pa(daily_returns),
        "sharpe": repo_sharpe(daily_returns),
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "var_5": value_at_risk_5(daily_returns),
        "avg_turnover": avg_turnover(weights_by_day),
        "tc_adjusted_annual_return": tc_adjusted_annual_return(
            daily_returns, weights_by_day, cost_bps),
        "num_test_days": int(len(np.asarray(daily_returns))),
    }


# ---------------------------------------------------------------------------
# Evaluations that produce (daily_returns, weights_by_day)
# ---------------------------------------------------------------------------

def reduce_negatives(vec: np.ndarray, clamp_min: float = -1.0) -> np.ndarray:
    """Copy of utilities/metrics.reduce_negatives (clamp then renormalize)."""
    clamped = np.maximum(vec, clamp_min)
    return clamped / np.sum(clamped)


def constant_weights_evaluation(test_returns: pd.DataFrame, weights) -> dict:
    """Static allocation held over the whole test period (SPO / Equal Weight)."""
    w = np.asarray(weights, dtype=np.float64)
    daily = test_returns.values @ w
    weights_by_day = np.tile(w, (len(test_returns), 1))
    return {"daily_returns": daily, "weights_by_day": weights_by_day}


def weight_schedule_evaluation(test_returns: pd.DataFrame,
                               weights_by_day: np.ndarray) -> dict:
    """Evaluation from an explicit per-day weight matrix (e.g. rolling MVO)."""
    w = np.asarray(weights_by_day, dtype=np.float64)
    if w.shape != (len(test_returns), test_returns.shape[1]):
        raise ValueError(f"weights_by_day shape {w.shape} does not match "
                         f"test data {test_returns.shape}")
    daily = np.sum(test_returns.values * w, axis=1)
    return {"daily_returns": daily, "weights_by_day": w}


def evaluate_actor_with_weights(
    actor: torch.nn.Module,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    interval: int,
    clamp_negatives: bool = True,
) -> dict:
    """DPO evaluation mirroring utilities/metrics.RLEvaluator.evaluate_dpo,
    but returning the daily weight matrix alongside daily returns.

    The actor allocates once per `interval` days from the trailing
    `window_size` rows, and the allocation is held for the next interval.
    Only forecast_size=0 is supported (all current benchmarks use 0).
    """
    n_assets = test_data.shape[1]
    window_size = int(actor.input_size / n_assets)

    if interval < window_size:
        raise ValueError("interval must be >= actor input window size")

    merged = pd.concat((train_data.tail(window_size), test_data))
    num_intervals = len(test_data) // interval + 1

    actor.eval()

    daily_returns = []
    weights_rows = []

    for i in range(num_intervals):
        train_start = i * interval
        test_start = train_start + interval
        test_end = test_start + interval

        rolling_train = merged[train_start:test_start]
        rolling_test = merged[test_start:test_end]

        if len(rolling_test) == 0:
            continue

        with torch.no_grad():
            state = torch.tensor(
                rolling_train.tail(window_size).values, dtype=torch.float32
            ).flatten()
            allocation = actor(state).numpy().squeeze()

        if clamp_negatives:
            allocation = reduce_negatives(allocation)

        interval_returns = np.sum(rolling_test.values * allocation, axis=1)
        daily_returns.extend(interval_returns)
        weights_rows.extend([allocation] * len(rolling_test))

    return {
        "daily_returns": np.asarray(daily_returns, dtype=np.float64),
        "weights_by_day": np.asarray(weights_rows, dtype=np.float64),
    }


def evaluate_actor_spo_with_weights(
    actor: torch.nn.Module,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    clamp_negatives: bool = True,
) -> dict:
    """SPO evaluation mirroring RLEvaluator.evaluate_spo with weights."""
    n_assets = test_data.shape[1]
    window_size = int(actor.input_size / n_assets)

    actor.eval()
    with torch.no_grad():
        state = torch.tensor(
            train_data.tail(window_size).values, dtype=torch.float32
        ).flatten()
        allocation = actor(state).numpy().squeeze()

    if clamp_negatives:
        allocation = reduce_negatives(allocation)

    return constant_weights_evaluation(test_data, allocation)
