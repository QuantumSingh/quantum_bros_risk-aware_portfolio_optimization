# Paper Terminology

## Dynamic Portfolio Optimization

The model changes portfolio weights over time instead of choosing one fixed portfolio forever.

## Rebalancing

Updating portfolio weights at fixed intervals. In this code, `DPO_INTERVAL = 30`, meaning the portfolio is rebalanced every 30 trading days.

## State

The information given to the RL agent before it makes a decision. In this project, this comes from recent asset price data and forecast features.

## Action

The portfolio allocation chosen by the agent. For DDPG/QDPG, the action is a vector of portfolio weights.

## Reward

The feedback signal used to train the agent. It tells the model whether its portfolio decision was good or bad.

## DDPG

Deep Deterministic Policy Gradient. A reinforcement learning algorithm for continuous actions, which makes sense because portfolio weights are continuous.

## QDPG

Quantum version of DDPG. The model uses a variational quantum circuit as the predictor/actor component.

## VQC

Variational Quantum Circuit. A quantum circuit with trainable parameters.

## Amplitude Encoding

A way of loading a classical vector into the amplitudes of a quantum state. If there are `n` qubits, the circuit can represent a vector of length `2^n`.

## Parameter Efficiency

The quantum model uses fewer trainable parameters than a large classical neural network. In this project, QDPG uses `num_weights=60`.

## Sharpe Ratio

A risk-adjusted return metric measuring return relative to volatility.

## Low-Frequency Portfolio Allocation

This is not high-frequency trading. The model makes periodic allocation decisions, such as every 30 trading days.
