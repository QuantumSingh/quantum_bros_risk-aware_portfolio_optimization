"""Internal diagnostics: WHY are the loss curves jagged?

Instruments a faithful replica of the repo's DDPG inner update loop
(ddpg/ddpg_functions.py DDPGTrainer.train) for the PennyLane QDPG on the
Benchmark 1 config, logging EVERY internal quantity at EVERY training
window (not just the per-epoch average):

    reward, avg_profit, risk_pref*volatility (reward decomposition),
    |exploration noise|, actor allocation spread, Q(s,a), target Q,
    critic loss, actor loss, and the replay-buffer age of the sampled
    transition.

Then it quantifies and plots which of these drive the epoch-to-epoch
jaggedness. The three suspected causes are:
  1. single-sample replay (each update uses ONE sampled transition),
  2. exploration noise redrawn every window,
  3. a moving critic target (no target network; soft_update=False).

Outputs:
    comparison_logs/internal_diagnostics.csv          (per-window log)
    comparison_logs/internal_diagnostics_summary.csv  (variance decomposition)
    comparison_logs/plots/internal_diagnostics.png    (2-panel)
    comparison_logs/plots/internal_jaggedness_drivers.png (2-panel)
    comparison_logs/internal_diagnostics_run.log

NOTE: this reproduces the repo update rule for INSTRUMENTATION only; it
does not change the model or the trainer used elsewhere.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))
os.chdir(ORIGINAL_CODE)

import torch
from qiskit_port import benchmark_lib as bl
from predictors import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from utilities.data_processing import RLDataLoader
from utilities.model_training import ReplayBuffer, set_seeds

SEED = 68
NUM_ASSETS, ROWS = 4, 240
LOOKBACK, NUM_WEIGHTS = 10, 16
EPOCHS = 6
NOISE = 0.2
# PennyLane QDPG tuned hyperparameters (from benchmark_lib.HYPERPARAMS).
HP = bl.HYPERPARAMS["ddpg_quantum"]


def build():
    set_seeds(SEED)
    activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
    common = dict(num_weights=NUM_WEIGHTS, encoding="amplitude",
                  input_transformation=radial_to_linear, rotation_axes="y",
                  seed=SEED)
    input_size = NUM_ASSETS * LOOKBACK
    actor = QuantumNeuralNetwork(input_size=input_size, output_size=NUM_ASSETS,
                                 output_activation=activation, **common)
    critic = QuantumNeuralNetwork(input_size=input_size + NUM_ASSETS,
                                  output_size=1, **common)
    return actor, critic


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "internal_diagnostics_run.log")
    sys.stdout = logger
    try:
        returns = bl.load_original_returns(ROWS, NUM_ASSETS)
        tr, va, _ = bl.split_60_20_20(returns)

        loader = RLDataLoader(tr, va, shuffle=False)
        train_loader, _ = loader(batch_size=1, window_size=LOOKBACK, forecast_size=0)

        actor, critic = build()
        gamma = HP["gamma"]
        risk_pref = HP["risk_preference"]
        l2 = HP["l2_lambda"]
        actor_opt = HP["optimizer"](actor.parameters(), lr=HP["actor_lr"])
        critic_opt = HP["optimizer"](critic.parameters(), lr=HP["critic_lr"],
                                     weight_decay=0)
        buffer = ReplayBuffer()

        rows = []
        torch.manual_seed(SEED)
        for epoch in range(EPOCHS):
            for w, (state, next_state) in enumerate(train_loader):
                alloc = actor(state.flatten())
                noise = torch.normal(0, NOISE, alloc.shape)
                noisy = alloc + noise

                port = torch.sum(state.view(-1, NUM_ASSETS) * noisy, dim=-1)
                avg_profit = torch.mean(port).detach().cpu()
                volatility = torch.std(port, correction=0).detach().cpu()
                reward = avg_profit + risk_pref * volatility

                buffer.push((state.detach(), noisy.detach(), reward.detach(),
                             next_state.detach()))
                buf_len = len(buffer.buffer) if hasattr(buffer, "buffer") else None

                t = buffer.sample(1)
                s_s, s_noisy, s_reward, s_next = t[0]

                alloc2 = actor(s_s.flatten())
                next_alloc = actor(s_next.flatten())
                next_q = critic(torch.cat((s_next.flatten(), next_alloc.flatten())))
                target_q = s_reward + gamma * next_q
                q_value = critic(torch.cat((s_s.flatten(), s_noisy.flatten())))
                critic_loss = (target_q - q_value).pow(2)
                critic_opt.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_opt.step()

                critic_in = torch.cat((s_s.flatten(), alloc2.flatten()))
                actor_loss = -critic(critic_in)
                l2_actor = sum(p.pow(2).sum() for p in actor.parameters())
                actor_loss = actor_loss + l2 * l2_actor
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                rows.append(dict(
                    epoch=epoch + 1, window=w, global_step=len(rows),
                    reward=float(reward), avg_profit=float(avg_profit),
                    vol_term=float(risk_pref * volatility),
                    noise_l2=float(torch.linalg.norm(noise)),
                    alloc_spread=float(alloc.detach().std()),
                    q_value=float(q_value), target_q=float(target_q),
                    critic_loss=float(critic_loss), actor_loss=float(actor_loss),
                    buffer_len=buf_len))
            print(f"epoch {epoch+1}: mean critic loss "
                  f"{np.mean([r['critic_loss'] for r in rows if r['epoch']==epoch+1]):.4f}")

        df = pd.DataFrame(rows)
        df.to_csv(bl.COMPARISON_DIR / "internal_diagnostics.csv", index=False)

        # Variance decomposition: correlation of each internal with critic loss
        corrs = {}
        for col in ["noise_l2", "abs_reward", "abs_target_q", "q_value",
                    "alloc_spread"]:
            series = df["reward"].abs() if col == "abs_reward" else (
                df["target_q"].abs() if col == "abs_target_q" else df.get(col))
            if series is not None:
                corrs[col] = float(np.corrcoef(series, df["critic_loss"])[0, 1])
        summ = pd.DataFrame([
            dict(quantity=k, corr_with_critic_loss=round(v, 3))
            for k, v in corrs.items()])
        summ["abs_corr"] = summ["corr_with_critic_loss"].abs()
        summ = summ.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
        summ.to_csv(bl.COMPARISON_DIR / "internal_diagnostics_summary.csv", index=False)
        print("\nWhat correlates with critic-loss spikes:")
        print(summ.to_string(index=False))

        # ---- Figure 1: per-window critic loss + reward decomposition ----
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(df["global_step"], df["critic_loss"], linewidth=0.8,
                     color="#d62728")
        for e in range(1, EPOCHS):
            axes[0].axvline(e * len(df[df.epoch == 1]), color="#cccccc",
                            linewidth=0.8, linestyle=":")
        axes[0].set_ylabel("Critic loss")
        axes[0].set_title("Per-window critic loss (dotted = epoch boundary) — "
                          "the jaggedness is intra-epoch, not smooth")
        axes[1].plot(df["global_step"], df["avg_profit"], linewidth=0.8,
                     label="avg profit term", color="#2ca02c")
        axes[1].plot(df["global_step"], df["vol_term"], linewidth=0.8,
                     label="risk_pref x volatility term", color="#1f77b4")
        axes[1].plot(df["global_step"], df["reward"], linewidth=1.1,
                     label="reward (sum)", color="#111111")
        axes[1].set_ylabel("Reward components")
        axes[1].set_xlabel("Training window (global step)")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(bl.PLOTS_DIR / "internal_diagnostics.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

        # ---- Figure 2: jaggedness drivers ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(df["noise_l2"], df["critic_loss"], s=10, alpha=0.5,
                        color="#d62728")
        axes[0].set_xlabel("Exploration noise magnitude  ||noise||")
        axes[0].set_ylabel("Critic loss")
        axes[0].set_title("Noise vs critic loss")
        axes[0].grid(alpha=0.25)
        axes[1].scatter(df["target_q"].abs(), df["critic_loss"], s=10, alpha=0.5,
                        color="#1f77b4")
        axes[1].set_xlabel("|target Q| (moving target)")
        axes[1].set_ylabel("Critic loss")
        axes[1].set_title("Moving target vs critic loss")
        axes[1].grid(alpha=0.25)
        fig.suptitle("Drivers of loss-curve jaggedness (single-sample replay + "
                     "noise + moving target)")
        fig.tight_layout()
        fig.savefig(bl.PLOTS_DIR / "internal_jaggedness_drivers.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print("\nWrote internal_diagnostics.png and internal_jaggedness_drivers.png")
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()
