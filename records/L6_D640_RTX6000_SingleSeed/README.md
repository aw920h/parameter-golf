# Single-GPU Profile: Shallow-Wide (L6/D640) Architecture

**Submitted by:** Shivang ([@aw920h](https://github.com/aw920h))

This folder contains the logs and configuration for single-seed profiling run of modified "Shallow and Wide" architecture for the OpenAI Parameter Golf challenge. The purpose of this run was to establish a baseline for throughput and performance on a single high-end GPU before scaling to the official 8xH100 evaluation environment.

---

## Final Results (Single Seed)

| Metric           | Value          |
|------------------|----------------|
| **`val_bpb`**    | **1.3429**     |
| `artifact_bytes` | 14,395,665 B   |
| `steps_completed`| 1322 / 20000   |
| `avg_step_time`  | 454.17 ms      |

---

## Hardware Configuration

*   **GPU:** 1x NVIDIA RTX 6000 Pro (96GB vRAM)
*   **CPU:** 16 vCPUs
*   **System RAM:** 512 GB

---

## Command Used

The following command was used to execute this training run. It modifies the baseline hyperparameters to create a shallower (6 layers) but wider (640 dimension) model.

```bash
NUM_LAYERS=6 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MATRIX_LR=0.05 \
python3 train_gpt.py
```

---

## Architectural Summary

The goal of this configuration was to trade model depth for width, hypothesizing that a faster per-step compute time would allow the model to process more tokens within the 10-minute wall-clock limit, leading to a better loss score.

*   **Depth Reduction:** Layers were reduced from the baseline's 9 to **6**.
*   **Width Expansion:** The model dimension was increased from 512 to **640**.
*   **Hardware Alignment:** Grouped-Query Attention was configured with **10 heads** and **5 KV heads**, ensuring the head dimension (64) is optimal for Tensor Core utilization.
*   **Size Compliance:** The resulting int8+zlib artifact size is **14.39 MB**, which is under the 16.0 MB competition limit.

This single-seed run confirms that the shallow-wide architecture is a promising direction, significantly outperforming the baseline `val_bpb` of 1.3736 on **NVIDIA A100-SXM4-80GB**. The next step is to secure an 8xH100 cluster to perform the official multi-seed validation required for a formal leaderboard submission.
