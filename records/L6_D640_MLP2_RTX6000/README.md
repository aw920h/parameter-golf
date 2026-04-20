# Single-GPU Profile: Shallow-Wide (L6/D640/MLP2) Architecture

**Submitted by:** Shivang ([@aw920h](https://github.com/aw920h))

This folder contains the logs and configuration for single-seed profiling run of "Shallow and Wide" architecture for the OpenAI Parameter Golf challenge. This configuration was identified as the optimal balance between model size, compute throughput, and performance on a single GPU.

---

## Final Results (Single Seed)

| Metric           | Value          |
|------------------|----------------|
| **`val_bpb`**    | **1.3400**     |
| `artifact_bytes` | 13,808,941 B   |
| `steps_completed`| 1323 / 20000   |
| `avg_step_time`  | 453.72 ms      |

---

## Hardware Configuration

The experiment was conducted on a machine with the following specifications:

*   **GPU:** 1x NVIDIA RTX 6000 Pro (96GB vRAM)
*   **CPU:** 16 vCPUs
*   **System RAM:** 512 GB

---

## Command Used

The following command was used to execute this training run. It modifies the baseline hyperparameters to create a shallower (6 layers), wider (640 dimension), and `MLP_MULT=2` model.

```bash
NUM_LAYERS=6 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_MULT=2 \
python3 train_gpt.py
```

---

## Architectural Summary

This configuration represents the optimized architecture from a series of single-GPU ablation studies. It successfully balances the trade-off between model intelligence and the constraints.

*   **Depth-to-Width Ratio:** Layers were reduced from the baseline's 9 to **6**, while the model dimension was increased from 512 to **640**. This resulted in a faster per-step compute time (453ms vs. the baseline's ~672ms), allowing for more optimization steps within the time limit.

*   **Hardware Alignment:** Grouped-Query Attention was configured with **10 heads** and **5 KV heads**, ensuring the head dimension (64) is optimal for leveraging the Tensor Cores.

*   **Size Compliance:** With `MLP_MULT=2`, the resulting int8+zlib artifact size is **13.80 MB**. This is well under the 16.0 MB competition limit, providing a significant safety margin. This configuration was chosen over a larger `MLP_MULT=3` model which, while scoring slightly better (1.3388), nut exceeding the 16.0 MB limit.

