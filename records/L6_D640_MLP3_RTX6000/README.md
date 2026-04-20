# Shallow-Wide (L6/D640/MLP3) Architecture

**Submitted by:** Shivang ([@aw920h](https://github.com/aw920h))

(Exceeds 16,000,000 byte limit)

This folder contains the logs and configuration for a single-seed boundary test of the "Shallow and Wide" architecture. The purpose of this run was to push the Feed-Forward Network (FFN) to its maximum capacity to test the ceiling before optimizing for the 16.0 MB artifact constraint.

---

## Final Results (Single Seed)

| Metric           | Value          | 
|------------------|----------------|
| **`val_bpb`**    | **1.3388**     | 
| `artifact_bytes` | 16,689,037 B   | 
| `steps_completed`| 1169 / 20000   | 
| `avg_step_time`  | 513.56 ms      | 

---

## Hardware Configuration

The experiment was conducted on a machine with the following specifications:

*   **GPU:** 1x NVIDIA RTX 6000 Pro (96GB vRAM)
*   **CPU:** 16 vCPUs
*   **System RAM:** 512 GB

---

## Command Used

The following command was used to execute this training run. It modifies the baseline to create a shallower (6 layers) and wider (640 dimension) model, while expanding the memory bank (`MLP_MULT=3`).

```bash
NUM_LAYERS=6 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_MULT=3 \
MATRIX_LR=0.04 \
python3 train_gpt.py
```

---

## Architectural Summary

This experiment was designed to test the parameter limits of the Int8+zlib quantization pipeline. 

*   **Intelligence Gain:** By expanding the MLP multiplier from 2 to **3**, the hidden memory bank size increased by 50% per layer. The AI successfully utilized this extra parameter density to drop the `val_bpb` to an impressive **1.3388**, despite running fewer steps (1169) than lighter models.
*   **Size Constraint:** The mathematical cost of expanding the FFN pushed the serialized int8+zlib file to **16.68 MB**. Because the competition strictly caps the artifact size at 16,000,000 bytes (decimal), this architecture is mathematically illegal.

