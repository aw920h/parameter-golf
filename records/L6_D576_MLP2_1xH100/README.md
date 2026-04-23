# Architectural Ablation Studies for OpenAI Parameter Golf

**Submitted by:** SHIVANG ([@aw920h](https://github.com/aw920h))

This repository documents a series of single-GPU ablation studies aimed at developing a highly optimized architecture for the OpenAI Parameter Golf challenge (10-minute / 16MB track). The experiments systematically map the trade-offs between model depth, width, memory expansion, and advanced optimization techniques to identify the most performant and legally compliant configuration before scaling to the official 8xH100 evaluation environment.

---

## Summary & Architecture

Through rigorous testing, the optimal architecture was identified as a **6-layer, 576-dimension "Shallow-Wide" Transformer**. This configuration achieves a `val_bpb` of **1.2823** while maintaining a safe artifact size of **14.88 MB**.

This architecture is the proposed candidate for the final multi-seed 8xH100 submission. It is projected to achieve a score well into the `1.1x` range when scaled to the full multi-GPU step count.

---

## Experimental Results Comparison

The following table summarizes the three most significant single-GPU (NVIDIA H200) runs, showcasing the journey from the baseline to the final proposed architecture.

| Experiment Name         | Architecture (L/D/MLP) | `val_bpb` (Score) | Artifact Size |
|-------------------------|------------------------|-------------------|---------------|
| **Bwst Test**       | `L=5, D=640, MLP=1.875`| `1.2659`       | `16.16 MB`    | 
| **Best Valid Test**   | `L=6, D=576, MLP=2`    | `1.2823`        | `14.88 MB`    | 

---

## Analysis of Key Experiments

### 1. The Boundary Test (`val_bpb: 1.2659`, Disqualified)

This experiment was designed to find the absolute intelligence ceiling of a shallow architecture. By pushing the model width to **640 dimensions** and using 5 layers for maximum speed, the model achieved an exceptional score of `1.2659`.

*   **Conclusion:** This proved that a "Shallow-Wide" approach is highly effective. However, the resulting **16.16 MB** artifact size breached the 16,000,000-byte hard limit, making the architecture illegal. This run successfully mapped the physical size constraint of the `D=640` configuration.

### 2. The Proposed Champion (`val_bpb: 1.2823`, Legal)

This configuration represents the optimal balance of intelligence, speed, and budget compliance.

*   **Architecture:** By reducing the model dimension to **576**, the parameter count was brought safely within the legal limit. The loss of width was compensated by adding an extra physical layer (`NUM_LAYERS=6`).
*   **Performance:** This build achieved a `val_bpb` of **1.2823** while weighing only **14.88 MB**, leaving a safe 1.12 MB overhead.
*   **Advanced Techniques:** This run successfully integrates the full stack of optimizations:
    *   `SP8192` Vocabulary
    *   Layer Recurrence (3x loop on the middle layer)
    *   Brotli Compression (Quality 11) with Weight Decay (0.095)
    *   An EMA "Bake-In" strategy at step 1450 to stabilize weights against quantization penalties.

---

## Final Proposed Architecture for 8xH100 Cluster

The **`L=6, D=576` Champion** build is the final proposed architecture. Its combination of a legally compliant file size, proven high performance, and architectural stability makes it the ideal candidate for scaling. The full command used for this run is documented below for reproducibility.

### Hardware & Command for Champion Run

*   **GPU:** 1x NVIDIA H100
*   **Command:**
    ```bash
    RUN_ID=lightning_champion_run \
    DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
    VOCAB_SIZE=8192 \
    NUM_LAYERS=6 \
    MODEL_DIM=576 \
    NUM_HEADS=9 \
    NUM_KV_HEADS=3 \
    MATRIX_LR=0.04 \
    QK_GAIN_INIT=5.25 \
    MUON_MOMENTUM_WARMUP_STEPS=100 \
    LOGIT_SOFTCAP=15.0 \
    ITERATIONS=1600 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
    ```
