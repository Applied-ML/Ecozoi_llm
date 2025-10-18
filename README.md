## Ecozoi Experiments: Enformer Variants (Komlós, RoPE, Rank-1)

### Overview
This repository runs a suite of experiments on DeepMind's Enformer architecture with three orthogonal modifications:
- Komlós sequence encoding (3-channel balanced encoding of DNA)
- RoPE: Rotary Positional Embeddings in self-attention
- Rank-1 value projection in attention (parameter/min-rank constraint)

evaluated performance with Pearson correlation (PearsonR) across human assay tracks, aggregating results across multiple random three seeds for statistical stability.

Key scripts:
- `run_experiments.py`: end-to-end training, evaluation, aggregation, plotting, and saving artifacts.
- `ecozoi_final.py`: Enformer implementation with toggles for Komlós, RoPE, and Rank-1.


## Model and Variants
Base model: Enformer with truncation to 896 target bins at 128 bp resolution for a 196,608 bp input sequence.

Experiments (8 total):
1. `vanilla`: baseline (no Komlós, no RoPE, no Rank-1)
2. `komlos`: Komlós encoding only
3. `rope`: RoPE only
4. `rank1`: Rank-1 value projection only
5. `komlos_rope`: Komlós + RoPE
6. `komlos_rank1`: Komlós + Rank-1
7. `rope_rank1`: RoPE + Rank-1
8. `all`: Komlós + RoPE + Rank-1

Heads: `{'human': 5313, 'mouse': 1643}` (experiments below focus on human; 5,313 tracks).


## Theory (short primers)
### Komlós encoding (balanced 3D representation)
Standard one-hot encodes DNA as 4 channels (A,C,G,T). Komlós encodes into 3 balanced dimensions (linear transforms of A,C,G,T) to reduce redundancy and improve conditioning. This preserves information while centering and balancing symbol contributions, often stabilizing optimization.

### Rotary Positional Embeddings (RoPE)
RoPE applies a rotation in a subspace of each query/key vector proportional to absolute position and relative offsets. It enables attention to represent relative positions naturally without adding external positional vectors, improving long-range generalization and extrapolation.

### Rank-1 value projection in attention
Standard multi-head attention uses full-dimensional value projections per head. Rank-1 constrains value projections to dimension 1 (or minimal) per head (optionally pairing with a small key dimension when RoPE is active). This reduces parameters and may act as a regularizer while relying on attention mixing for expressivity.

### PearsonR (evaluation metric)
Pearson correlation between predictions and targets is computed using streaming sufficient statistics. In our pipeline, per-batch predictions/targets with shape `(batch, bins=896, tracks=5313)` are accumulated; final reductions yield per-track or category-aggregated correlation. We report mean PearsonR across the specified target sets.


## Data
- Format: TFRecords with ZLIB compression; examples contain:
  - `sequence`: DNA sequence (one-hot or Komlós-encoded), length 196,608 bp
  - `target`: assay targets of shape `(896 bins, 5313 tracks)` for human
- Training set: 5 TFRecord files × 256 sequences/file = 1,280 sequences total (each sequence is 196,608 bp).
- Validation set: uses `max_steps=256` for evaluation sampling.

Note: `run_experiments.py` reads from `data_limited/<organism>/tfrecords` via helpers and applies mapping (identity or Komlós) on-the-fly.


## Training and Evaluation Protocol
### Hyperparameters (from `run_experiments.py`)
- `num_epochs = 20`
- `steps_per_epoch = 5`
- `num_warmup_steps = 50` (linear warmup)
- `target_learning_rate = 5e-4` (Adam optimizer via Sonnet)
- Batch size: 1 (streaming iterator over TFRecords)
- Seeds: `[42, 123, 2025]` (full 8-experiment suite repeated per seed)

For each seed:
1. Set global seeds for Python, NumPy, TensorFlow, and PyTorch (ensures identical initialization across the 8 experiments for that seed).
2. For each of the 8 variants, build model with the requested toggles, train for the configured epochs and steps, then evaluate on validation with `max_steps=256`.
3. Save per-experiment artifacts (checkpoints and SavedModel) and metrics.

Aggregation across seeds:
- Save per-seed metrics (`experiment_results_human_seed_<seed>.csv`).
- Aggregate all rows (`experiment_results_human_all_seeds.csv`).
- Compute experiment-wise averages over seeds (`experiment_results_human_avg.csv`).
- Produce plots including mean PearsonR bars and category summaries.


## Outputs and Artifacts
All results are written to `output_results/`:
- CSVs:
  - `experiment_results_human_seed_<seed>.csv`
  - `experiment_results_human_all_seeds.csv`
  - `experiment_results_human_avg.csv`
  - `experiment_results_human_categories.csv`
- Plots in `output_results/plots/`:
  - `pearson_overall_human.png` (mean over seeds)
  - `pearson_by_category_human.png`
  - `scatter_vanilla_vs_all_*.png`




## Interpreting Results
- The primary scalar for each experiment is `pearson_human_mean` (mean PearsonR across the human tracks or selected categories).
- Category summaries (`DNASE`, `CAGE`, `CHIP`, `ATAC`) are computed by averaging PearsonR over index ranges derived from `targets.txt`.
- Plots compare baseline vs. modified variants and highlight per-category deltas; scatter plots visualize pointwise improvements (e.g., `vanilla` vs. `all`).

## Visualization

#### 1. Overall Performance Plot
Bar chart showing mean Pearson correlation across all 5,313 targets for each experiment.

![Overall Performance](plots/pearson_overall_human.png)

#### 2. Category Performance Plot
Bar chart showing mean Pearson correlation for each category (DNASE, CAGE, CHIP, ATAC) across experiments.

![Category Performance](plots/pearson_by_category_human.png)

#### 3. Scatter Plots (Vanilla vs All)
Per-category scatter plots comparing individual target correlations:

**DNASE Category:**
![DNASE Scatter](plots/scatter_vanilla_vs_all_DNASE.png)

**CAGE Category:**
![CAGE Scatter](plots/scatter_vanilla_vs_all_CAGE.png)

**CHIP Category:**
![CHIP Scatter](plots/scatter_vanilla_vs_all_CHIP.png)

**ATAC Category:**
![ATAC Scatter](plots/scatter_vanilla_vs_all_ATAC.png)


## Notes and Reproducibility
- Seeding ensures that, for a given seed, all 8 experiments share identical initial weights; cross-seed averaging reduces variance.
- Saved checkpoints and SavedModels enable later inference and fine-grained analysis without retraining.
- Input resolution: 128 bp per bin; target length: 896 bins; input sequence length: 196,608 bp.
- Human head size: 5,313 tracks.


## Citation



```bibtex
@article{avsec2021effective,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, {\v{Z}}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwi{\'n}ska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature methods},
  volume={18},
  number={10},
  pages={1196--1203},
  year={2021},
  publisher={Nature Publishing Group}
}
```




