#!/usr/bin/env python3
"""
Ecozoi Experiments Runner
========================
This script runs the Enformer experiments with different configurations (vanilla, komlos, rope, rank1, etc.)
and generates comparison plots and results.

"""

import os
import sys
import time
import logging
from datetime import datetime

# Set up logging for progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecozoi_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_progress(message):
    """Log progress with timestamp"""
    logger.info(f"[PROGRESS] {message}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


try:
    import math
    import numpy as np
    import tensorflow as tf
    import sonnet as snt
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import glob
    import json
    import functools
    
    
    
    from ecozoi_final import get_experiment
    
    log_progress("All imports successful")
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Reproducibility helpers
def set_global_seed(seed):
    """Set seeds for numpy, tensorflow, torch, and python to stabilize init."""
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# Dataset and utility functions
def get_targets(organism):
    """Load target metadata for organism"""
    targets_txt = f'data_limited/{organism}/targets.txt'
    return pd.read_csv(targets_txt, sep='\t')

def organism_path(organism):
    return os.path.join('data_limited', organism)

def get_dataset(organism, subset, num_threads=8):
    """Load TFRecord dataset for organism and subset"""
    metadata = get_metadata(organism)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads)
    dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                          num_parallel_calls=num_threads)
    return dataset

def get_metadata(organism):
    """Load metadata for organism"""
    path = os.path.join(organism_path(organism), 'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)

def tfrecord_files(organism, subset):
    """Get sorted list of TFRecord files for organism and subset"""
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
    )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile"""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence': sequence, 'target': target}

# Training and evaluation functions
def create_step_function(model, optimizer):
    """Create training step function"""
    @tf.function
    def train_step(batch, head, optimizer_clip_norm_global=0.2):
        with tf.GradientTape() as tape:
            outputs = model(batch['sequence'], is_training=True)[head]
            loss = tf.reduce_mean(
                tf.keras.losses.poisson(batch['target'], outputs))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return loss
    return train_step

# Metrics classes
def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])

class CorrelationStats(tf.keras.metrics.Metric):
    """Contains shared code for PearsonR and R2"""
    def __init__(self, reduce_axis=None, name='pearsonr'):
        super(CorrelationStats, self).__init__(name=name)
        self._reduce_axis = reduce_axis
        self._shape = None

    def _initialize(self, input_shape):
        self._shape = _reduced_shape(input_shape, self._reduce_axis)
        weight_kwargs = dict(shape=self._shape, initializer='zeros')
        self._count = self.add_weight(name='count', **weight_kwargs)
        self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
        self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
        self._true_squared_sum = self.add_weight(name='true_squared_sum', **weight_kwargs)
        self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
        self._pred_squared_sum = self.add_weight(name='pred_squared_sum', **weight_kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._shape is None:
            self._initialize(y_true.shape)
        y_true.shape.assert_is_compatible_with(y_pred.shape)
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        self._product_sum.assign_add(tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))
        self._true_sum.assign_add(tf.reduce_sum(y_true, axis=self._reduce_axis))
        self._true_squared_sum.assign_add(tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))
        self._pred_sum.assign_add(tf.reduce_sum(y_pred, axis=self._reduce_axis))
        self._pred_squared_sum.assign_add(tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))
        self._count.assign_add(tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    def reset_states(self):
        if self._shape is not None:
            tf.keras.backend.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])

class PearsonR(CorrelationStats):
    """Pearson correlation coefficient"""
    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        super(PearsonR, self).__init__(reduce_axis=reduce_axis, name=name)

    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count
        covariance = (self._product_sum - true_mean * self._pred_sum - pred_mean * self._true_sum + 
                     self._count * true_mean * pred_mean)
        true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
        pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
        tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation

class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}

def evaluate_model(model, dataset, head, max_steps=None):
    """Evaluate model on dataset"""
    metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
    
    @tf.function
    def predict(x):
        return model(x, is_training=False)[head]

    for i, batch in tqdm(enumerate(dataset), desc=f"Evaluating {head}"):
        if max_steps is not None and i > max_steps:
            break
        metric.update_state(batch['target'], predict(batch['sequence']))

    return metric.result()

def plot_category_scatter(cat_name, idx_list, filename, vanilla_r, all_r):
    """Plot scatter plot for category comparison"""
    x = vanilla_r[idx_list]
    y = all_r[idx_list]

    plt.figure(figsize=(5.5, 5.5))
    sns.scatterplot(x=x, y=y, s=10, alpha=0.5, edgecolor=None)

    # y=x reference
    minv = float(np.nanmin(np.concatenate([x, y])))
    maxv = float(np.nanmax(np.concatenate([x, y])))
    pad = 0.02 * (maxv - minv + 1e-8)
    lo, hi = minv - pad, maxv + pad
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y = x')

    # Per-category means
    cat_mean_vanilla = float(np.nanmean(x))
    cat_mean_all = float(np.nanmean(y))
    plt.text(0.02, 0.98,
             f"{cat_name} mean r (vanilla): {cat_mean_vanilla:.4f}\n{cat_name} mean r (all): {cat_mean_all:.4f}",
             ha='left', va='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))

    plt.xlabel('PearsonR (vanilla)')
    plt.ylabel('PearsonR (all)')
    plt.title(f'{cat_name}: per-target PearsonR (N={len(idx_list)})')
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()  
    log_progress(f"Saved scatter plot: {filename}")

def main():
    """Main experiment runner"""
    start_time = time.time()
    log_progress("Starting Ecozoi experiments")
    
    #Configuration
    # experiment_names = ["vanilla", "komlos", "rope", "rank1",
    #                     "komlos_rope", "komlos_rank1", "rope_rank1", "all"]

    experiment_names = ['vanilla','all']
    
    #seeds = [42, 123, 2025]
    seeds=[42]
    
    # Output directories
    output_dir = os.path.join(os.getcwd(), 'output_results')
    plots_dir = os.path.join(output_dir, 'plots')
    ckpt_root = os.path.join(output_dir, 'checkpoints')
    savedmodel_root = os.path.join(output_dir, 'savedmodels')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(savedmodel_root, exist_ok=True)
    
    
    try:
        fh = logging.FileHandler(os.path.join(output_dir, 'ecozoi_experiments.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    except Exception:
        pass
    
    # Training parameters (adjust as neede)
    steps_per_epoch = 5
    num_epochs = 60
    num_warmup_steps = 50
    target_learning_rate = 5e-4
    
    log_progress(f"Running {len(experiment_names)} experiments: {experiment_names}")
    log_progress(f"Training config: {num_epochs} epochs, {steps_per_epoch} steps/epoch")
    
    
    log_progress("Loading target metadata...")
    df_targets_human = get_targets('human')
    indices_dnase = df_targets_human.index[df_targets_human['description'].str[:4] == 'DNAS'].tolist()
    indices_cage = df_targets_human.index[df_targets_human['description'].str[:4] == 'CAGE'].tolist()
    indices_chip = df_targets_human.index[df_targets_human['description'].str[:4] == 'CHIP'].tolist()
    indices_atac = df_targets_human.index[df_targets_human['description'].str[:4] == 'ATAC'].tolist()
    
    log_progress(f"Target counts - DNASE: {len(indices_dnase)}, CAGE: {len(indices_cage)}, "
                f"CHIP: {len(indices_chip)}, ATAC: {len(indices_atac)}")
    
    
    all_results_by_seed = {}
    all_metrics_by_seed = {}
    
    for seed in seeds:
        log_progress(f"=== Starting experiments for seed {seed} ===")
        set_global_seed(seed)
        results = []
        all_metrics_human = {}
        
        for exp_idx, name in enumerate(experiment_names):
            exp_start = time.time()
            log_progress(f"[seed {seed}] Starting experiment {exp_idx+1}/{len(experiment_names)}: {name}")
            
            try:
                
                set_global_seed(seed)
                
                
                mapper, model = get_experiment(
                    name,
                    channels=1536,
                    num_heads=8,
                    num_transformer_layers=11,
                    pooling_type='max'
                )
                log_progress(f"[seed {seed}] Model {name} created successfully")
                
                
                human_train = get_dataset('human', 'train').batch(1).map(mapper).repeat().prefetch(2)
                
                
                learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
                optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
                train_step = create_step_function(model, optimizer)
                
               
                data_it = iter(human_train)
                global_step = 0
                loss_human = None
                
                log_progress(f"[seed {seed}] Starting training for {name}...")
                for epoch_i in range(num_epochs):
                    epoch_start = time.time()
                    for step in tqdm(range(steps_per_epoch), desc=f"{name} (seed {seed}) epoch {epoch_i+1}/{num_epochs}"):
                        global_step += 1
                        if global_step > 1:
                            lr_frac = tf.math.minimum(1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))
                            learning_rate.assign(target_learning_rate * lr_frac)
                        
                        batch_human = next(data_it)
                        loss_human = train_step(batch=batch_human, head='human')
                    
                    epoch_time = time.time() - epoch_start
                    log_progress(f"[seed {seed}] {name} epoch {epoch_i+1} completed in {epoch_time:.1f}s, loss: {float(loss_human.numpy()):.4f}")
                
                # Evaluation
                log_progress(f"[seed {seed}] Starting evaluation for {name}...")
                human_valid = get_dataset('human', 'valid').batch(1).map(mapper).prefetch(2)
                metrics_human = evaluate_model(model, human_valid, head='human', max_steps=256)
                
                pearson_human = metrics_human['PearsonR'].numpy()
                all_metrics_human[name] = pearson_human
                
                # Save weights/checkpoints and SavedModel for later inference
                exp_ckpt_dir = os.path.join(ckpt_root, f'seed_{seed}', name)
                exp_sm_dir = os.path.join(savedmodel_root, f'seed_{seed}', name)
                os.makedirs(exp_ckpt_dir, exist_ok=True)
                os.makedirs(os.path.dirname(exp_sm_dir), exist_ok=True)
                try:
                    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
                    ckpt_path = ckpt.save(os.path.join(exp_ckpt_dir, 'ckpt'))
                    log_progress(f"[seed {seed}] Saved checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.error(f"[seed {seed}] Failed saving checkpoint for {name}: {e}")
                try:
                    # Export SavedModel for inference
                    tf.saved_model.save(model, exp_sm_dir)
                    log_progress(f"[seed {seed}] Saved SavedModel: {exp_sm_dir}")
                except Exception as e:
                    logger.error(f"[seed {seed}] Failed saving SavedModel for {name}: {e}")
                
                results.append({
                    'experiment': name,
                    'seed': seed,
                    'train_loss_human': float(loss_human.numpy()) if loss_human is not None else np.nan,
                    'pearson_human_mean': float(np.nanmean(pearson_human)),
                })
                
                exp_time = time.time() - exp_start
                log_progress(f"[seed {seed}] Experiment {name} completed in {exp_time:.1f}s, mean PearsonR: {float(np.nanmean(pearson_human)):.4f}")
                
            except Exception as e:
                logger.error(f"[seed {seed}] Error in experiment {name}: {e}")
                # Continue with next experiment
                continue
        
        
        all_results_by_seed[seed] = results
        all_metrics_by_seed[seed] = all_metrics_human
        df_seed = pd.DataFrame(results).set_index('experiment').sort_index()
        seed_csv = os.path.join(output_dir, f'experiment_results_human_seed_{seed}.csv')
        df_seed.to_csv(seed_csv, index=True)
        log_progress(f"Saved per-seed results: {seed_csv}")
    
    
    log_progress("Generating results and plots...")
    
    
    log_progress("Aggregating results across seeds...")
    df_all = pd.concat([pd.DataFrame(r) for r in all_results_by_seed.values()], ignore_index=True)
    
    df_avg = (df_all.groupby('experiment', as_index=True)
                    .agg(pearson_human_mean=('pearson_human_mean', 'mean'),
                         train_loss_human=('train_loss_human', 'mean'))
                    .sort_index())
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS (average over seeds)")
    print("="*50)
    print(df_avg)
    print("="*50 + "\n")
    
    
    df_all.set_index(['seed','experiment']).sort_index().to_csv(os.path.join(output_dir, 'experiment_results_human_all_seeds.csv'))
    df_avg.to_csv(os.path.join(output_dir, 'experiment_results_human_avg.csv'), index=True)
    log_progress("Saved experiment_results_human_all_seeds.csv and experiment_results_human_avg.csv")
    
    # Overall PearsonR plot
    plt.figure(figsize=(6,4))
    sns.barplot(x=df_avg.index, y='pearson_human_mean', data=df_avg.reset_index(), color='#4C72B0')
    plt.title('Human PearsonR (mean over seeds)')
    plt.ylabel('PearsonR')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pearson_overall_human.png'), dpi=200)
    plt.close()
    log_progress("Saved pearson_overall_human.png")
    
    
    per_seed_cat = []
    for seed in seeds:
        metrics_seed = all_metrics_by_seed.get(seed, {})
        for name in experiment_names:
            if name in metrics_seed:
                ph = metrics_seed[name]
                per_seed_cat.append({
                    'seed': seed,
                    'experiment': name,
                    'DNASE_mean': float(np.nanmean(ph[indices_dnase[0]:indices_dnase[-1]+1])) if len(indices_dnase)>0 else np.nan,
                    'CAGE_mean': float(np.nanmean(ph[indices_cage[0]:indices_cage[-1]+1])) if len(indices_cage)>0 else np.nan,
                    'CHIP_mean': float(np.nanmean(ph[indices_chip[0]:indices_chip[-1]+1])) if len(indices_chip)>0 else np.nan,
                    'ATAC_mean': float(np.nanmean(ph[indices_atac[0]:indices_atac[-1]+1])) if len(indices_atac)>0 else np.nan,
                })
    if per_seed_cat:
        df_cat_all = pd.DataFrame(per_seed_cat)
        df_human_cats = (df_cat_all.groupby('experiment', as_index=True)
                          .agg(DNASE_mean=('DNASE_mean','mean'),
                               CAGE_mean=('CAGE_mean','mean'),
                               CHIP_mean=('CHIP_mean','mean'),
                               ATAC_mean=('ATAC_mean','mean'))
                          .sort_index())
    else:
        df_human_cats = pd.DataFrame(columns=['DNASE_mean','CAGE_mean','CHIP_mean','ATAC_mean'])
    print("\n" + "="*50)
    print("CATEGORY RESULTS")
    print("="*50)
    print(df_human_cats)
    print("="*50 + "\n")
    
    df_human_cats.to_csv(os.path.join(output_dir, 'experiment_results_human_categories.csv'), index=True)
    log_progress("Saved experiment_results_human_categories.csv (averaged over seeds)")
    
    # Per-category plot
    df_melt = df_human_cats.reset_index().melt(id_vars='experiment', var_name='category', value_name='PearsonR')
    plt.figure(figsize=(10,5))
    sns.barplot(data=df_melt, x='experiment', y='PearsonR', hue='category')
    plt.title('Human PearsonR by category')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pearson_by_category_human.png'), dpi=200)
    plt.close()
    log_progress("Saved pearson_by_category_human.png")
    
    
    first_seed_with_metrics = next((s for s in seeds if s in all_metrics_by_seed and 'vanilla' in all_metrics_by_seed[s] and 'all' in all_metrics_by_seed[s]), None)
    if first_seed_with_metrics is not None:
        log_progress(f"Generating scatter plots (seed {first_seed_with_metrics})...")
        vanilla_r = all_metrics_by_seed[first_seed_with_metrics]['vanilla']
        all_r = all_metrics_by_seed[first_seed_with_metrics]['all']
        
        plot_category_scatter('DNASE', indices_dnase, os.path.join(plots_dir, 'scatter_vanilla_vs_all_DNASE.png'), vanilla_r, all_r)
        plot_category_scatter('CAGE', indices_cage, os.path.join(plots_dir, 'scatter_vanilla_vs_all_CAGE.png'), vanilla_r, all_r)
        plot_category_scatter('CHIP', indices_chip, os.path.join(plots_dir, 'scatter_vanilla_vs_all_CHIP.png'), vanilla_r, all_r)
        plot_category_scatter('ATAC', indices_atac, os.path.join(plots_dir, 'scatter_vanilla_vs_all_ATAC.png'), vanilla_r, all_r)
    
    # Final summary
    total_time = time.time() - start_time
    log_progress(f"All experiments completed in {total_time/3600:.1f} hours")
    log_progress("Generated files (under output_results/):")
    log_progress("- experiment_results_human_all_seeds.csv")
    log_progress("- experiment_results_human_avg.csv")
    log_progress("- experiment_results_human_categories.csv")
    log_progress("- plots/pearson_overall_human.png")
    log_progress("- plots/pearson_by_category_human.png")
    log_progress("- plots/scatter_vanilla_vs_all_*.png")
    log_progress("- ecozoi_experiments.log")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total runtime: {total_time/3600:.1f} hours")
    print("Check the generated files and log for results.")
    print("="*60)

if __name__ == "__main__":
    main()
