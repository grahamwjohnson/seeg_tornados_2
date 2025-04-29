import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import EngFormatter, LogLocator, LogFormatterSciNotation

def detect_write_frequency(true_steps, expected_freq=5):
    """Confirm write frequency matches expected value"""
    diffs = np.diff(true_steps)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) > 0:
        unique, counts = np.unique(positive_diffs, return_counts=True)
        detected_freq = unique[np.argmax(counts)]
        if detected_freq != expected_freq:
            print(f"Note: Detected write frequency {detected_freq} differs from expected {expected_freq}")
    return expected_freq

def process_steps(df, write_freq=5):
    """Process steps using known write frequency"""
    wandb_steps = df.iloc[:, 0].values
    true_steps = df.iloc[:, 4].values
    
    scaled_true_steps = true_steps * write_freq
    last_occurrences = {step: i for i, step in enumerate(scaled_true_steps)}
    valid_indices = sorted(last_occurrences.values())
    
    processed = pd.DataFrame({
        'WandB_Step': wandb_steps[valid_indices],
        'True_Step': scaled_true_steps[valid_indices],
        'Reconstructed_Step': np.arange(len(valid_indices)) * write_freq
    })
    
    print(f"\nProcessing Summary:")
    print(f"- Write frequency: {write_freq}x")
    print(f"- Original entries: {len(df)}")
    print(f"- Unique steps after filtering: {len(valid_indices)}")
    print(f"- Total training steps represented: {len(valid_indices) * write_freq:,}\n")
    
    return processed

def process_metric_file(file_path, step_mapping):
    """Process metric file"""
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) < 2:
            return None
            
        wandb_steps = df.iloc[:, 0].astype(float)
        values = df.iloc[:, 1].astype(float)
        metric_name = df.columns[1]
        
        return pd.merge(
            step_mapping,
            pd.DataFrame({'WandB_Step': wandb_steps, metric_name: values}),
            on='WandB_Step',
            how='left'
        ).dropna()
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Configure Seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.8',
        'grid.color': '0.9',
        'axes.titlepad': 14
    })
    plt.rcParams['figure.facecolor'] = 'white'
    palette = sns.color_palette("husl", 2)
    
    base_dir = '/media/glommy1/tornados/trained_models/dataset_train100.0_val0.0/ekaltadeta_Tue_Mar_25_16_49_04_2025/loss_csvs'
    step_file = f"{base_dir}/steps.csv"
    metric_files = [
        f'{base_dir}/total_loss.csv',
        f'{base_dir}/kl.csv'
    ]

    try:
        # Process data
        df_steps = pd.read_csv(step_file)
        step_mapping = process_steps(df_steps, write_freq=5)
        
        metrics = {}
        for file in metric_files:
            name = file.split('/')[-1].split('.')[0]
            df_metric = process_metric_file(file, step_mapping)
            if df_metric is not None and not df_metric.empty:
                metrics[name] = df_metric

        # Create figure
        fig, axes = plt.subplots(len(metrics)+1, 1, 
                               figsize=(14, 5+3*len(metrics)),
                               dpi=120)
        
        # Plot step progression (linear scale)
        ax = axes[0]
        sns.lineplot(
            data=step_mapping, 
            x='Reconstructed_Step', 
            y='True_Step',
            ax=ax, 
            linewidth=2,
            color=palette[0]
        )
        ax.set_title(
            f'Training Step Progression\n(Total Steps: {step_mapping["Reconstructed_Step"].iloc[-1]:,.0f})', 
            pad=12, 
            fontsize=13
        )
        ax.set_ylabel('Step', fontsize=11)
        ax.xaxis.set_major_formatter(EngFormatter())
        sns.despine(ax=ax)
        ax.grid(True, alpha=0.4)

        # Plot each metric with log-log where appropriate
        for i, (name, df) in enumerate(metrics.items(), 1):
            ax = axes[i]
            metric_col = [c for c in df.columns if c not in ['WandB_Step', 'True_Step', 'Reconstructed_Step']][0]
            
            # Apply log-log scaling for loss plots
            if 'loss' in name.lower():
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
                ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
                ax.xaxis.set_major_formatter(LogFormatterSciNotation())
                ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                grid_alpha = 0.2
            else:
                ax.xaxis.set_major_formatter(EngFormatter())
                grid_alpha = 0.4
            
            sns.lineplot(
                data=df, 
                x='Reconstructed_Step', 
                y=metric_col,
                ax=ax, 
                linewidth=1.8,
                color=palette[1]
            )
            
            ax.set_title(
                f'{name.replace("_", " ").title()}', 
                pad=12, 
                fontsize=13
            )
            ax.set_ylabel('Value', fontsize=11)
            sns.despine(ax=ax)
            ax.grid(True, alpha=grid_alpha, which='both' if 'loss' in name.lower() else 'major')

        # Common x-axis label
        axes[-1].set_xlabel('Training Steps (Scaled ×5)', fontsize=11)
        
        plt.tight_layout(h_pad=2.5)
        plt.savefig(f'{base_dir}/training_metrics_loglog.png', bbox_inches='tight', dpi=120)
        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")