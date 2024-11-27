import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_metrics_file(file_path):
    """
    Parse metrics file to extract F1-scores and other key metrics

    Parameters:
    -----------
    file_path : str
        Path to the metrics file

    Returns:
    --------
    dict
        Extracted metrics
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract overall accuracy
    accuracy_match = re.search(r'Accuracy: (\d+\.\d+)', content)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    # Extract class-specific F1-scores
    class_metrics = re.findall(r'(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', content)

    metrics = {
        'overall_accuracy': accuracy,
        'class_metrics': {}
    }

    for class_data in class_metrics:
        class_id, precision, recall, f1_score = class_data
        metrics['class_metrics'][int(class_id)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        }

    return metrics


def run_multiple_experiments(base_dir, num_experiments=10):
    """
    Run multiple experiments to calculate performance metrics and standard deviation

    Parameters:
    -----------
    base_dir : str
        Base directory containing metrics files
    num_experiments : int, optional
        Number of experiments to run (default 10)

    Returns:
    --------
    pd.DataFrame
        DataFrame with performance metrics across experiments
    """
    results = []
    methods = ['t-SNE', 'UMAP', 'DBSCAN']

    for _ in range(num_experiments):
        method_results = []

        for method in methods:
            metrics_path = os.path.join(base_dir, method.lower(), 'metrics.txt')

            try:
                metrics = parse_metrics_file(metrics_path)

                # Focus on minority class (class 1)
                minority_metrics = metrics['class_metrics'].get(1, {})

                method_results.append({
                    'Method': method,
                    'Accuracy': metrics['overall_accuracy'],
                    'F1_Score_Minority': minority_metrics.get('f1_score', 0),
                    'Precision_Minority': minority_metrics.get('precision', 0),
                    'Recall_Minority': minority_metrics.get('recall', 0)
                })
            except Exception as e:
                print(f"Error processing {method} metrics: {e}")

        results.extend(method_results)

    return pd.DataFrame(results)


def create_performance_visualization(results_df):
    """
    Create visualization of model performance and stability

    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with performance metrics
    """
    # Set up the plot style
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    agg_data = results_df.groupby('Method').agg({
        'F1_Score_Minority': ['mean', 'std'],
        'Accuracy': ['mean', 'std']
    }).reset_index()
    agg_data.columns = ['Method', 'F1_Mean', 'F1_Std', 'Accuracy_Mean', 'Accuracy_Std']

    # Plot F1-Score for Minority Class
    plt.subplot(1, 2, 1)
    sns.barplot(x='Method', y='F1_Mean', data=agg_data,
                yerr=agg_data['F1_Std'], capsize=0.1,
                palette='viridis')
    plt.title('F1-Score for Minority Class\nwith Standard Deviation', fontsize=10)
    plt.ylabel('F1-Score')
    plt.xlabel('Dimensionality Reduction Method')

    # Plot Overall Accuracy
    plt.subplot(1, 2, 2)
    sns.barplot(x='Method', y='Accuracy_Mean', data=agg_data,
                yerr=agg_data['Accuracy_Std'], capsize=0.1,
                palette='viridis')
    plt.title('Overall Accuracy\nwith Standard Deviation', fontsize=10)
    plt.ylabel('Accuracy')
    plt.xlabel('Dimensionality Reduction Method')

    plt.tight_layout()

    # Ensure results directory exists
    os.makedirs('results/comparison', exist_ok=True)

    # Save the plot
    plt.savefig('results/comparison/model_performance_comparison.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\nPerformance Summary:")
    print(agg_data.to_string(index=False))

    # Save summary to CSV
    agg_data.to_csv('results/comparison/performance_summary.csv', index=False)


# Main execution
if __name__ == '__main__':
    # Run analysis
    results_df = run_multiple_experiments('results')
    create_performance_visualization(results_df)

    # Optional: Save full results for detailed analysis
    results_df.to_csv('results/comparison/full_performance_results.csv', index=False)