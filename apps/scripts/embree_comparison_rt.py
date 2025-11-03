import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Dict, Optional
import os

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    # "text.latex.preamble": r"\usepackage{amsmath}",  # Optional for math
})


def compute_weighted_average_performance(df: pd.DataFrame, layouts: List[str]) -> pd.DataFrame:
    """
    Compute weighted average trace time across all ray_counts for each layout.
    Weight by ray_count so that configurations with more rays have more influence.
    Returns performance in ns/ray.

    Handles intersect column:
    - embree layouts: use intersect from data (mt or pc)
    - non-embree layouts: determine intersect based on 'q' in layout name
      - contains 'q' -> intersect = 'mt'
      - no 'q' -> intersect = 'pc'
    """
    results = []

    for layout in layouts:
        layout_df = df[df['layout'] == layout]
        if len(layout_df) == 0:
            continue

        # Determine the correct intersect value for this layout
        is_embree = layout.startswith('embree')

        if is_embree:
            # For embree layouts, use the intersect value from the data
            # Should be consistent for all rows of same layout
            if 'intersect' in layout_df.columns:
                expected_intersect = layout_df['intersect'].iloc[0]
                # Filter to only rows with matching intersect
                layout_df = layout_df[layout_df['intersect']
                                      == expected_intersect]
        else:
            # For non-embree layouts, determine intersect based on presence of 'q'
            if 'q' in layout.lower():
                expected_intersect = 'mt'
            else:
                expected_intersect = 'pc'

            # Filter to only rows with matching intersect
            if 'intersect' in layout_df.columns:
                layout_df = layout_df[layout_df['intersect']
                                      == expected_intersect]

        if len(layout_df) == 0:
            continue

        # Compute weighted average: sum(trace_time * ray_count) / sum(ray_count)
        total_ray_time = (layout_df['trace_time-ms']
                          * layout_df['ray_count']).sum()
        total_rays = layout_df['ray_count'].sum()

        if total_rays > 0:
            weighted_avg_time_ms = total_ray_time / total_rays
            # Convert to ns/ray
            ns_per_ray = (weighted_avg_time_ms * 1e6) / \
                (total_rays / len(layout_df))
        else:
            ns_per_ray = np.nan

        # Get total memory (should be constant across ray_counts for same layout)
        total_memory = layout_df['total-memory-b'].iloc[0] if len(
            layout_df) > 0 else np.nan

        # Create display name: prefix non-embree layouts with "bvh8-"
        if is_embree:
            display_layout = layout
        else:
            # Add bvh8- prefix if not already present
            display_layout = layout if layout.startswith(
                'bvh8-') else f'bvh8-{layout}'

        results.append({
            'layout': display_layout,
            'original_layout': layout,
            'ns_per_ray': ns_per_ray,
            'total_memory_mb': total_memory / (1024**2) if not pd.isna(total_memory) else np.nan
        })

    return pd.DataFrame(results)


def compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pareto frontier: points where no other point has both lower memory and lower ns/ray.
    Sort by memory for plotting.
    """
    # Remove NaN values
    df_clean = df.dropna(subset=['total_memory_mb', 'ns_per_ray'])

    if len(df_clean) == 0:
        return pd.DataFrame()

    pareto_points = []

    for idx, row in df_clean.iterrows():
        # Check if this point is dominated by any other point
        is_dominated = False
        for _, other_row in df_clean.iterrows():
            if (other_row['total_memory_mb'] <= row['total_memory_mb'] and
                other_row['ns_per_ray'] < row['ns_per_ray']) or \
               (other_row['total_memory_mb'] < row['total_memory_mb'] and
                    other_row['ns_per_ray'] <= row['ns_per_ray']):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(row)

    pareto_df = pd.DataFrame(pareto_points)
    # Sort by memory for plotting
    pareto_df = pareto_df.sort_values('total_memory_mb')

    return pareto_df


def is_on_pareto_frontier(row, pareto_df):
    """Check if a point is on the Pareto frontier"""
    if len(pareto_df) == 0:
        return False

    for _, pareto_row in pareto_df.iterrows():
        if (abs(row['total_memory_mb'] - pareto_row['total_memory_mb']) < 0.01 and
                abs(row['ns_per_ray'] - pareto_row['ns_per_ray']) < 0.01):
            return True
    return False


def create_best_worst_plots(df: pd.DataFrame, layouts: List[str],
                            machines: List[str], ray_types: List[str],
                            scenes: List[str], output_file: str):
    """
    Create 4 plots in a row showing best and worst case performance differences on x86 and arm.
    For each machine, finds the single best and worst (scene, ray_type) combination by:
    - Computing Pareto frontier for each scenario
    - Averaging ns/ray over Pareto frontier layouts (or best performer if no frontier exists)
    - Selecting scenario with min/max average as best/worst case
    Format: [x86-best, x86-worst, arm-best, arm-worst]
    """
    # Create figure with 4 subplots in a row
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Define colorblind-friendly colors
    bonsai_color = '#0173B2'  # Blue for bonsai layouts
    embree_color = '#DE8F05'  # Orange for embree layouts

    plot_configs = [
        ('x86', 'best'),
        ('x86', 'worst'),
        ('arm', 'best'),
        ('arm', 'worst')
    ]

    for idx, (machine, case) in enumerate(plot_configs):
        ax = axes[idx]

        # First, compute average performance for each (scene, ray_type) combination
        # across Pareto frontier layouts (or best performer if no Pareto frontier exists)
        scenario_avg_performance = {}

        for ray_type in ray_types:
            for scene in scenes:
                # Filter data
                filtered_df = df[
                    (df['machine'] == machine) &
                    (df['ray_type'] == ray_type) &
                    (df['scene'] == scene)
                ]

                if len(filtered_df) == 0:
                    continue

                # Compute weighted averages
                summary_df = compute_weighted_average_performance(
                    filtered_df, layouts)

                if len(summary_df) == 0:
                    continue

                # Compute Pareto frontier
                pareto_df = compute_pareto_frontier(summary_df)

                # Calculate average performance
                if len(pareto_df) > 0:
                    # Average over Pareto frontier layouts
                    avg_perf = pareto_df['ns_per_ray'].mean()
                else:
                    # No Pareto frontier - use best performing layout
                    avg_perf = summary_df['ns_per_ray'].min()

                scenario_avg_performance[(scene, ray_type)] = {
                    'avg_performance': avg_perf,
                    'data': summary_df,
                    'pareto_count': len(pareto_df)
                }

        if len(scenario_avg_performance) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    fontsize=20, transform=ax.transAxes)
            ax.set_title(f"{machine.upper()} - {case.capitalize()}",
                         fontsize=20, fontweight='bold')
            continue

        # Find the best or worst scenario based on average performance
        if case == 'best':
            # Best case: scenario with minimum average ns_per_ray
            best_scenario = min(scenario_avg_performance.items(),
                                key=lambda x: x[1]['avg_performance'])
            selected_scene, selected_ray_type = best_scenario[0]
            case_df = best_scenario[1]['data']
        else:  # worst
            # Worst case: scenario with maximum average ns_per_ray
            worst_scenario = max(scenario_avg_performance.items(),
                                 key=lambda x: x[1]['avg_performance'])
            selected_scene, selected_ray_type = worst_scenario[0]
            case_df = worst_scenario[1]['data']

        print(f"{machine.upper()} {case}: selected ({selected_scene}, {selected_ray_type}) "
              f"with {best_scenario[1]['pareto_count'] if case == 'best' else worst_scenario[1]['pareto_count']} Pareto points")

        # Compute Pareto frontier for this case
        pareto_df = compute_pareto_frontier(case_df)

        # Plot scatter points
        for _, row in case_df.iterrows():
            layout = row['layout']

            # Check if on Pareto frontier
            on_pareto = is_on_pareto_frontier(row, pareto_df)

            # Determine color and marker
            is_embree = layout.startswith('embree')
            color = embree_color if is_embree else bonsai_color
            alpha = 0.8 if on_pareto else 0.6
            marker = '^' if is_embree else 'o'

            # Remove prefixes for display
            display_name = layout
            if is_embree:
                display_name = layout.replace('embree-', '')
            elif display_name != 'bvh8':
                display_name = layout.replace('bvh8-', '')

            ax.scatter(row['total_memory_mb'], row['ns_per_ray'],
                       s=200, color=color, alpha=alpha, edgecolors='black',
                       linewidth=2, marker=marker)

            # Add label
            ax.annotate(display_name,
                        (row['total_memory_mb'], row['ns_per_ray']),
                        textcoords="offset points",
                        xytext=(13.5, 12),
                        ha='center',
                        fontsize=12,
                        fontweight='bold')

        # Plot Pareto frontier line
        if len(pareto_df) > 0:
            ax.plot(pareto_df['total_memory_mb'], pareto_df['ns_per_ray'],
                    'k--', linewidth=2.5, alpha=0.6, zorder=1)

        # Formatting with scene and ray_type in title
        title = f"({selected_scene}, {machine}, {selected_ray_type})"
        ax.set_title(fr"\textbf{{{title}}}",
                     fontsize=18, fontweight='bold', pad=7)
        ax.set_xlabel(r"\textbf{Total Memory (MB)}",
                      fontsize=12, fontweight='bold')
        ax.set_ylabel(r"\textbf{Latency (ns/ray)}",
                      fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add some padding to axes
        if len(case_df) > 0:
            x_margin = (case_df['total_memory_mb'].max() -
                        case_df['total_memory_mb'].min()) * 0.15
            y_margin = (case_df['ns_per_ray'].max() -
                        case_df['ns_per_ray'].min()) * 0.15

            if not pd.isna(x_margin) and x_margin > 0:
                ax.set_xlim(case_df['total_memory_mb'].min() - x_margin,
                            case_df['total_memory_mb'].max() + x_margin)
            if not pd.isna(y_margin) and y_margin > 0:
                ax.set_ylim(case_df['ns_per_ray'].min() - y_margin,
                            case_df['ns_per_ray'].max() + y_margin)

    # Adjust layout
    plt.subplots_adjust(wspace=0.25, left=0.05,
                        right=0.98, top=0.96, bottom=0.08)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bonsai_color,
               markersize=14, markeredgecolor='black', markeredgewidth=2,
               label='Our Work', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=embree_color,
               markersize=14, markeredgecolor='black', markeredgewidth=2,
               label='Embree', linestyle='None')
    ]

    fig.legend(handles=legend_elements, loc='center', fontsize=13,
               framealpha=0.95, edgecolor='black', ncol=1,
               bbox_to_anchor=(0.515, 0.795))

    # Save
    output_path = output_file + "_best_worst.pdf"
    plt.savefig(output_path, dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_scatter_plots(df: pd.DataFrame, layouts: List[str],
                         machines: List[str], ray_types: List[str],
                         scenes: List[str], output_file: str):
    """
    Create a grid of scatter plots (4 per row).
    Each subplot shows memory (x-axis) vs ns/ray (y-axis) for a machine/ray_type/scene combo.
    """

    # Calculate grid dimensions
    n_plots = len(machines) * len(ray_types) * len(scenes)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))

    # Create figure with tighter spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))

    # Flatten axes array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # Define colorblind-friendly colors
    bonsai_color = '#0173B2'  # Blue for bonsai layouts
    embree_color = '#DE8F05'  # Orange for embree layouts

    plot_idx = 0

    for machine in machines:
        for ray_type in ray_types:
            for scene in scenes:
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]

                # Filter data
                filtered_df = df[
                    (df['machine'] == machine) &
                    (df['ray_type'] == ray_type) &
                    (df['scene'] == scene)
                ]

                if len(filtered_df) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                            fontsize=20, transform=ax.transAxes)
                    ax.set_title(f"{scene}\n{machine.upper()} | {ray_type.capitalize()}",
                                 fontsize=20, fontweight='bold')
                    ax.set_xlabel('Total Memory (MB)',
                                  fontsize=18, fontweight='bold')
                    ax.set_ylabel('Performance (ns/ray)',
                                  fontsize=18, fontweight='bold')
                    plot_idx += 1
                    continue

                # Compute weighted averages
                summary_df = compute_weighted_average_performance(
                    filtered_df, layouts)

                if len(summary_df) == 0:
                    ax.text(0.5, 0.5, 'No Layouts', ha='center', va='center',
                            fontsize=20, transform=ax.transAxes)
                    ax.set_title(f"{scene}\n{machine.upper()} | {ray_type.capitalize()}",
                                 fontsize=20, fontweight='bold')
                    ax.set_xlabel('Total Memory (MB)',
                                  fontsize=18, fontweight='bold')
                    ax.set_ylabel('Performance (ns/ray)',
                                  fontsize=18, fontweight='bold')
                    plot_idx += 1
                    continue

                # Compute Pareto frontier
                pareto_df = compute_pareto_frontier(summary_df)

                # Plot scatter points
                for _, row in summary_df.iterrows():
                    layout = row['layout']

                    # Check if on Pareto frontier
                    on_pareto = is_on_pareto_frontier(row, pareto_df)

                    # Determine color and marker
                    is_embree = layout.startswith('embree')

                    # Always use the original color (not gray)
                    color = embree_color if is_embree else bonsai_color

                    # Use different alpha for non-Pareto points to show they're dominated
                    alpha = 0.8 if on_pareto else 0.6

                    marker = '^' if is_embree else 'o'

                    # Remove 'embree-' prefix from layout name for display
                    # Remove 'bvh8-' prefix from non-embree layouts for display
                    display_name = layout
                    if is_embree:
                        display_name = layout.replace('embree-', '')
                    elif display_name != 'bvh8':
                        display_name = layout.replace('bvh8-', '')

                    ax.scatter(row['total_memory_mb'], row['ns_per_ray'],
                               s=200, color=color, alpha=alpha, edgecolors='black',
                               linewidth=2, marker=marker)

                    # Add label near point
                    ax.annotate(display_name,
                                (row['total_memory_mb'], row['ns_per_ray']),
                                textcoords="offset points",
                                xytext=(13.5, 12),
                                ha='center',
                                fontsize=12,
                                fontweight='bold')

                # Plot Pareto frontier line
                if len(pareto_df) > 0:
                    ax.plot(pareto_df['total_memory_mb'], pareto_df['ns_per_ray'],
                            'k--', linewidth=2.5, alpha=0.6, zorder=1)

                # Formatting
                title = f"({scene}, {machine}, {ray_type})"
                ax.set_title(fr"\textbf{{{title}}}",
                             fontsize=18, fontweight='bold', pad=7)
                ax.set_xlabel(r"\textbf{Total Memory (MB)}",
                              fontsize=12, fontweight='bold')
                ax.set_ylabel(r"\textbf{Latency (ns/ray)}",
                              fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(axis='both', which='major', labelsize=14)

                # Add some padding to axes
                x_margin = (summary_df['total_memory_mb'].max(
                ) - summary_df['total_memory_mb'].min()) * 0.15
                y_margin = (summary_df['ns_per_ray'].max(
                ) - summary_df['ns_per_ray'].min()) * 0.15

                if not pd.isna(x_margin) and x_margin > 0:
                    ax.set_xlim(summary_df['total_memory_mb'].min() - x_margin,
                                summary_df['total_memory_mb'].max() + x_margin)
                if not pd.isna(y_margin) and y_margin > 0:
                    ax.set_ylim(summary_df['ns_per_ray'].min() - y_margin,
                                summary_df['ns_per_ray'].max() + y_margin)

                plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Adjust layout - much tighter spacing
    plt.subplots_adjust(hspace=0.35, wspace=0.25, left=0.05,
                        right=0.98, top=0.96, bottom=0.05)

    # Add centered legend for entire figure (smaller)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bonsai_color,
               markersize=14, markeredgecolor='black', markeredgewidth=2,
               label='Our Work', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=embree_color,
               markersize=14, markeredgecolor='black', markeredgewidth=2,
               label='Embree', linestyle='None')
    ]

    # Place legend in center of figure - smaller size
    fig.legend(handles=legend_elements, loc='center', fontsize=13,
               framealpha=0.95, edgecolor='black', ncol=1,
               bbox_to_anchor=(0.515, 0.795))

    # Save
    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_file}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate scatter plots comparing memory vs performance')
    parser.add_argument('input_csv', help='Input CSV file with benchmark data')
    parser.add_argument('--layouts', nargs='+', required=True,
                        help='List of layouts to compare')
    parser.add_argument('--machines', nargs='+', default=['x86', 'arm'],
                        help='Machines to generate plots for (default: x86 arm)')
    parser.add_argument('--ray-types', nargs='+', default=['primary', 'secondary'],
                        help='Ray types to generate plots for (default: primary secondary)')
    parser.add_argument('--scenes', nargs='+', required=True,
                        help='Scenes to generate plots for')
    parser.add_argument('--output', default='memory_vs_performance',
                        help='Output file name (default: memory_vs_performance)')
    parser.add_argument('--best', action='store_true',
                        help='Generate best/worst case performance comparison plots (4 graphs: x86-best, x86-worst, arm-best, arm-worst)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Generating plots for:")
    print(f"  Layouts: {args.layouts}")
    print(f"  Machines: {args.machines}")
    print(f"  Ray types: {args.ray_types}")
    print(f"  Scenes: {args.scenes}")

    # Generate appropriate plots based on --best flag
    if args.best:
        print("\nGenerating best/worst case comparison plots...")
        create_best_worst_plots(
            df, args.layouts, args.machines, args.ray_types, args.scenes, args.output
        )
    else:
        print("\nGenerating standard scatter plots...")
        create_scatter_plots(
            df, args.layouts, args.machines, args.ray_types, args.scenes, args.output
        )

    print(f"\nDone! Output saved to {args.output}")


if __name__ == "__main__":
    main()
