import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, FuncFormatter

import numpy as np

def plot_speedups(problem_sizes, benchmark_data, filename, title="Benchmark Speed-up Comparison"):
    """
    Plots speed-ups for multiple benchmarks.

    Parameters:
    - problem_sizes: list of x-axis values (problem sizes)
    - benchmark_data: list of tuples (benchmark_name, list_of_speedups)
    - title: title of the plot (optional)
    """

    # Color-blind friendly color palette (Color Universal Design - CUD)
    # See: https://jfly.uni-koeln.de/color/
    colors = [
        "#000000",  # black
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermilion
        "#CC79A7",   # reddish purple
        "#999999"  # medium gray
    ]

    # Line styles for extra clarity in B&W printing
    line_styles = ['-', '--', '-.', ':']

    plt.figure(figsize=(10, 6))
    
    for i, (name, speedups) in enumerate(benchmark_data):
        color = colors[i % len(colors)]
        linestyle = line_styles[(i // len(colors)) % len(line_styles)]
        plt.plot(problem_sizes, speedups, label=name, color=color, linestyle=linestyle, marker='o')

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)

    ax = plt.gca()

    # Custom ticks: every other power of 2 on x-axis
    x_min = min(problem_sizes)
    x_max = max(problem_sizes)
    exponents = np.arange(int(np.floor(np.log2(x_min))), int(np.ceil(np.log2(x_max))) + 1, 2)
    xticks = [2 ** e for e in exponents]
    ax.set_xticks(xticks)
    plt.ylim(1, 2**24)
    # plt.ylim(1, 2**18) # for 2D graph

    # Optional: pretty "2^x" labels
    # ax.set_xticklabels([f"$2^{{{e}}}$" for e in exponents])

    # Set ticks at powers of 2 (x and y)
    # ax.xaxis.set_major_locator(LogLocator(base=2.0, subs=[1.0], numticks=20))
    ax.yaxis.set_major_locator(LogLocator(base=2.0, subs=[1.0], numticks=20))

    # Format tick labels as plain numbers (no scientific notation)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain')

    plt.xlabel("Problem Size")
    plt.ylabel("Speed-up over linear scan")
    # plt.title("SELECT COUNT(*) WHERE predictate;")
    plt.legend(loc="best", frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, format="pdf")

def plot_join_speedups(data, filename, title="Nested vs. Stacked Join Runtimes"):
    sizes, runtime_nested, runtime_build, runtime_query = zip(*data)
    x = np.arange(len(sizes))
    bar_width = 0.35

    # Color-blind friendly palette (CUD)
    colors = {
        "nested": "#D55E00",  # vermilion
        "build":  "#E69F00",  # orange
        "query":  "#56B4E9",  # sky blue
        "half_build": "#F0E442",  # yellow
        "single_query": "#009E73",  # bluish green
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Nested bars
    ax.bar(x - bar_width / 2, runtime_nested, width=bar_width,
           color=colors["nested"], label="Nested Join", edgecolor='black', linewidth=0.5)

    # Stacked bars: build + query
    ax.bar(x + bar_width / 2, runtime_query, width=bar_width,
           color=colors["query"], label="Dual Tree Join", edgecolor='black', linewidth=0.5)
    ax.bar(x + bar_width / 2, runtime_build, width=bar_width,
           bottom=runtime_query, color=colors["build"], label="Dual Tree Build",
           edgecolor='black', linewidth=0.5)

    # Axis styling
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Runtime (ns)")

    # --- X TICKS as powers of 2 ---
    # Convert sizes (e.g., [1024, 2048, 4096]) → exponents ([10, 11, 12])
    x_powers = [int(np.log2(s)) for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$2^{{{p}}}$" for p in x_powers])

    # Y ticks as powers of 2
    ax.set_yscale("log", base=2)
    max_runtime = max(runtime_nested)  # or sum of stacked bars if you want total
    y_exp_min = 0
    y_exp_max = int(np.ceil(np.log2(max_runtime)))
    step = 4  # show every 4th power of 2

    y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, step)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(2**i) if i == 0 else rf"$2^{{{i}}}$" for i in range(y_exp_min, y_exp_max + 1, step)])

    # Grid, legend, and layout
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc="best", frameon=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format="pdf")

# def plot_join_speedups(data, filename, title="Join Speedups vs Nested Loop"):
#     # unpack data
#     sizes, runtime_nested, runtime_build, runtime_query = zip(*data)
#     sizes = np.array(sizes)
#     runtime_nested = np.array(runtime_nested)
#     runtime_build = np.array(runtime_build)
#     runtime_query = np.array(runtime_query)

#     # compute speedups
#     speedup_query = runtime_nested / runtime_query
#     speedup_total = runtime_nested / (runtime_build + runtime_query)

#     x = np.arange(len(sizes))
#     bar_width = 0.35

#     # make figure
#     fig, ax = plt.subplots(figsize=(8, 5))

#     # bars
#     bars1 = ax.bar(x - bar_width/2, speedup_query, bar_width, label="Dual Tree Traversal")
#     bars2 = ax.bar(x + bar_width/2, speedup_total, bar_width, label="Dual Tree Traversal (with Build)")

#     # log2 y scale
#     ax.set_yscale("log", base=2)
#     y_exp_min = 0
#     y_exp_max = int(np.ceil(np.log2(max(speedup_query.max(), speedup_total.max()))))
#     y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, 2)]  # 2^0, 2^2, 2^4, etc.
#     ax.set_yticks(y_ticks)
#     ax.set_yticklabels(["1" if i == 0 else rf"$2^{{{i}}}$" for i in range(y_exp_min, y_exp_max + 1, 2)])

#     # x ticks (2^power style)
#     ax.set_xticks(x)
#     ax.set_xticklabels([rf"$2^{{{int(np.log2(s))}}}$" for s in sizes])

#     ax.set_xlabel("Input size")
#     ax.set_ylabel("Speedup over Nested Join")
#     # ax.set_title(title)
#     ax.legend()
#     ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(filename, format="pdf")

def plot_join_speedups(data, filename, title="Nested vs. Stacked Join Runtimes"):
    # Unpack five-tuples
    sizes, runtime_nested, runtime_build, runtime_single, runtime_dual = zip(*data)
    x = np.arange(len(sizes))
    bar_width = 0.3  # slightly smaller to fit three bars per x

    # Color-blind friendly palette
    colors = {
        "nested": "#D55E00",       # vermilion
        "build":  "#E69F00",       # orange
        "dual_query":  "#56B4E9",  # sky blue
        "half_build": "#F0E442",   # yellow
        "single_query": "#009E73", # bluish green
    }

    # Convert to numpy arrays for arithmetic
    runtime_nested = np.array(runtime_nested)
    runtime_build = np.array(runtime_build)
    runtime_single = np.array(runtime_single)
    runtime_dual = np.array(runtime_dual)

    # Compute totals for stacked bars
    single_total = runtime_single + runtime_build / 2
    dual_total   = runtime_dual + runtime_build

    # Compute speedups
    speedup_single = runtime_nested / single_total
    speedup_dual   = runtime_nested / dual_total

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar 1: Nested
    ax.bar(x - bar_width, runtime_nested, width=bar_width,
           color=colors["nested"], label="Nested Join", edgecolor='black', linewidth=0.5)

    # Bar 2: Single-tree join + half build
    single_total = np.array(runtime_single) + np.array(runtime_build)/2
    ax.bar(x, runtime_single, width=bar_width,
           color=colors["single_query"], label="Single Tree Join", edgecolor='black', linewidth=0.5)
    ax.bar(x, np.array(runtime_build)/2, width=bar_width,
           bottom=runtime_single, color=colors["half_build"], label="Single Build",
           edgecolor='black', linewidth=0.5)

    # Bar 3: Dual-tree join + full build
    dual_total = np.array(runtime_dual) + np.array(runtime_build)
    ax.bar(x + bar_width, runtime_dual, width=bar_width,
           color=colors["dual_query"], label="Dual Tree Join", edgecolor='black', linewidth=0.5)
    ax.bar(x + bar_width, runtime_build, width=bar_width,
           bottom=runtime_dual, color=colors["build"], label="Dual Build",
           edgecolor='black', linewidth=0.5)

    for i in range(len(x)):
        # Single-tree speedup label
        ax.text(x[i], single_total[i]*1.05, f"{speedup_single[i]:.1f}×",
                ha='center', va='bottom', fontsize=7, fontweight="bold", rotation=0)
        # Dual-tree speedup label
        ax.text(x[i] + bar_width, dual_total[i]*1.05, f"{speedup_dual[i]:.1f}×",
                ha='center', va='bottom', fontsize=7, fontweight="bold", rotation=0)

    # Axis styling
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Runtime (ns)")

    # X ticks as powers of 2
    x_powers = [int(np.log2(s)) for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$2^{{{p}}}$" for p in x_powers])

    # Y-axis log2 scaling and sparse ticks
    ax.set_yscale("log", base=2)
    max_runtime = max(max(runtime_nested), max(dual_total), max(single_total))
    y_exp_min = 0
    y_exp_max = int(np.ceil(np.log2(max_runtime)))
    step = 4
    y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, step)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(2**i) if i == 0 else rf"$2^{{{i}}}$" for i in range(y_exp_min, y_exp_max + 1, step)])

    # Grid, legend, title
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    # plt.show()

def plot_speedups_over_nested(data, filename, title="Speedups over Nested Join"):
    # Unpack five-tuples
    sizes, runtime_nested, runtime_build, runtime_single, runtime_dual = zip(*data)
    x = np.arange(len(sizes))
    bar_width = 0.2  # smaller to fit 4 bars per x

    # Color-blind friendly palette
    colors = {
        "single_plus_half_build": "#F0E442",  # yellow
        "single": "#009E73",                  # bluish green
        "dual_plus_build": "#E69F00",         # orange
        "dual": "#56B4E9",                     # sky blue
    }

    # Convert to numpy arrays for arithmetic
    runtime_nested = np.array(runtime_nested)
    runtime_build  = np.array(runtime_build)
    runtime_single = np.array(runtime_single)
    runtime_dual   = np.array(runtime_dual)

    # Compute speedups
    speed_single_plus_half = runtime_nested / (runtime_single + runtime_build/2)
    speed_single           = runtime_nested / runtime_single
    speed_dual_plus_build  = runtime_nested / (runtime_dual + runtime_build)
    speed_dual             = runtime_nested / runtime_dual

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the four bars
    ax.bar(x - 1.5*bar_width, speed_single_plus_half, width=bar_width,
           color=colors["single_plus_half_build"], label="Single + Build",
           edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*bar_width, speed_single, width=bar_width,
           color=colors["single"], label="Single Index Join", edgecolor='black', linewidth=0.5)
    ax.bar(x + 0.5*bar_width, speed_dual_plus_build, width=bar_width,
           color=colors["dual_plus_build"], label="Dual + Build",
           edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*bar_width, speed_dual, width=bar_width,
           color=colors["dual"], label="Dual Index Join", edgecolor='black', linewidth=0.5)

    # Axis styling
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Speedup over Nested Join")

    # --- X ticks as powers of 2 ---
    x_powers = [int(np.log2(s)) for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$2^{{{p}}}$" for p in x_powers])

    # --- Y-axis log2 scaling with sparse ticks ---
    ax.set_yscale("log", base=2)
    max_speedup = max(speed_single_plus_half.max(), speed_single.max(),
                      speed_dual_plus_build.max(), speed_dual.max())
    y_exp_min = 0
    y_exp_max = int(np.ceil(np.log2(max_speedup)))
    step = 1  # show every 2^1
    y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, step)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(2**i) if i == 0 else rf"$2^{{{i}}}$"
                        for i in range(y_exp_min, y_exp_max + 1, step)])

    # Grid, legend, title
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", frameon=False)
    # ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")

def plot_speedups_over_nested(data, filename, title="Speedups over Nested Join"):
    # Unpack five-tuples
    sizes, runtime_nested, runtime_build, runtime_single, runtime_dual = zip(*data)
    x = np.arange(len(sizes))

    # Color-blind friendly palette
    colors = {
        "single_plus_half_build": "#F0E442",  # yellow
        "single": "#009E73",                  # bluish green
        "dual_plus_build": "#E69F00",         # orange
        "dual": "#56B4E9",                     # sky blue
    }

    # Convert to numpy arrays for arithmetic
    runtime_nested = np.array(runtime_nested)
    runtime_build  = np.array(runtime_build)
    runtime_single = np.array(runtime_single)
    runtime_dual   = np.array(runtime_dual)

    # Compute speedups
    speed_single_plus_half = runtime_nested / (runtime_single + runtime_build/2)
    speed_single           = runtime_nested / runtime_single
    speed_dual_plus_build  = runtime_nested / (runtime_dual + runtime_build)
    speed_dual             = runtime_nested / runtime_dual

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot four lines with markers
    ax.plot(x, speed_single_plus_half, '-o', color=colors["single_plus_half_build"],
            label="Single + Build", linewidth=2, markersize=6)
    ax.plot(x, speed_single, '-s', color=colors["single"],
            label="Single Index Join", linewidth=2, markersize=6)
    ax.plot(x, speed_dual_plus_build, '-^', color=colors["dual_plus_build"],
            label="Dual + Build", linewidth=2, markersize=6)
    ax.plot(x, speed_dual, '-d', color=colors["dual"],
            label="Dual Index Join", linewidth=2, markersize=6)

    # Axis styling
    ax.set_xlabel("Problem Size")
    ax.set_ylabel("Speedup over Nested Join")

    # --- X ticks as powers of 2 ---
    x_powers = [int(np.log2(s)) for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$2^{{{p}}}$" for p in x_powers])

    # --- Y-axis log2 scaling with sparse ticks ---
    ax.set_yscale("log", base=2)
    max_speedup = max(speed_single_plus_half.max(), speed_single.max(),
                      speed_dual_plus_build.max(), speed_dual.max())
    y_exp_min = 0
    y_exp_max = int(np.ceil(np.log2(max_speedup)))
    step = 1  # show every 2^1
    y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, step)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(2**i) if i == 0 else rf"$2^{{{i}}}$"
                        for i in range(y_exp_min, y_exp_max + 1, step)])

    # Grid, legend, title
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(filename, format="pdf")

def plot_speedups_over_nested(data, filename, title="Speedups over Nested Join"):
    # Unpack five-tuples
    sizes, runtime_nested, runtime_build, runtime_single, runtime_dual = zip(*data)
    x = np.arange(len(sizes))
    bar_width = 0.35

    # Color-blind friendly palette
    colors = {
        "single_plus_half_build": "#F0E442",  # yellow
        "single": "#009E73",                  # bluish green
        "dual_plus_build": "#E69F00",         # orange
        "dual": "#56B4E9",                    # sky blue
        "dual_plus_half_build": "#D55E00",  # vermilion
    }

    # Convert to numpy arrays for arithmetic
    runtime_nested = np.array(runtime_nested)
    runtime_build  = np.array(runtime_build)
    runtime_single = np.array(runtime_single)
    runtime_dual   = np.array(runtime_dual)

    # Compute speedups
    speed_single_plus_half = runtime_nested / (runtime_single + runtime_build / 2)
    speed_single           = runtime_nested / runtime_single
    speed_dual_plus_build  = runtime_nested / (runtime_dual + runtime_build)
    speed_dual_plus_half   = runtime_nested / (runtime_dual + runtime_build / 2)
    speed_dual             = runtime_nested / runtime_dual

    fig, ax = plt.subplots(figsize=(12, 6))

    # # --- Left group: Single tree ---
    # ax.bar(x - bar_width / 2, speed_single_plus_half, width=bar_width,
    #        color=colors["single_plus_half_build"], label="Single + Build(1)",
    #        edgecolor='black', linewidth=0.5)
    # ax.bar(x - bar_width / 2, speed_single, width=bar_width,
    #        bottom=speed_single_plus_half, color=colors["single"],
    #        label="Single Index Join", edgecolor='black', linewidth=0.5)

    # # --- Right group: Dual tree ---
    # ax.bar(x + bar_width / 2, speed_dual_plus_build, width=bar_width,
    #        color=colors["dual_plus_build"], label="Dual + Build(2)",
    #        edgecolor='black', linewidth=0.5)
    # ax.bar(x + bar_width / 2, speed_dual_plus_half, width=bar_width,
    #        bottom=speed_dual_plus_build, color=colors["dual_plus_half_build"],
    #        label="Dual  + Build(1)", edgecolor='black', linewidth=0.5)
    # ax.bar(x + bar_width / 2, speed_dual, width=bar_width,
    #        bottom=speed_dual_plus_half, color=colors["dual"],
    #        label="Dual Index Join", edgecolor='black', linewidth=0.5)

    ax.plot(x, speed_single_plus_half, '-o', color=colors["single_plus_half_build"],
            label="Single + Build(1)", linewidth=2, markersize=6)
    ax.plot(x, speed_single, '-s', color=colors["single"],
            label="Single Index Join", linewidth=2, markersize=6)
    ax.plot(x, speed_dual_plus_build, '-^', color=colors["dual_plus_build"],
            label="Dual + Build(2)", linewidth=2, markersize=6)
    ax.plot(x, speed_dual_plus_half, '-d', color=colors["dual_plus_half_build"],
            label="Dual + Build(1)", linewidth=2, markersize=6)
    ax.plot(x, speed_dual, '-x', color=colors["dual"],
            label="Dual Index Join", linewidth=2, markersize=6)

    # --- Axis styling ---
    ax.set_xlabel("Problem Size", fontsize=20)
    ax.set_ylabel("Speedup over Nested Join", fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # X ticks as powers of 2
    x_powers = [int(np.log2(s)) for s in sizes]
    ax.set_xticks(x)
    ax.set_xticklabels([rf"$2^{{{p}}}$" for p in x_powers])

    # --- Horizontal baseline for Nested Join ---
    ax.axhline(
        y=1, color='black', linewidth=5, linestyle='-', clip_on=False,
        xmin=-0.005, xmax=1.005,  # extend slightly off both sides
        zorder=3
    )

    # Label just above the right end of the line
    ax.text(
        0.84, 1.1, "Nested Join",
        color='black', fontsize=18, fontweight='bold',
        va='bottom', ha='left', transform=ax.get_yaxis_transform()
    )
    ax.set_ylim(bottom=1)

    # Y-axis log2 scaling and ticks
    ax.set_yscale("log", base=2)
    max_speedup = np.max([
        (speed_single_plus_half + speed_single),
        (speed_dual_plus_build + speed_dual)
    ])
    y_exp_min = 0
    y_exp_max = int(np.ceil(np.log2(max_speedup)))
    step = 1
    y_ticks = [2**i for i in range(y_exp_min, y_exp_max + 1, step)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(["1" if i == 0 else rf"$2^{{{i}}}$" for i in range(y_exp_min, y_exp_max + 1, step)])

    # --- Grid, legend, layout ---
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", frameon=False, fontsize=20)
    ax.set_title(title, fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, format="pdf")

# Example usage:
if __name__ == "__main__":
    problem_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728]

    # Uniform
    # filename = "uniform-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [15.1818, 10.4097, 18.4619, 30.122, 54.1387, 72.5403, 107.619, 186.227, 235.526, 260.047, 305.175, 389.91, 414.915, 407.12, 454.723, 468.686, 457.671, 429.6, 416.503, 363.271]),
    #     ("x == 42", [17.803, 48.2088, 53.978, 85.6481, 217.714, 252.698, 585.265, 959.143, 1842.76, 3654.13, 6711.82, 12932.1, 24312.9, 45513.7, 90087.1, 170618, 326328, 624606, 1.08092e+06, 2.51629e+06]),
    #     ("abs(x) <= 10", [9.13953, 6.88418, 11.5266, 18.2135, 34.2045, 40.7766, 70.0548, 97.7704, 131.039, 146.745, 173.557, 195.97, 270.953, 225.106, 226.397, 234.728, 237.569, 239.394, 207.368, 193.013]),
    #     ("x^2 <= 100", [11.25, 6.89548, 12.8193, 18.4186, 33.1936, 42.667, 74.2751, 104.89, 156.737, 186.766, 225.475, 259.864, 219.504, 224.378, 229.225, 245.98, 236.07, 203.595, 220.536, 171.15]),
    #     ("round(x) == 10", [14.8608, 29.2152, 40.8707, 60.0065, 84.9378, 136.54, 157.664, 286.73, 408.06, 663.971, 1261.61, 1767.83, 2533.45, 3521.88, 3162.99, 4402.27, 4111.89, 4247.78, 4337.07, 4711.61]),
    #     ("x^2 - 4 * x + 3 <= 0", [12.7253, 12.6339, 31.0671, 54.2882, 52.9483, 82.2267, 142.769, 154.662, 303.889, 455.484, 749.594, 1104.88, 1501.67, 1868.37, 2261.46, 1987.08, 2042.79, 2270.51, 2197.83, 2196.68]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [9.296, 6.99714, 12.3171, 19.1152, 31.7722, 44.1095, 68.9195, 103.325, 126.355, 134.628, 188.33, 198.424, 217.024, 227.148, 221.245, 242.273, 250.031, 241.385, 233.963, 212.206]),
    #     ("abs(x - u) > s * 3", [28.439, 69.9394, 113, 249.541, 331.581, 741.606, 1250.62, 2752.29, 7716.66, 9155.58, 23861.2, 62274.8, 82216.8, 119493, 252605, 469664, 1.16649e+06, 2.68187e+06, 3.33572e+06, 8.04281e+06]),
    # ]
    # Normal
    # filename = "normal-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [7.50437, 12.7945, 25.6058, 32.1049, 45.6977, 62.8349, 127.025, 140.833, 214.036, 208.702, 261.703, 220.043, 268.946, 277.295, 273.298, 290.932, 269.226, 264.528, 267.21, 191.693]),
    #     ("x == 42", [14.526, 29.0267, 47.7213, 87.1765, 163.368, 277.973, 537.856, 998.991, 1871.04, 3469.09, 8601.24, 13073.1, 27774.5, 44730.6, 91152.5, 168782, 354574, 641502, 1.10658e+06, 1.92917e+06]),
    #     ("abs(x) <= 10", [5.2517, 7.79216, 14.7255, 16.0241, 27.2659, 35.4983, 61.4847, 68.2523, 107.118, 110.051, 133.428, 145.271, 125.646, 141.414, 136.898, 129.783, 134.951, 140.219, 130.909, 123.52]),
    #     ("x^2 <= 100", [5.9475, 7.89267, 14.4464, 15.7938, 22.4593, 36.4382, 64.3474, 89.2552, 124.653, 108.547, 165.912, 124.31, 131.391, 147.089, 138.917, 152.727, 133.168, 122.916, 135.509, 125.272]),
    #     ("round(x) == 10", [14.5519, 15.828, 23.4308, 48.3734, 69.7956, 88.4923, 175.35, 396.993, 365.856, 716.712, 932.307, 1741.89, 2153.8, 1924.23, 2176.51, 2162.16, 2777.59, 2885.08, 2605.31, 2711.39]),
    #     ("x^2 - 4 * x + 3 <= 0", [14.2532, 12.4887, 30.6167, 23.8736, 36.4709, 63.9877, 107.66, 181.108, 235.023, 401.658, 649.492, 912.642, 945.37, 1181.25, 1493.86, 1630.96, 1458.42, 1502.43, 1516.02, 1529.26]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [5.69712, 8.02087, 14.4487, 16.0199, 21.6599, 35.7449, 60.5742, 87.8741, 122.346, 150.023, 137.517, 117.511, 129.013, 141.586, 138.062, 142.578, 135.763, 140.033, 132.312, 111.265]),
    #     ("abs(x - u) > s * 3", [28.6835, 62.7286, 125.3, 268.241, 287.944, 689.107, 1243.42, 3024.43, 5455.68, 9933.69, 17941.2, 39827.5, 87436.7, 197386, 265718, 587652, 1.06312e+06, 1.46774e+06, 4.63017e+06, 9.40224e+06]),
    # ]
    # Exponential
    # filename = "exponential-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [6.08871, 9.3075, 17.9612, 24.9353, 44.6987, 79.6194, 128.039, 202.739, 237.587, 270.594, 367.151, 307.148, 382.484, 436.523, 439.354, 381.67, 409.518, 385.616, 347.116, 367.865]),
    #     ("x == 42", [6.55906, 11.7259, 38.575, 88.9327, 154.025, 295.096, 620.024, 1075.13, 2164.36, 3508.78, 8088.05, 13097.6, 28844.9, 49467.5, 78805.7, 159948, 174677, 262659, 601987, 774277]),
    #     ("abs(x) <= 10", [3.44572, 4.89017, 11.3892, 16.9555, 28.2583, 49.2348, 69.172, 137.589, 169.346, 148.905, 183.01, 194.681, 226.361, 237.931, 250.709, 257.931, 257.761, 247.607, 197.292, 186.165]),
    #     ("x^2 <= 100", [3.46644, 5.71223, 12.1823, 17.2004, 29.8744, 47.5062, 74.0556, 138.076, 134.16, 161.582, 212.602, 246.511, 221.795, 264.57, 259.236, 247.663, 247.657, 239.548, 244.312, 180.227]),
    #     ("round(x) == 10", [7.63889, 9.11688, 44.3846, 55.3488, 108.5, 151.511, 138.158, 279.662, 506.79, 710.799, 1114.44, 1888.04, 2759.59, 3936.95, 5052.92, 4771.06, 4169.44, 6402.73, 5228.96, 5609.47]),
    #     ("x^2 - 4 * x + 3 <= 0", [5.0125, 7.99377, 39.7586, 32.788, 61.102, 88.7165, 112.967, 212.547, 308.554, 447.42, 801.782, 978.2, 1208.02, 2216.13, 1999.09, 2235.08, 2539.49, 2563.36, 2630.7, 2626.38]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [3.68871, 5.28258, 11.0686, 15.895, 14.7817, 49.2224, 65.3483, 146.476, 152.65, 149.022, 215.861, 247.866, 232.113, 256.477, 229.693, 209.03, 238.644, 210.94, 240.624, 221.746]),
    #     ("abs(x - u) > s * 3", [17.2232, 31.1067, 93.9592, 188.429, 343.056, 704.086, 1470.23, 3291.46, 5070.18, 9110.4, 16682.4, 40046.4, 82085.3, 177159, 253112, 882116, 1.17467e+06, 2.1359e+06, 3.0603e+06, 7.94143e+06]),
    # ]
    # Lognormal
    # filename = "lognormal-speedups.pdf"
    # benchmark_data = [
    #     # ("-10 <= x <= 10", [21.7356, 53.5, 119.981, 194.879, 444.603, 786.207, 1412.71, 3985.14, 4974.74, 10172.7, 12217.7, 54602.9, 62423.7, 125658, 236648, 542176, 870496, 1.67988e+06, 3.52552e+06, 7.98439e+06]),
    #     # ("x == 42", [22.988, 56.2424, 110.983, 207.452, 415.726, 859.736, 1513.04, 3984.11, 5295.64, 8503.39, 20025.2, 31282.8, 62506.2, 103781, 273021, 878782, 1.27524e+06, 1.68605e+06, 3.71709e+06, 1.1376e+07]),
    #     # ("abs(x) <= 10", [30.3065, 53.8571, 103.823, 222.983, 443.603, 1012.58, 1320.69, 3981.51, 3405.64, 9181.08, 22035.6, 40261, 54065.3, 162586, 327323, 418691, 989841, 2.18293e+06, 4.26043e+06, 6.32641e+06]),
    #     # ("x^2 <= 100", [25.16, 60.1452, 111.121, 257.74, 415.387, 786.414, 1633.92, 2352.43, 3564.52, 12548, 21972.7, 44570.9, 87611, 176965, 355544, 504840, 1.28513e+06, 2.55883e+06, 3.71003e+06, 1.03071e+07]),
    #     # ("round(x) == 10", [28.6515, 53.1429, 103.952, 195.455, 416.452, 843.981, 1514.26, 2996, 8004.14, 8536.02, 15477.9, 31635.4, 100034, 145749, 207880, 497844, 1.40202e+06, 2.62106e+06, 3.73657e+06, 8.63008e+06]),
    #     # ("x^2 - 4 * x + 3 <= 0", [28.6515, 59.5323, 119.593, 222.552, 444.034, 990.478, 1634.5, 4087.88, 5140.27, 10205.4, 24281.5, 28096.6, 81359.4, 178759, 392110, 467886, 1.31458e+06, 2.35661e+06, 3.07476e+06, 1.41749e+07]),
    #     # ("sqrt(abs(x)) <= sqrt(10)", [32.5345, 56.5, 111.19, 222.121, 476.778, 1012.69, 1992.37, 3644.03, 6210.7, 8998.16, 16799.1, 44585.3, 68584.5, 92579.8, 266570, 350815, 879058, 2.01494e+06, 3.97221e+06, 7.50748e+06]),
    #     # ("abs(x - u) > s * 3", [5.58743, 7.61434, 18.5175, 16.0335, 34.0917, 47.7825, 79.6847, 88.1435, 110.028, 132.186, 130.7, 172.466, 183.179, 192.976, 199.021, 197.301, 196.565, 164.311, 120.579, 78.1464]),
    #     ("-10 <= x <= 10", [10.6207, 16.4018, 19.7166, 56.4574, 48.1908, 78.973, 161.682, 231.25, 308.731, 665.486, 680.76, 1233.16, 1242.72, 1234.15, 1250.6, 1436.55, 1372.03, 1543.14, 1436.26, 1285.98]),
    #     ("x == 42", [13.8182, 20.8736, 43.6265, 103.329, 151.926, 366.241, 305.021, 936.398, 1983.04, 3835.4, 6495.71, 13650.9, 22194.3, 20189.9, 31938.9, 70148.9, 122759, 148091, 252214, 469230]),
    #     ("abs(x) <= 10", [14.4355, 16.2946, 20.5307, 58.128, 43.7177, 76.6913, 136.929, 256.287, 348.716, 659.818, 603.632, 1081.26, 1217.98, 1557.4, 1345.65, 1332.75, 1376.53, 1482.61, 1401.17, 1312.04]),
    #     ("x^2 <= 100", [13.2714, 16.0714, 20.4134, 59.6364, 46.7147, 76.8645, 129.402, 238.06, 335.621, 671.104, 709.41, 1185.02, 1238.58, 1256.12, 1230.28, 1243.22, 1319.33, 1464.3, 1400.08, 1343.17]),
    #     ("round(x) == 10", [12.16, 22.9367, 39.6484, 83.2759, 151.705, 313.284, 376.108, 1020.46, 1020.8, 2227.36, 3054.19, 7047.31, 10298.7, 11574.3, 18963.5, 40646.8, 63552.9, 98317.6, 150832, 129449]),
    #     ("x^2 - 4 * x + 3 <= 0", [11.5443, 21.6747, 37.9789, 91.2911, 158.648, 231.977, 346.812, 621.743, 1538.04, 1982.17, 2511.05, 3982.49, 5845.94, 10908.5, 14254, 24042.4, 40442.4, 63284.8, 72256.7, 65102.1]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [13.9394, 16.8981, 19.5615, 52.9197, 45.3271, 56.8548, 134.12, 228.862, 309.66, 624.993, 625.85, 1169.88, 1126.2, 1426.99, 1440.1, 1287.1, 1366.11, 1481.47, 1218.54, 1337.78]),
    #     ("abs(x - u) > s * 3", [3.70251, 5.46216, 7.52734, 13.7473, 20.8212, 23.1591, 40.1874, 37.7666, 50.3218, 51.0199, 55.0447, 54.1448, 54.3702, 53.0934, 62.0524, 55.8918, 49.9007, 49.4327, 37.9312, 33.1164]),
    # ]
    # Cauchy
    # filename = "cauchy-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [11.7258, 18.4051, 20.0407, 27.7256, 50.7888, 77.8535, 131.219, 198.398, 298.171, 381.244, 516.694, 569.337, 750.142, 706.061, 763.589, 804.108, 817.464, 861.106, 831.562, 811.494]),
    #     ("x == 42", [12.3879, 23.85, 49.2414, 88.1466, 262.76, 301.085, 588.617, 1142.79, 1986.57, 3466.12, 7285.83, 11832.3, 20612.2, 48090.8, 77499.4, 152939, 266389, 365367, 669497, 1.29265e+06]),
    #     ("abs(x) <= 10", [12.3534, 16.9824, 19.6169, 29.9507, 48.7707, 77.6919, 134.31, 193.211, 276.314, 416.064, 580.347, 600.728, 796.532, 630.028, 721.453, 795.446, 790.129, 807.899, 824.349, 812.138]),
    #     ("x^2 <= 100", [11.876, 15.6612, 12.3495, 28.8156, 50.5683, 78.9668, 127.729, 188.771, 260.766, 402.55, 444.474, 708.769, 601.526, 606.565, 905.422, 678.192, 783.94, 829.19, 819.824, 773.511]),
    #     ("round(x) == 10", [12.3879, 23.7167, 46.7249, 88.2845, 81.9283, 163.605, 249.374, 391.976, 660.994, 951.542, 1718.32, 2457.72, 4238.95, 8868.88, 9989.3, 11706.3, 12260.9, 12318.1, 17444.1, 14142.3]),
    #     ("x^2 - 4 * x + 3 <= 0", [11.464, 22.896, 40.1316, 79.3256, 109.615, 146.658, 190.578, 281.818, 387.393, 709.243, 996.355, 1610.08, 2769.25, 3949.87, 5562.27, 4503.99, 5016.19, 7930.22, 7428.61, 7524.72]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [12.3879, 18.1709, 19.8452, 28.5221, 48.4295, 75.6989, 100.639, 170.314, 269.873, 386.292, 506.211, 750.702, 668.937, 821.124, 739.336, 800.31, 787.499, 815.672, 802.62, 764.937]),
    #     ("abs(x - u) > s * 3", [4.24267, 5.66245, 9.40131, 15.8964, 24.5493, 42.2001, 50.4061, 60.0445, 77.5007, 63.9039, 82.9453, 88.4367, 90.9358, 86.1784, 93.8416, 79.6343, 90.8116, 89.0416, 37.3923, 35.9119]),
    # ]
    # Uniform count
    # filename = "uniform-count-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [15.3382, 21.9805, 80.5527, 307.202, 249.791, 378.724, 365.962, 823.976, 948.922, 1154.25, 1289.35, 1373.09, 1274.58, 1459.59, 1445.14, 1456.61, 1472.54, 1483.15, 1471.1, 1423.77]),
    #     ("x == 42", [7.88, 10.784, 50.4481, 92.5275, 526.069, 280.758, 696.705, 1099.49, 2325.18, 3876.18, 6545.55, 14030.6, 26217.7, 51352.5, 96668.3, 185223, 381206, 616422, 1.2917e+06, 2.48861e+06]),
    #     ("abs(x) <= 10", [7.87435, 19.2082, 52.4491, 70.5565, 126.029, 206.247, 265.611, 402.471, 489.032, 606.907, 628.038, 651.625, 715.258, 719.002, 726.564, 773.374, 783.138, 776.632, 784.118, 751.792]),
    #     ("x^2 <= 100", [7.80628, 16.7739, 51.25, 80.5093, 77.398, 229.133, 283.902, 394.45, 507.685, 592.26, 609.506, 664.119, 689.79, 739.272, 703.835, 813.95, 790.241, 780.883, 784.921, 748.599]),
    #     ("round(x) == 10", [7.07179, 24.0889, 42.0906, 92.9011, 187.917, 313.343, 404.26, 1011.82, 1754.44, 2717.16, 4998.98, 6537.5, 9588.5, 11943.1, 13030.2, 14262.4, 14640.4, 15184.8, 15102.3, 14501.8]),
    #     ("x^2 - 4 * x + 3 <= 0", [6.78974, 17.5481, 36.2466, 75.5536, 96.3631, 239.781, 401.306, 474.103, 1099.27, 1565.26, 2078.31, 3423.39, 4536.73, 5345.08, 6352.1, 6343.22, 7108.02, 7427.07, 7577.47, 7508.62]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [8.06373, 16.7238, 51.4545, 70.064, 123.078, 211.031, 268.323, 359.891, 452.851, 591.652, 666.33, 703.098, 725.735, 761.572, 757.888, 754.297, 776.821, 778.219, 778.482, 755.534]),
    #     ("abs(x - u) > s * 3", [19.6265, 44.5537, 107.25, 195.471, 431.24, 1162.21, 1383.39, 2949.32, 6597.85, 9333.76, 18632.4, 33200.5, 66154.9, 211350, 423065, 558728, 1.19509e+06, 2.23447e+06, 3.96001e+06, 1.1235e+07]),
    # ]
    # Uniform count + agg
    # filename = "uniform-count-agg-speedups.pdf"
    # benchmark_data = [
    #     ("-10 <= x <= 10", [21.977, 24.1168, 66.4018, 114.426, 234.695, 396.935, 676.059, 1419, 2438.44, 4239.76, 8217.4, 15235.5, 27765.8, 50110.5, 110215, 182273, 329710, 613331, 1.22837e+06, 2.36048e+06]),
    #     ("x == 42", [14.9487, 27.759, 53.1494, 91.8848, 232.22, 346.767, 577.158, 1076, 2324.14, 3939.96, 7450.14, 13358.1, 26861.8, 54832.4, 102446, 195109, 342920, 682434, 1.28423e+06, 2.24933e+06]),
    #     ("abs(x) <= 10", [15.3418, 16.7467, 49.4444, 64.0742, 147.407, 250.53, 412.632, 686.361, 1304.31, 2285.53, 4323.44, 7877.13, 14516.6, 28164.9, 58367.9, 98059.1, 192158, 384419, 647771, 1.19302e+06]),
    #     ("x^2 <= 100", [16.7746, 23.1019, 51.0947, 78.1969, 142.491, 248.913, 365.667, 685.065, 1325.37, 2505.36, 4569.95, 8264.5, 14836.5, 26805.9, 63770.7, 101506, 191641, 367916, 655705, 1.2195e+06]),
    #     ("round(x) == 10", [17.5455, 27.8072, 44.3077, 97.5306, 162.77, 297.976, 468.166, 960.39, 1753.27, 3175.24, 6726.15, 11316.7, 26291.4, 42506.7, 65334.6, 133672, 250710, 458949, 924484, 1.74788e+06]),
    #     ("x^2 - 4 * x + 3 <= 0", [15.16, 16.1448, 33.6058, 70.112, 106.81, 205.005, 377.747, 485.847, 1047.12, 1585.77, 2704.03, 5284.25, 9800.15, 16001.8, 28210.2, 42267.3, 96334.9, 176692, 327098, 592357]),
    #     ("sqrt(abs(x)) <= sqrt(10)", [15.9467, 17.2069, 46.75, 68.9237, 137.632, 373.738, 353.692, 641.075, 1233.43, 2254.44, 3988.38, 7565.82, 13468.6, 26238.6, 55189.7, 102829, 168582, 301854, 664243, 966829]),
    #     ("abs(x - u) > s * 3", [23.8776, 51.5556, 45.9464, 220.823, 478.758, 709.654, 1408.69, 3644.92, 5391.58, 8703.08, 26660, 35030.4, 86918.7, 263048, 250821, 603538, 1.12835e+06, 2.59007e+06, 5.57852e+06, 1.11997e+07]),
    # ]

    # 2D uniform filters
    # filename = "2D-uniform-speedups.pdf"
    # benchmark_data = [
    #     ("abs(x - y) <= 10", [1.92558, 1.9822, 2.7526, 3.70408, 5.21994, 7.01555, 9.84808, 13.2863, 13.5144, 18.1106, 19.5384, 19.6378, 20.9164, 24.9453, 25.3749, 32.356, 38.4207, 30.5715, 38.3264, 28.2067]),
    #     ("abs(x + y) <= 10", [1.90835, 2.24681, 2.82442, 3.64605, 5.00395, 6.76053, 10.3281, 11.2688, 15.3365, 18.2497, 20.1509, 23.4004, 19.7611, 31.1993, 33.5719, 37.3856, 40.3451, 42.036, 32.911, 40.2959]),
    #     ("x^2 + y^2 <= 100", [10.4599, 8.00282, 18.3017, 32.0988, 54.623, 94.5, 100.018, 128.616, 175.557, 320.888, 417.791, 787.969, 1238.11, 2050.62, 2800.82, 2358.23, 3739, 5002.28, 7039.6, 7265.25]),
    # ]
    #benchmark_data = [
    #    ("abs(x - y) <= 1", [2.28257, 2.45208, 4.03682, 4.57497, 6.93354, 8.7877, 11.269, 19.064, 22.1018, 33.529, 40.9162, 48.6059, 49.6361, 75.1572, 59.4984, 60.9436, 76.1018, 90.5013, 107.39, 144.863]),
    #    ("abs(x + y) <= 1", [2.03709, 2.9743, 3.925, 4.49379, 4.64067, 7.3227, 10.735, 16.771, 24.1081, 31.7993, 40.7497, 58.0564, 62.6511, 72.1707, 59.0321, 73.1941, 88.4843, 111.143, 133.206, 164.753]),
    #    ("x^2 + y^2 <= 10", [9.88276, 10.5116, 18.4821, 40.62, 54.3584, 109.796, 204.764, 321.126, 334.576, 787.216, 621.68, 1472.38, 1937.58, 3593.47, 5360.94, 9712, 14857.4, 20406.5, 13710.6, 40552.6]),
    #    ("COUNT(abs(x - y) <= 1)", [2.56783, 2.75667, 4.92857, 5.472, 8.22766, 10.0687, 14.5174, 23.1282, 29.1501, 34.7735, 47.7445, 53.2059, 71.9404, 79.7839, 53.7382, 81.1756, 90.4653, 137.974, 152.506, 210.833]),
    #    ("COUNT(abs(x + y) <= 1)", [2.55789, 4.20166, 3.9375, 6.35057, 7.34282, 9.79079, 14.1636, 19.4633, 27.3147, 34.8554, 40.9309, 56.0646, 57.8372, 64.7521, 76.8672, 92.2995, 114.175, 141.741, 187.081, 242.161]),
    #    ("COUNT(x^2 + y^2 <= 10)", [10.7829, 9.99074, 20.5, 34.7672, 67.2756, 132.545, 230.273, 321.101, 485.56, 949.936, 937.568, 1568.96, 2752.15, 5866.75, 9766.91, 16983.5, 31008.3, 42705.2, 72200.3, 112087]),
    #    ("aug COUNT(abs(x - y) <= 1)", [2.75152, 2.94, 3.64769, 6.27119, 8.90756, 11.4199, 14.7595, 24.6812, 29.5648, 36.1551, 46.8789, 50.0358, 62.3618, 61.9241, 64.1359, 79.3949, 98.0777, 128.301, 149.499, 205.479]),
    #    ("aug COUNT(abs(x + y) <= 1)", [2.50609, 3.44407, 4.15735, 7.04708, 7.92345, 8.35926, 13.9929, 19.9534, 29.0697, 34.8649, 47.4343, 50.6926, 59.8492, 71.64, 74.9425, 90.8408, 114.661, 148.419, 181.32, 253.6]),
    #    ("aug COUNT(x^2 + y^2 <= 10)", [10.5188, 10.0962, 17.5603, 37.3426, 68.8605, 126.536, 230.896, 326.486, 507.455, 961.172, 894.574, 1599.81, 3129.64, 5574.36, 9993.4, 17203.2, 25976.1, 47274, 64980.5, 124944]),
    #]

    # Redwood numbers, count uniform
    filename = "1d-count-agg-uniform.pdf"
    benchmark_data = [
        # ("range_query", [6.78846, 10.5857, 23.6491, 69.4697, 156.08, 377.134, 526.985, 889.113, 1075.94, 1323.5, 1666.76, 1853.92, 1887.54, 2001.1, 1634.4, 1739.63, 1946.77, 2072.2, 2045.4, 2022.5]),
        # ("eq_query", [8.81818, 13.85, 26.3095, 50.7442, 127.706, 157.309, 352.34, 701.404, 1351.13, 2186.32, 4443.39, 7089.51, 13299.1, 22004.8, 47388.3, 91137.9, 162132, 356120, 579487, 1.26348e+06]),
        # ("abs_query", [8.05263, 10.0286, 23.9091, 35.8, 59.6098, 113.624, 154.484, 223.78, 303.688, 347.781, 432.394, 449.521, 500.087, 493.737, 500.629, 519.765, 507.608, 511.309, 516.032, 506.254]),
        # ("sqr_query", [8.16216, 10.0145, 22.193, 37.1385, 56.186, 121.663, 148.068, 238.03, 312.064, 368.022, 462.689, 468.359, 482.234, 474.316, 479.538, 502.291, 491.017, 501.061, 534.492, 519.206]),
        # ("round_query", [7.19481, 13.2169, 22.2347, 39.6455, 76.7456, 145.175, 192.508, 443.268, 767.514, 1382.31, 2410.63, 3992.94, 7502.61, 9819.23, 12112.2, 14736.8, 15510.3, 17087.7, 17672.1, 18928.3]),
        # ("poly_query", [6.76744, 7.78667, 16, 32.4265, 45.602, 96.8478, 183.438, 205.318, 445.367, 719.79, 1032.25, 1827.53, 2670.06, 3208.6, 3844.37, 4050.08, 4276.28, 4656.11, 4611.55, 4869.96]),
        # ("sqrt_query", [6.35294, 8.70787, 22.3226, 30.6591, 55.3956, 100.942, 136.58, 202.498, 280.542, 343.948, 422.501, 474.314, 493.756, 516.395, 554.715, 528.844, 533.259, 535.403, 545.422, 541.852]),
        # ("stddev_query", [13.2381, 26.65, 55, 103.238, 218.211, 410.85, 870.632, 1646.24, 3451.7, 6880.58, 12038.3, 24634.1, 50180.2, 111090, 201458, 356125, 612742, 1.50045e+06, 3.00788e+06, 5.12868e+06]),
        ("range_query", [8.91111, 10.4405, 31.4815, 55.1765, 156.701, 497.667, 655.431, 1470.8, 2783.11, 5224.83, 10092.2, 19020.1, 29295.3, 56714.2, 116888, 172870, 305642, 685654, 1.4103e+06, 2.75774e+06]),
        ("eq_query", [8.84848, 14.6364, 27.55, 50.6977, 128.618, 151.702, 360.294, 639.481, 1233.02, 2301.88, 4678.1, 7799.87, 14647.8, 24671.9, 51900.9, 97197.9, 170608, 367293, 590439, 1.22766e+06]),
        ("abs_query", [7.5122, 10.3625, 22.9464, 34.8714, 63.8553, 131.89, 146.723, 360.953, 665.164, 1185.95, 2254.75, 4310.67, 7516.78, 14576.6, 32062.3, 50003.2, 92059, 184398, 359225, 721206]),
        ("sqr_query", [8.35135, 10.6, 24.0364, 36.4783, 66.4595, 133.453, 146.859, 371.187, 571.25, 1135.88, 2085.28, 3985.08, 6682.8, 13819.1, 30373.7, 49115.3, 86727.1, 182649, 352822, 684753]),
        ("round_query", [6.11712, 11.2439, 18.936, 37.0703, 64.7143, 122.416, 163.857, 379.97, 630.477, 1162.78, 2156.94, 3835.08, 9518.79, 14421.9, 24384.2, 46315.5, 83608.9, 167900, 320699, 593961]),
        ("poly_query", [5.94737, 7.52809, 15.5915, 31.2113, 46.2784, 100.068, 186.606, 167.081, 428.098, 731.314, 1080.69, 2136.36, 3994.21, 7824.33, 13977.3, 24835.3, 42190.6, 84808.6, 150095, 286555]),
        ("sqrt_query", [5.65152, 8.40741, 23.3438, 32.8765, 56.1222, 97.6476, 124.099, 292.777, 539.696, 904.269, 1929.66, 3333.35, 5308.19, 12293.5, 26769.9, 43604.3, 79786.8, 156963, 309013, 572283]),
        ("stddev_query", [12, 27.125, 55.2, 109.05, 219.75, 432.6, 823.571, 1644.33, 3005, 6614.95, 13170.4, 26342.2, 48655.7, 93602.7, 204734, 407195, 753158, 1.64915e+06, 3.13303e+06, 5.98839e+06]),
    ]

    # plot_speedups(problem_sizes, benchmark_data, filename)

    # abs(a.x - b.x) < 0.1
    # join_data = [
    #     (256, 72260, 40192, 16079),
    #     (512, 282467, 91972, 47203),
    #     (1024, 1123798, 223903, 138832),
    #     (2048, 4307993, 558566, 385803),
    #     (4096, 17057687, 1218542, 1155337),
    #     (8192, 68629747, 2879897, 3407198),
    #     (16384, 274842284, 6361664, 9902301),
    #     (32768, 1102898960, 14615374, 28394981),
    #     (65536, 4411367630, 33020261, 84401328),
    #     (131072, 17661258899, 76528929, 244106540),
    #     (262144, 71440646577, 169291628, 737583273),
    #     (524288, 283311176987, 373083121, 2195300718),
    # ]

    # # TODO: divide the second/third/fourth values in the tuple by 10^9
    # join_data = [(s, n / 1e6, b / 1e6, q / 1e6) for (s, n, b, q) in join_data]

    # plot_join_speedups(join_data, "absd_join.pdf")

    join_data = [
        # Redwood
        # (256, 72260, 40192, 16079),
        # (512, 282467, 91972, 47203),
        # (1024, 1123798, 223903, 138832),
        # (2048, 4307993, 558566, 385803),
        # (4096, 17057687, 1218542, 1155337),
        # (8192, 68629747, 2879897, 3407198),
        # (16384, 274842284, 6361664, 9902301),
        # (32768, 1102898960, 14615374, 28394981),
        # (65536, 4411367630, 33020261, 84401328),
        # (131072, 17661258899, 76528929, 244106540),
        # (262144, 71440646577, 169291628, 737583273),
        # (524288, 283311176987, 373083121, 2195300718),
        # NV xysort
        # (256, 79015, 49301, 23080),
        # (512, 296281, 111223, 62537),
        # (1024, 1203997, 264957, 199084),
        # (2048, 5112924, 696601, 579050),
        # (4096, 20091245, 1540622, 1786819),
        # (8192, 81233328, 3597652, 5394450),
        # (16384, 315869800, 8010781, 15541355)
        # NV xsort
        # (256, 148677, 44814, 8017),
        # (512, 579720, 87188, 16412),
        # (1024, 1523370, 105146, 25330),
        # (2048, 7235643, 250391, 60643),
        # (4096, 29251171, 577844, 150074),
        # (8192, 115247409, 1172098, 322740),
        # (16384, 410399194, 2119070, 833044),
        # NV x sort + single join
        # (256, 121602, 34870, 12161, 5288),
        # (512, 400525, 36845, 29948, 9787),
        # (1024, 1704972, 70693, 75090, 22055),
        # (2048, 6806057, 179564, 195398, 42417),
        # (4096, 26836733, 325212, 511707, 103967),
        # (8192, 106598268, 709346, 1354618, 380343),
        # (16384, 447743803, 1705924, 3233912, 826778),
        # (32768, 1679621374, 3358989, 7769758, 2432420),
        # (65536, 7040297950, 8754361, 20404731, 7455322),
        # (131072, 27712228212, 17295744, 49279661, 26107396),

        # (256, 158212, 22435, 14488, 4907),
        # (512, 333645, 32633, 35468, 10089),
        # (1024, 1565591, 73926, 96379, 28686),
        # (2048, 4913234, 159015, 196836, 62136),
        # (4096, 19643892, 325770, 501666, 108169),
        # (8192, 78011230, 716467, 1244656, 267385),
        # (16384, 308046972, 1507782, 3143845, 814728),
        # (32768, 1230241369, 3245641, 8361954, 2606991),
        # (65536, 5433083114, 7608248, 19993623, 7725709),
        # (131072, 20551899932, 16037536, 43164260, 26462298),

        # (256, 97882, 48982, 8999, 5171),
        # (512, 426451, 58800, 25302, 17140),
        # (1024, 1573123, 86819, 73955, 19290),
        # (2048, 6150111, 174219, 216474, 42474),
        # (4096, 24318362, 386536, 565936, 113908),
        # (8192, 97597118, 826455, 1283013, 288381),
        # (16384, 386507295, 1634711, 2980161, 624868),
        # (32768, 1541055473, 3494444, 7361962, 1644155),
        # (65536, 6084745449, 7628233, 15909707, 4068740),
        # (131072, 23055398685, 15153340, 37956472, 17754672),
        # (262144, 95787067960, 32377159, 96158080, 87690948),
        # (524288, 380740657664, 72070898, 219010531, 967240807),
        # (1048576, 1309441894924, 150969259, 659162350, 2902636571),

        # (256, 154396, 20389, 15907, 8946),
        # (512, 402133, 43954, 36898, 11364),
        # (1024, 1590588, 75643, 89794, 38271),
        # (2048, 6348421, 164102, 225129, 38420),
        # (4096, 25733547, 367059, 505020, 79628),
        # (8192, 103118583, 794241, 1050489, 237797),
        # (16384, 411847261, 1669152, 2284297, 495928),
        # (32768, 1674747012, 3563463, 4836002, 823800),
        # (65536, 6987493305, 8113216, 10835468, 1566707),
        # (131072, 26796687898, 15560003, 25546868, 3447784),
        # (262144, 108922537053, 34677219, 53655919, 7148533),
        # (524288, 433958424414, 73091891, 124283236, 17115182),

        (256, 103616, 15962, 14935, 5160),
        (512, 372099, 37643, 32845, 9118),
        (1024, 1361251, 59889, 72774, 17671),
        (2048, 5546464, 125325, 161808, 35471),
        (4096, 22439438, 273730, 366697, 73464),
        (8192, 89138350, 599691, 809158, 134211),
        (16384, 358489837, 1265852, 1763541, 253481),
        (32768, 1428950512, 2726797, 3853308, 508170),
        (65536, 5716404133, 5873947, 8413501, 1041846),
        (131072, 21079740278, 12142194, 18466053, 2224876),
        (262144, 83629531550, 25280658, 41191904, 4966844),
        (524288, 328020614719, 53695998, 95086948, 11986116),
        (1048576, 1314440288434, 113096673, 229186909, 33864286),

    ]

    plot_speedups_over_nested(join_data, "absd_join.pdf", "Absolute Difference Join")

    chebyshev_data = [
        (256, 80671, 48334, 31199, 26248),
        (512, 297732, 116051, 88948, 77141),
        (1024, 1623412, 265423, 280198, 223903),
        (2048, 6644616, 741839, 811189, 750292),
        (4096, 25624770, 1632143, 2845764, 2319554),
        (8192, 106337381, 3900133, 9421325, 5796208),
        (16384, 370751909, 9332519, 22915367, 15401104),
        (32768, 1671305400, 19932874, 83314258, 53883721),
        (65536, 5302545226, 42547417, 204689617, 120557899),
        (131072, 21930876262, 99617066, 627647049, 398696511),
        (262144, 81684275721, 171209802, 1267879236, 685268393),
        (524288, 326959472758, 375757099, 3500890363, 1897006998),
        (1048576, 1305345743874, 831299024, 10526197895, 5463122310),
    ]
    plot_speedups_over_nested(chebyshev_data, "chebyshev.pdf", "Chebyshev Join")

    donut_data = [
        (256, 118505, 40369, 19863, 12074),
        (512, 455117, 98862, 47918, 19668),
        (1024, 1770213, 209928, 109573, 47000),
        (2048, 7048294, 527376, 222529, 86742),
        (4096, 28153938, 1227049, 509523, 161343),
        (8192, 113895148, 2825074, 1135263, 332603),
        (16384, 453958683, 6409502, 2540319, 654870),
        (32768, 1818525800, 14771765, 5505763, 1276762),
        (65536, 7274952627, 33623657, 12248558, 2599406),
        (131072, 29106472711, 77281318, 26258038, 5127426),
        (262144, 116553417270, 171209802, 57908932, 10620254),
        (524288, 466124667357, 375757099, 126003265, 21090289),
        (1048576, 1843728642522, 831299024, 287175502, 43021638),
    ]
    plot_speedups_over_nested(donut_data, "donut.pdf", "Donut Join")

    euc_data = [
        (256, 89741, 40369, 14772, 8912),
        (512, 344990, 98862, 35258, 16553),
        (1024, 1328363, 209928, 79143, 40439),
        (2048, 5237976, 527376, 169422, 75727),
        (4096, 21004261, 1227049, 373446, 138207),
        (8192, 84500437, 2825074, 861143, 281687),
        (16384, 335180154, 6409502, 1958833, 565174),
        (32768, 1339713890, 14771765, 4312916, 1097134),
        (65536, 5361791008, 33623657, 9574006, 2242592),
        (131072, 21453091628, 77281318, 20549458, 4417971),
        (262144, 86223667794, 171209802, 45590754, 9184993),
        (524288, 344413347010, 375757099, 100061961, 18202540),
        (1048576, 1373815078853, 831299024, 228100063, 36959691),
    ]
    plot_speedups_over_nested(euc_data, "euclidean.pdf", "Euclidean Distance Join")

    manh_data = [
        (256, 79270, 40369, 12230, 7017),
        (512, 285916, 98862, 33819, 13246),
        (1024, 1114800, 209928, 71375, 34016),
        (2048, 4404349, 527376, 167243, 61468),
        (4096, 17653379, 1227049, 370678, 117676),
        (8192, 70801856, 2825074, 846362, 229515),
        (16384, 282412117, 6409502, 1880751, 451297),
        (32768, 1133386088, 14771765, 4105981, 871043),
        (65536, 4527130564, 33623657, 8976407, 1782211),
        (131072, 18157169294, 77281318, 19522493, 3534548),
        (262144, 73041633337, 171209802, 42943159, 7316946),
        (524288, 292207104761, 375757099, 95118710, 14538484),
        (1048576, 1158905683029, 831299024, 212403686, 29571783),
    ]
    plot_speedups_over_nested(manh_data, "manhattan.pdf", "Manhattan Distance Join")
