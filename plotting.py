
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

from ipywidgets import interact, FloatSlider
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Own imports
from constants import COORDS_DF
from utils import read_data_file, select_df_subset, combination_to_string, VARIABLE_NAMES


def plot_bridge_3d_structure(highlight_nodes=None, color_scale: dict=None, s: int=4, annotations: dict=None):
    """
    Plots the bridge structure in 3D using Plotly.
    Optionally highlights nodes in highlight_nodes, or uses a color scale if provided.
    Args:
        highlight_nodes (list, optional): List of node numbers to highlight.
        color_scale (dictionary-like, optional): Dictionary of {node_number: value}
            of values bewteen 0 and 1 to use for coloring nodes.
        annotations (dict, optional): Dictionary mapping node_number -> string to append to hover text.
    """
    if color_scale is not None:
        # color_scale is a dict: {node_number: value}, nodes not present get value np.nan
        node_numbers = COORDS_DF["Node Number"].values
        node_colors = [color_scale.get(nn, np.nan) for nn in node_numbers]

        hover_text = []
        for nn, val in zip(node_numbers, node_colors):
            if pd.isna(val):
                txt = f"Node: {nn}<br>Value: NaN"
            else:
                txt = f"Node: {nn}<br>Value: {val:.3e}"
            if annotations and nn in annotations:
                txt += f"<br>{annotations[nn]}"
            hover_text.append(txt)

        fig = go.Figure(data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s*2,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title="Value"),
                showscale=True,
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        fig.update_layout(
            title="Bridge Structure (colored by value)",
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    else:
        # Default color for all nodes
        colors = ['black'] * len(COORDS_DF)
        # Highlight specified nodes
        if highlight_nodes is not None:
            node_indices = COORDS_DF[COORDS_DF["Node Number"].isin(highlight_nodes)].index
            for idx in node_indices:
                colors[idx] = 'red'

        hover_text = []
        for nn in COORDS_DF["Node Number"]:
            txt = f"Node: {nn}"
            if annotations and nn in annotations:
                txt += f"<br>{annotations[nn]}"
            hover_text.append(txt)

        fig = go.Figure(data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s,
                color=colors,
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        fig.update_layout(
            title="Bridge Structure (highlighted nodes in red)",
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    fig.write_html("visualization/bridge_structure_3d.html", auto_open=True)
    # fig.show()


def plot_bridge_3d_variable_over_time_combination(combination, s: int = 4, log_scale: bool = False):
    """
    Loads the dataframe for the given combination and visualizes a variable over time in 3D.

    Args:
        combination (tuple): (train_config, load, season, health, variable)
        var_name (str): Name of the column to visualize.
        s (int): Marker size.
        log_scale (bool): If True, apply log10 to the variable values for color mapping.
    """
    df = read_data_file(*combination, filter_out_invalid_nodes=True)
    var_name = VARIABLE_NAMES[combination[-1]]
    plot_bridge_3d_variable_over_time_df(
        df, var_name, s=s, log_scale=log_scale, title=combination_to_string(combination)
    )


def plot_bridge_3d_variable_over_time_df(df, var_name, s: int = 4, log_scale: bool = False, title: str = None):
    """
    Visualizes a given variable for all nodes over all time steps in 3D using Plotly.
    Adds a slider to select the time step instead of playback animation.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        var_name (str): Name of the column to visualize.
        s (int): Marker size.
        log_scale (bool): If True, apply log10 to the variable values for color mapping.
    """

    time_points = np.sort(df["time"].unique())

    def get_node_vals(df_t):
        vals = [
            df_t[df_t["Node Number"] == nn][var_name].values[0]
            if nn in df_t["Node Number"].values else np.nan
            for nn in COORDS_DF["Node Number"]
        ]
        if log_scale:
            vals = [np.log10(val) if not np.isnan(val) and val > 0 else np.nan for val in vals]
        return vals

    frames = []
    for i, t in enumerate(time_points):
        df_t = df[df["time"] == t]
        node_vals = get_node_vals(df_t)
        hover_text = [
            f"Node: {nn}<br>{var_name}: {val:.3e}" if not np.isnan(val) else f"Node: {nn}<br>{var_name}: NaN"
            for nn, val in zip(COORDS_DF["Node Number"], node_vals)
        ]
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=COORDS_DF["X"],
                y=COORDS_DF["Y"],
                z=COORDS_DF["Z"],
                mode='markers',
                marker=dict(
                    size=s,
                    color=node_vals,
                    colorscale='Viridis',
                    colorbar=dict(title=f"log10({var_name})" if log_scale else var_name),
                    showscale=True,
                ),
                text=hover_text,
                hoverinfo='text'
            )],
            name=str(i),
            layout=go.Layout(title_text=f"{title if title is not None else var_name} at time {t:.2f}" + (" (log scale)" if log_scale else ""))
        ))

    # Initial frame
    df0 = df[df["time"] == time_points[0]]
    node_vals0 = get_node_vals(df0)
    hover_text0 = [
        f"Node: {nn}<br>{var_name}: {val:.3f}" if not np.isnan(val) else f"Node: {nn}<br>{var_name}: NaN"
        for nn, val in zip(COORDS_DF["Node Number"], node_vals0)
    ]

    steps = []
    for i, t in enumerate(time_points):
        step = dict(
            method="animate",
            args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            label=f"{t:.2f}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        steps=steps
    )]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s,
                color=node_vals0,
                colorscale='Viridis',
                colorbar=dict(title=f"log10({var_name})" if log_scale else var_name),
                showscale=True,
            ),
            text=hover_text0,
            hoverinfo='text'
        )],
        layout=go.Layout(
            title=title if title is not None else f"{var_name} for all nodes over time" + (" (log scale)" if log_scale else ""),
            scene=dict(
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            sliders=sliders
        ),
        frames=frames
    )

    fig.write_html("visualization/bridge_3d_variable_over_time.html", auto_open=True)
    fig.show()


def plot_bridge_3d_load(combination, time_point):
    """
    Plots bridge loads in 3D for a single time point.
    Args:
        combination (tuple): (train_config, load, season, health, variable)
        time_point (float): Time value to plot
    """
    df_subset = select_df_subset(combination)
    variable_to_plot = VARIABLE_NAMES[combination[-1]]
    timestamp_subset = df_subset[df_subset["time"] == time_point]
    node_loads = [
        timestamp_subset[timestamp_subset["Node Number"] == nn][variable_to_plot].values
        for nn in COORDS_DF["Node Number"]
    ]
    node_loads_flat = [nl[0] if len(nl) > 0 else np.nan for nl in node_loads]

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        COORDS_DF["X"],
        COORDS_DF["Y"],
        COORDS_DF["Z"],
        c=node_loads_flat, marker='o', s=2,
        cmap="viridis"
    )
    fig.colorbar(scatter, shrink=0.5)
    ax.set_title(f"Bridge Loads at time {time_point}\nFor combination: {combination_to_string(combination)}")
    plt.show()


def plot_bridge_loads_3d_slider(df, combination):
    """
    Plots bridge loads in 3D with a time slider.
    Args:
        df (pd.DataFrame): Dataframe to visualize.
        combination (tuple): (train_config, load, season, health, variable)
    """
    time_points = np.sort(df["time"].unique())
    variable_to_plot = VARIABLE_NAMES[combination[-1]]

    def plot_at_time(time_point_index):
        time_point = time_points[int(time_point_index)]
        timestamp_subset = df[df["time"] == time_point]
        node_loads = [
            timestamp_subset[timestamp_subset["Node Number"] == nn][variable_to_plot].values
            for nn in COORDS_DF["Node Number"]
        ]
        node_loads_flat = [nl[0] if len(nl) > 0 else np.nan for nl in node_loads]

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            COORDS_DF["X"],
            COORDS_DF["Y"],
            COORDS_DF["Z"],
            c=node_loads_flat, marker='o', s=2,
            cmap="viridis"
        )
        fig.colorbar(scatter, shrink=0.5)
        ax.set_title(f"Bridge Loads at time {time_point}\nFor combination: {combination_to_string(combination)}")
        fig.savefig(f"../visualization/bridge_loads_3d_{time_point}.png")
        # fig.show()

    interact(
        plot_at_time,
        time_point_index=FloatSlider(
            min=0,
            max=len(time_points)-1,
            step=1,
            value=0,
            description='Time Index'
        )
    )
    

def plot_node_variation_across_scenarios(df, node_number, variable="TotalDeformation"):
    """
    Plot the variation of a variable for a given node across all scenarios and time steps.

    Args:
        node_number (int): Node number to plot.
        variable (str): Variable/column name to plot (default: "TotalDeformation").
    """

    # Filter for the node
    node_df = df[df["Node Number"] == node_number]

    # Sort by scenario and time for consistent plotting
    node_df = node_df.sort_values(["scenario", "time"])

    # Pivot for heatmap: rows=scenario, cols=time, values=variable
    pivot = node_df.pivot(index="scenario", columns="time", values=variable)

    plt.figure(figsize=(10, 6))
    plt.title(f"Variation of {variable} for Node {node_number} across scenarios and time")
    plt.xlabel("Time")
    plt.ylabel("Scenario")
    im = plt.imshow(pivot, aspect="auto", cmap="viridis", interpolation="nearest",
                    extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.max(), pivot.index.min()])
    plt.colorbar(im, label=variable)
    plt.show()

    
def visualize_voxel_snapshot(X_vol, sample_idx=0, channel_idx=4, threshold=0.01):
    """
    Visualizes a single channel of a 3D Voxel volume using a 3D scatter plot.
    
    Args:
        X_vol: The 5D tensor (N, 32, 12, 8, 8)
        sample_idx: Which bridge snapshot to look at.
        channel_idx: Which variable (0-7). Default 4 is Equivalent Stress.
        threshold: Only plot voxels with values above this (to see the bridge shape).
    """
    # 1. Extract the specific 3D volume for one channel
    # Shape becomes (32, 12, 8)
    vol_3d = X_vol[sample_idx, :, :, :, channel_idx]
    
    # 2. Get the indices of non-zero (or high value) voxels
    indices = np.where(vol_3d > threshold)
    values = vol_3d[indices]
    
    # 3. Create a temporary plotting dataframe
    # We use the voxel indices as coordinates
    plot_df = pd.DataFrame({
        'X': indices[0],
        'Y': indices[1],
        'Z': indices[2],
        'Value': values
    })
    
    # 4. Use a 3D Scatter Plot (similar to your existing visualization)
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by the physical value (e.g. Stress)
    scatter = ax.scatter(plot_df['X'], plot_df['Z'], plot_df['Y'], 
                         c=plot_df['Value'], cmap='jet', s=50, alpha=0.6)
    
    ax.set_xlabel('Length (Bins)')
    ax.set_ylabel('Width (Bins)')
    ax.set_zlabel('Height (Bins)')
    plt.colorbar(scatter, label=f'Channel {channel_idx} intensity')
    plt.title(f"Voxel Visualization: Sample {sample_idx}, Variable: {VARIABLE_NAMES[channel_idx]}")
    
    # Set equal aspect ratio to avoid bridge looking squashed
    ax.set_box_aspect((32, 8, 12)) 
    plt.show()


def plot_nodes_time_series_for_combinations(combinations, node_numbers, variable, scenario=None, health=None):
    """
    Loads dataframes for each combination and plots the time series for selected nodes.

    Args:
        combinations: Iterable of tuples, each a combination for read_data_file.
        node_numbers: List of node numbers to plot.
        variable: The variable/column name to plot.
        scenario: (Optional) Scenario ID to filter.
        health: (Optional) Health state to filter.
    """
    if type(node_numbers) == int:
        node_numbers = [node_numbers]

    n_nodes = len(node_numbers)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(15, 4 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for ax, node in zip(axes, node_numbers):
        for combo in combinations:
            df = read_data_file(*combo, filter_out_invalid_nodes=True)
            subset = df[df['Node Number'] == node]
            if health is not None:
                subset = subset[subset['health'] == health]
            if subset.empty:
                continue
            scenarios = subset['scenario'].unique() if 'scenario' in subset.columns else [None]
            for sc in scenarios:
                sc_subset = subset[subset['scenario'] == sc] if sc is not None else subset
                if scenario is not None and sc != scenario:
                    continue
                label = f"{combination_to_string(combo)} | Scenario {sc}" if sc is not None else combination_to_string(combo)
                ax.plot(sc_subset['time'], sc_subset[variable], label=label)
        ax.set_xlabel('Time')
        ax.set_ylabel(variable)
        ax.set_title(f'Time Series of {variable} for Node {node}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_nodes_time_series(df, node_numbers, variable, scenario=None, health=None):
    """
    Plots the time series of a given variable for selected nodes.
    If multiple scenarios are present, plots each scenario as a separate line.
    If multiple nodes are given, plots each in its own subplot (vertically).

    Args:
        df: DataFrame containing 'Node Number', 'time', 'scenario', and the variable to plot.
        node_numbers: List of node numbers to plot.
        variable: The variable/column name to plot.
        scenario: (Optional) Scenario ID to filter.
        health: (Optional) Health state to filter.
    """
    if type(node_numbers) == int:
        node_numbers = [node_numbers]

    n_nodes = len(node_numbers)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(15, 4 * n_nodes), sharex=True)
    if n_nodes == 1:
        axes = [axes]

    for ax, node in zip(axes, node_numbers):
        subset = df[df['Node Number'] == node]
        if health is not None:
            subset = subset[subset['health'] == health]
        if subset.empty:
            ax.text(0.5, 0.5, f"No data for Node {node} with given filters.", ha='center', va='center')
            ax.set_title(f'Node {node}')
            continue
        scenarios = subset['scenario'].unique() if 'scenario' in subset.columns else [None]
        for sc in scenarios:
            sc_subset = subset[subset['scenario'] == sc] if sc is not None else subset
            if scenario is not None and sc != scenario:
                continue
            ax.plot(sc_subset['time'], sc_subset[variable], label=f'Scenario {sc}')
        ax.set_xlabel('Time')
        ax.set_ylabel(variable)
        ax.set_title(f'Time Series of {variable} for Node {node}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()