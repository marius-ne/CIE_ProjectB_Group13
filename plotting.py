
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

from ipywidgets import interact, FloatSlider
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Own imports
from constants import COORDS_DF
from utils import read_data_file, select_df_subset, combination_to_string, VARIABLE_NAMES


def plot_bridge_3d_structure(highlight_nodes=None, color_scale: dict=None, s: int=4):
    """
    Plots the bridge structure in 3D using Plotly.
    Optionally highlights nodes in highlight_nodes, or uses a color scale if provided.
    Args:
        highlight_nodes (list, optional): List of node numbers to highlight.
        color_scale (dictionary-like, optional): Dictionary of {node_number: value}
            of values bewteen 0 and 1 to use for coloring nodes.
    """
    if color_scale is not None:
        # color_scale is a dict: {node_number: value}, nodes not present get value 0
        node_numbers = COORDS_DF["Node Number"].values
        node_colors = [color_scale.get(nn, np.nan) for nn in node_numbers]
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
            )
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

        fig = go.Figure(data=[go.Scatter3d(
            x=COORDS_DF["X"],
            y=COORDS_DF["Y"],
            z=COORDS_DF["Z"],
            mode='markers',
            marker=dict(
                size=s,
                color=colors,
            )
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
    fig.show()


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
            f"Node: {nn}<br>{var_name}: {val:.3f}" if not np.isnan(val) else f"Node: {nn}<br>{var_name}: NaN"
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
    
    