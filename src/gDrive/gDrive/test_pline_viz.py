# test_spline_viz.py
import numpy as np
import plotly.graph_objects as go
from spline_generator import FastSplineGenerator

def main():
    # 1. Setup
    generator = FastSplineGenerator()
    
    # Define Waypoints (An S-Curve)
    waypoints = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [4.0, 2.0],
        [4.0, 4.0]
    ]
    
    print("Generating Spline...")
    # Generate the path
    # Avg Speed 1.0 m/s, Resolution 0.05m
    path = generator.generate(waypoints, avg_speed=1.0, ds=0.05)
    
    if path is None:
        print("Error generating path.")
        return

    # Extract Data for Plotting
    path_x = path[:, 0]
    path_y = path[:, 1]
    path_v = path[:, 3] # Velocity magnitude
    
    wp_x = [p[0] for p in waypoints]
    wp_y = [p[1] for p in waypoints]

    # 2. Create Plotly Figure
    fig = go.Figure()

    # Trace 1: The Spline Path (Colored by Speed)
    fig.add_trace(go.Scatter(
        x=path_x, 
        y=path_y,
        mode='lines+markers',
        name='Quintic Spline',
        marker=dict(
            size=4,
            color=path_v, # Color points by velocity
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Speed (m/s)")
        ),
        line=dict(width=2, color='blue') # Fallback color
    ))

    # Trace 2: The Original Waypoints (Red Dots)
    fig.add_trace(go.Scatter(
        x=wp_x, 
        y=wp_y,
        mode='markers+text',
        name='Waypoints',
        marker=dict(size=12, color='red', symbol='x'),
        text=[f"WP{i}" for i in range(len(waypoints))],
        textposition="top center"
    ))

    # 3. Quiver Plot (Arrows for Heading) - Sampled every 10th point
    skip = 10
    fig.add_trace(go.Cone(
        x=path_x[::skip],
        y=path_y[::skip],
        u=np.cos(path[::skip, 2]), # cos(theta)
        v=np.sin(path[::skip, 2]), # sin(theta)
        w=np.zeros_like(path_x[::skip]), # Z-component (0 for 2D)
        sizemode="absolute",
        sizeref=0.2,
        anchor="tail",
        showscale=False,
        name='Heading',
        colorscale=[[0, 'black'], [1, 'black']]
    ))

    # Layout Settings
    fig.update_layout(
        title="Quintic Hermite Spline Test",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1), # Equal Aspect Ratio
        template="plotly_white",
        width=900,
        height=700
    )

    print("Opening Plotly visualization...")
    fig.show()

if __name__ == "__main__":
    main()