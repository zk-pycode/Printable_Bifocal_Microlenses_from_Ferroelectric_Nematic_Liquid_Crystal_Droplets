"""
src_func/debug_plots.py

Diagnostic visualisation functions called during the pipeline run.

Functions
---------
plot_thickness_profile
    Radial droplet height profile, with each point on the curve colored
    by the Michel-Levy retardation color at that thickness.

plot_director_field_layers
    A 6-panel diagnostic figure for each of n_layers horizontal (z) slices
    through the droplet, showing nx, ny, nz components, in-plane quiver,
    azimuthal angle φ, and effective birefringence Δn_eff.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from .color_chart import compute_michel_levy_color


def plot_thickness_profile(base_radius, droplet_height, z_min, avg_delta_n,
                           N_E, N_O, debug_path):
    """
    Plot the spherical-cap droplet profile with Michel-Levy colors.

    Parameters
    ----------
    base_radius : float
        Droplet base radius in metres.
    droplet_height : float
        Maximum droplet height (at r=0) in metres.
    z_min : float
        z-coordinate of the droplet base (metres).  Not used directly here
        but passed for signature consistency with the pipeline.
    avg_delta_n : float
        Column-averaged birefringence Δn (dimensionless).  Used to convert
        thickness to retardation for the color lookup.
    N_E, N_O : float
        Extraordinary / ordinary refractive indices (not directly used here;
        the retardation color is computed from avg_delta_n * D(r)).
    debug_path : str
        Output directory for the PNG file.
    """
    print("\nGenerating thickness profile plot...")

    # Sample the radial profile at 500 points from centre to edge
    n_radial_bins = 500
    r_bins = np.linspace(0, base_radius, n_radial_bins)

    # Spherical-cap height profile: D(r) = h * sqrt(1 - (r/R)²)
    thickness_radial = droplet_height * np.sqrt(1 - (r_bins / base_radius) ** 2)

    # Retardation at each radial position [nm] = avg_Δn * D(r) * 1e9
    retardation_radial = avg_delta_n * thickness_radial * 1e9   # nm

    # Convert to microns for axis labels
    r_array_um        = r_bins * 1e6
    thickness_array_um = thickness_radial * 1e6

    # Look up the Michel-Levy color for each retardation value
    rgb_profile = compute_michel_levy_color(retardation_radial)   # shape (500, 3)

    # Convert float RGB [0,1] to hex color strings for matplotlib
    rgb_profile_hex = [
        '#%02x%02x%02x' % (
            int(np.clip(r * 255, 0, 255)),
            int(np.clip(g * 255, 0, 255)),
            int(np.clip(b * 255, 0, 255))
        ) for r, g, b in rgb_profile
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the colored profile as overlapping short segments (one per bin gap)
    for i in range(len(r_array_um) - 1):
        ax.plot(
            r_array_um[i:i + 2],          # two-point x segment
            thickness_array_um[i:i + 2],  # two-point y segment
            color=rgb_profile_hex[i],
            linewidth=25,                  # thick so segments overlap seamlessly
            solid_capstyle='butt',         # flat line caps to avoid overlap gaps
            zorder=1
        )

    # Overlay a thin black outline of the profile for clarity
    ax.plot(r_array_um, thickness_array_um, 'k-', linewidth=2, zorder=10,
            label='Droplet Profile')
    ax.axhline(0, color='k', linewidth=2, zorder=10)  # substrate baseline

    max_height = droplet_height * 1e6
    max_radius = base_radius * 1e6

    # Reference lines for base radius and apex height
    ax.axvline(max_radius, color='r', linestyle='--', alpha=0.6,
               linewidth=1.5, label=f'Base Radius = {max_radius:.1f} μm')
    ax.axhline(max_height, color='g', linestyle='--', alpha=0.6,
               linewidth=1.5, label=f'Max Height = {max_height:.1f} μm')

    # Mark the two key geometric points
    ax.plot(0, max_height, 'ko', markersize=8, zorder=11)   # apex
    ax.plot(max_radius, 0,  'ko', markersize=8, zorder=11)  # edge

    ax.set_title('Droplet Thickness Profile Colored by Retardation\n' +
                 f'(Average Δn = {avg_delta_n:.3f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Radial Distance r (μm)', fontsize=12)
    ax.set_ylabel('Droplet Thickness D(r) (μm)', fontsize=12)
    ax.set_xlim(0, max_radius * 1.05)
    ax.set_ylim(0, max_height * 1.15)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=10, loc='upper right')

    plt.savefig(os.path.join(debug_path, 'thickness_profile_colored.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Saved thickness profile to: {debug_path}")


def plot_director_field_layers(mesh_coords, n_vectors, N_E, N_O,
                               base_radius, droplet_height, z_min, z_max,
                               debug_path, n_layers=5):
    """
    Generate per-layer director field diagnostic plots at n_layers z-heights.

    Parameters
    ----------
    mesh_coords : ndarray, shape (N, 3)
    n_vectors   : ndarray, shape (N, 3)
    N_E, N_O    : float   Refractive indices
    base_radius : float   Droplet base radius (m), used to mask outside the droplet
    droplet_height : float  Maximum droplet height (m)
    z_min, z_max   : float  z-extent of the mesh (m)
    debug_path     : str    Output directory
    n_layers       : int    Number of z-slices to plot (0 = skip entirely)
    """
    # Skip immediately if no layers requested — avoids building interpolators for nothing
    if n_layers == 0:
        print("  Skipping director layer plots (n_debug_layers = 0).")
        return

    print(f"\nGenerating director field layer plots ({n_layers} layers)...")

    # Z positions: spread from 5% to 95% of the droplet height to avoid
    # the very bottom (near-substrate noise) and top (sparse mesh near apex).
    z_layers = np.linspace(z_min + 0.05 * droplet_height,
                           z_max - 0.05 * droplet_height, n_layers)

    # Build a 2-D grid of evaluation points in the xy-plane
    resolution = 100   # 100×100 grid per layer (fast enough for debug)
    x_min, y_min = mesh_coords[:, 0].min(), mesh_coords[:, 1].min()
    x_max, y_max = mesh_coords[:, 0].max(), mesh_coords[:, 1].max()

    x_plot = np.linspace(x_min, x_max, resolution)
    y_plot = np.linspace(y_min, y_max, resolution)
    X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

    # --- Build interpolators ONCE outside the layer loop ---
    # LinearNDInterpolator builds a 3-D Delaunay triangulation internally;
    # this is expensive and must not be repeated for each layer.
    interp_nx = LinearNDInterpolator(mesh_coords, n_vectors[:, 0])
    interp_ny = LinearNDInterpolator(mesh_coords, n_vectors[:, 1])
    interp_nz = LinearNDInterpolator(mesh_coords, n_vectors[:, 2])

    for layer_idx, z_val in enumerate(z_layers):

        # Assemble 3-D query points: same xy grid, fixed z = z_val
        points_layer = np.column_stack([
            X_plot.ravel(),
            Y_plot.ravel(),
            np.full(resolution * resolution, z_val)  # z is constant for this slice
        ])

        # Evaluate director components at the layer grid (reuse the pre-built interpolators)
        n_x = interp_nx(points_layer).reshape(resolution, resolution)
        n_y = interp_ny(points_layer).reshape(resolution, resolution)
        n_z = interp_nz(points_layer).reshape(resolution, resolution)

        # Derived optical quantities
        phi = np.arctan2(n_y, n_x)                       # azimuthal director angle
        theta_tilt = np.arccos(np.clip(n_z, -1, 1))      # polar tilt from z-axis
        denom   = np.sqrt(N_E ** 2 * np.cos(theta_tilt) ** 2 +
                          N_O ** 2 * np.sin(theta_tilt) ** 2)
        delta_n = (N_E * N_O) / denom - N_O              # effective birefringence Δn_eff

        # Mask pixels outside the droplet base circle (distance from origin > R)
        R_xy = np.sqrt(X_plot ** 2 + Y_plot ** 2)
        mask = R_xy <= base_radius   # True inside the droplet footprint

        n_x_masked    = np.where(mask, n_x,    np.nan)
        n_y_masked    = np.where(mask, n_y,    np.nan)
        n_z_masked    = np.where(mask, n_z,    np.nan)
        phi_masked    = np.where(mask, phi,    np.nan)
        delta_n_masked = np.where(mask, delta_n, np.nan)

        # --- Build the 6-panel figure ---
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        extent = [x_min * 1e6, x_max * 1e6, y_min * 1e6, y_max * 1e6]

        # Row 0: nx, ny, nz component maps (diverging RdBu_r colormap, ±1 range)
        for ax, data, label, title in [
            (axes[0, 0], n_x_masked, 'n_x', 'Director Component n_x'),
            (axes[0, 1], n_y_masked, 'n_y', 'Director Component n_y'),
            (axes[0, 2], n_z_masked, 'n_z', 'Director Component n_z'),
        ]:
            im = ax.imshow(data, cmap='RdBu_r', origin='lower',
                           extent=extent, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label=label)
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            ax.set_title(title)
            ax.set_aspect('equal')

        # Panel [1,0]: in-plane director quiver plot (every 5th point for clarity)
        ax4 = axes[1, 0]
        skip = 5   # subsample the grid so arrows don't overlap
        ax4.quiver(X_plot[::skip, ::skip] * 1e6, Y_plot[::skip, ::skip] * 1e6,
                   n_x_masked[::skip, ::skip], n_y_masked[::skip, ::skip],
                   color='blue', alpha=0.6)
        ax4.set_xlabel('x (μm)')
        ax4.set_ylabel('y (μm)')
        ax4.set_title('In-Plane Director Field (n_x, n_y)')
        ax4.set_aspect('equal')
        ax4.set_xlim(x_min * 1e6, x_max * 1e6)
        ax4.set_ylim(y_min * 1e6, y_max * 1e6)

        # Panel [1,1]: azimuthal angle φ = atan2(ny, nx) — circular 'twilight' colormap
        im5 = axes[1, 1].imshow(phi_masked, cmap='twilight', origin='lower',
                                 extent=extent, vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im5, ax=axes[1, 1], label='φ (rad)')
        axes[1, 1].set_xlabel('x (μm)')
        axes[1, 1].set_ylabel('y (μm)')
        axes[1, 1].set_title('Director Angle φ = atan2(n_y, n_x)')
        axes[1, 1].set_aspect('equal')

        # Panel [1,2]: effective birefringence Δn_eff = n_eff(θ) − n_o
        # Fixed color range [0, 0.18] to match typical LC birefringences
        im6 = axes[1, 2].imshow(delta_n_masked, cmap='viridis', origin='lower',
                                 extent=extent, vmin=0, vmax=0.18)
        plt.colorbar(im6, ax=axes[1, 2], label='Δn_eff')
        axes[1, 2].set_xlabel('x (μm)')
        axes[1, 2].set_ylabel('y (μm)')
        axes[1, 2].set_title('Effective Birefringence Δn_eff')
        axes[1, 2].set_aspect('equal')

        # Super-title with layer index and absolute / normalised z position
        plt.suptitle(
            f'Director Field Analysis — Layer {layer_idx + 1}/{n_layers}\n'
            f'z = {z_val * 1e6:.2f} μm  '
            f'(z/H = {(z_val - z_min) / droplet_height:.2f})',
            fontsize=14, fontweight='bold'
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(debug_path, f'layer_{layer_idx:02d}_z{z_val * 1e6:.1f}um.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()

    print(f"  Saved {n_layers} director field layer plots to: {debug_path}")
