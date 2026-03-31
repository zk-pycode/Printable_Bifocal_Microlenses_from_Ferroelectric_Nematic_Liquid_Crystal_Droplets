"""
src_func/pom_generator.py


Pipeline stages
---------------
1. Output directories
       POM_output/POM_images/   ← final frame PNGs
       POM_output/debug_plots/  ← diagnostics

2. Data loading  (data_loader.py)
       Read mesh coordinates and director field from HDF5.
       Build Q-tensor interpolators (sign-invariant).

3. Geometry extraction
       Derive base radius from mesh x-extents.
       Compute droplet height from contact angle (params) or z-extents (mesh).
       Compute average birefringence Δn over all mesh nodes.

4. Debug plots  (debug_plots.py)
       Radial thickness profile colored by Michel-Levy.
       Director field layer plots (optional, controlled by n_debug_layers).

5. Pixel-column table
       Determine which output pixels fall inside the droplet footprint
       (ConvexHull of base nodes).
       For each inside pixel, find the surface z via the spherical-cap model
       and collect n_z_samples evenly spaced 3-D query points along the column.
       Evaluate all query points in one batch call to the Q-tensor interpolators.

6. Base color map  (color_chart.py)
       For each valid pixel column: strip NaN layers, enforce sign-consistency
       of the eigenvector sign, compute Δn·thickness retardation, and look up
       the Michel-Levy sRGB color.  Save base_color_map.png and height_contour_map.png.

7. POM image generation  (jones_calculus.py)
       For each polarizer angle:
         - Run calculate_intensity_jones_calculus on every valid column (Numba JIT).
         - Optional per-frame intensity normalisation.
         - Modulate base color image by intensity → save frame_NNN.png.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from progiter import ProgIter
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

from .color_chart import compute_michel_levy_color
from .jones_calculus import get_avg_neff_column, calculate_intensity_jones_calculus
from .data_loader import (load_director_field_from_h5,
                          create_director_interpolator,
                          evaluate_director_at_points)
from .debug_plots import plot_thickness_profile, plot_director_field_layers


def run_pom_pipeline(params):
    """
    Execute the full Jones-matrix POM image generation pipeline.

    Parameters
    ----------
    params : POMParameters
        Configuration object from params_JPOM.py.

    Returns
    -------
    output_base : str
        Absolute path to the POM_output directory that was created.
    """
    start_time = time.time()

    # -----------------------------------------------------------------------
    # STAGE 1 — Set up output directories
    # -----------------------------------------------------------------------
    # POM_output/ is created inside the simulation directory so output stays
    # co-located with the input HDF5 file.
    output_base = os.path.join(params.simulation_dir, "POM_output")
    pom_path    = os.path.join(output_base, "POM_images")   # final frames go here
    debug_path  = os.path.join(output_base, "debug_plots")  # diagnostic plots

    os.makedirs(pom_path,   exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)

    print(f"Simulation directory: {params.simulation_dir}")
    print(f"Timestep: {params.timestep} "
          f"({'last' if params.timestep == -1 else params.timestep})")
    print(f"Output directory: {output_base}")
    print(f"  - POM images: {pom_path}")
    print(f"  - Debug plots: {debug_path}")
    print(f"Dummy layers: Bottom={params.n_dummy_layers_bottom} (+x)")
    print(f"Refractive indices: N_E={params.N_E}, N_O={params.N_O}")
    print(f"Light source: Uniform white light (flat spectrum)")
    print(f"Normalize intensity: {params.normalize_intensity}")

    # -----------------------------------------------------------------------
    # STAGE 2 — Load director field and build interpolators
    # -----------------------------------------------------------------------
    h5_path = os.path.join(params.simulation_dir, "simulation_P.h5")

    # mesh_coords: (N, 3) node positions in metres
    # n_vectors:   (N, 3) unit director at each node
    mesh_coords, n_vectors = load_director_field_from_h5(h5_path,
                                                         timestep=params.timestep)

    # Five Q-tensor interpolators (sign-invariant director interpolation)
    interp_Qxx, interp_Qxy, interp_Qxz, interp_Qyy, interp_Qyz = \
        create_director_interpolator(mesh_coords, n_vectors)

    # -----------------------------------------------------------------------
    # STAGE 3 — Extract droplet geometry
    # -----------------------------------------------------------------------
    x_min, y_min, z_min = mesh_coords.min(axis=0)
    x_max, y_max, z_max = mesh_coords.max(axis=0)

    # Base radius = half the x-extent of the mesh (assumes circular base)
    base_radius = (x_max - x_min) / 2

    if params.contact_angle_deg is not None:
        # Override: compute height from the user-specified contact angle
        # Spherical-cap relation:  h = R * tan(θ / 2)
        theta_rad      = np.deg2rad(params.contact_angle_deg)
        droplet_height = base_radius * np.tan(theta_rad / 2)
    else:
        # Fallback: use the full z-range of the mesh as the droplet height
        droplet_height = z_max - z_min

    print(f"\nDroplet geometry:")
    print(f"  Base radius: {base_radius * 1e6:.2f} μm")
    if params.contact_angle_deg is not None:
        print(f"  Contact angle: {params.contact_angle_deg:.2f}° (from params)")
    else:
        contact_angle_deg = np.rad2deg(2 * np.arctan2(droplet_height, base_radius))
        print(f"  Contact angle: {contact_angle_deg:.2f}° (from mesh)")
    print(f"  Height: {droplet_height * 1e6:.2f} μm")

    # --- Average birefringence over all mesh nodes ---
    # Used for the thickness-profile debug plot and a quick retardation estimate.
    # theta_tilt[i] = angle between director[i] and the z-axis
    theta_tilt = np.arccos(np.clip(n_vectors[:, 2], -1, 1))
    denom = np.sqrt(params.N_E ** 2 * np.cos(theta_tilt) ** 2 +
                    params.N_O ** 2 * np.sin(theta_tilt) ** 2)
    delta_n_values = (params.N_E * params.N_O) / denom - params.N_O
    avg_delta_n = np.mean(delta_n_values)

    print(f"\nOptical parameters:")
    print(f"  Spectrum integration: 380-750 nm (visible range)")
    print(f"  n_e: {params.N_E}, n_o: {params.N_O}")
    print(f"  Average Δn: {avg_delta_n:.4f}")

    # -----------------------------------------------------------------------
    # STAGE 4 — Debug plots (optional)
    # -----------------------------------------------------------------------
    print("\nGENERATING DEBUG PLOTS ---\n")

    # Radial droplet profile colored by Michel-Levy retardation
    plot_thickness_profile(base_radius, droplet_height, z_min, avg_delta_n,
                           params.N_E, params.N_O, debug_path)

    # Director field at n_debug_layers z-slices (0 = skip)
    plot_director_field_layers(mesh_coords, n_vectors, params.N_E, params.N_O,
                               base_radius, droplet_height, z_min, z_max,
                               debug_path, n_layers=params.n_debug_layers)

    # -----------------------------------------------------------------------
    # STAGE 5 — Build the pixel-column interpolation table
    # -----------------------------------------------------------------------
    RES = params.resolution_xy

    # Uniform output grid spanning the mesh x-y extents
    x_out = np.linspace(x_min, x_max, RES)
    y_out = np.linspace(y_min, y_max, RES)
    X_out, Y_out = np.meshgrid(x_out, y_out)

    print("\nPREPARING INTERPOLATION DATA ---\n")

    # --- Determine which output pixels lie inside the mesh footprint ---
    # Use the convex hull of all base-plane mesh nodes as the droplet boundary.
    print("  Detecting mesh boundary...")
    base_mask     = np.abs(mesh_coords[:, 2] - z_min) < 1e-8   # nodes at z = z_min
    base_points_2d = mesh_coords[base_mask, :2]                  # (K, 2) xy coords

    hull      = ConvexHull(base_points_2d)
    hull_path = Path(base_points_2d[hull.vertices])   # matplotlib Path for contains_points

    grid_points_2d = np.column_stack([X_out.ravel(), Y_out.ravel()])
    inside_mesh    = hull_path.contains_points(grid_points_2d)   # bool, shape (RES², )

    # --- Build a smooth radial boundary distance function ---
    # For each pixel inside the hull, we compute r_normalised = r / R(angle),
    # where R(angle) is the actual boundary radius in that direction.
    # This handles non-circular footprints correctly.
    centroid_x = np.mean(base_points_2d[:, 0])
    centroid_y = np.mean(base_points_2d[:, 1])
    print(f"  Base centroid: ({centroid_x * 1e6:.2f}, {centroid_y * 1e6:.2f}) μm")

    boundary_points = base_points_2d[hull.vertices]
    boundary_dx     = boundary_points[:, 0] - centroid_x
    boundary_dy     = boundary_points[:, 1] - centroid_y
    boundary_angles = np.arctan2(boundary_dy, boundary_dx)
    boundary_dists  = np.sqrt(boundary_dx ** 2 + boundary_dy ** 2)

    # Sort by angle so interp1d works correctly
    sort_idx        = np.argsort(boundary_angles)
    boundary_angles = boundary_angles[sort_idx]
    boundary_dists  = boundary_dists[sort_idx]

    # Tile the sorted angles/distances to make the function periodic over [-π, π]
    boundary_angles_periodic = np.concatenate([boundary_angles - 2 * np.pi,
                                               boundary_angles,
                                               boundary_angles + 2 * np.pi])
    boundary_dists_periodic  = np.concatenate([boundary_dists, boundary_dists, boundary_dists])

    # Cubic spline interpolant: angle → boundary radius
    boundary_dist_func = interp1d(boundary_angles_periodic, boundary_dists_periodic,
                                  kind='cubic', assume_sorted=True)

    # --- Collect 3-D query points for all valid pixel columns ---
    # For each inside pixel (i, j):
    #   1. Compute normalised radial position r_norm = r / R(angle)  in [0, 1]
    #   2. Compute surface z from spherical-cap: z_surf = z_min + h * sqrt(1 - r_norm²)
    #   3. Sample n_z_samples evenly from z_min to z_surf
    pixel_indices       = []   # list of (i, j, surface_z) for inside pixels
    interpolation_points = []  # flat list of (x, y, z) query points

    for i in range(RES):
        for j in range(RES):
            idx = i * RES + j
            if not inside_mesh[idx]:
                continue   # pixel is outside the droplet footprint

            x_pos, y_pos = X_out[i, j], Y_out[i, j]
            dx = x_pos - centroid_x
            dy = y_pos - centroid_y

            # Angle from centroid to this pixel
            angle = np.arctan2(dy, dx)

            # Actual boundary radius in this direction
            boundary_dist = boundary_dist_func(angle)
            current_dist  = np.sqrt(dx ** 2 + dy ** 2)

            # Normalised radial position: 0 at centre, 1 at boundary
            r_normalized = (current_dist / boundary_dist) if boundary_dist > 1e-12 else 0.0
            r_normalized = min(r_normalized, 1.0)   # clamp to avoid sqrt of negative

            # Spherical-cap surface height at this pixel
            surface_z = z_min + droplet_height * np.sqrt(1 - r_normalized ** 2)

            # n_z_samples evenly spaced z values from substrate to droplet surface
            z_samples = np.linspace(z_min, surface_z, params.n_z_samples)
            for k in range(params.n_z_samples):
                interpolation_points.append([x_pos, y_pos, z_samples[k]])

            pixel_indices.append((i, j, surface_z))

    # Evaluate the Q-tensor interpolators at all query points in one vectorised call
    print(f"Interpolating {len(interpolation_points)} points...")
    all_n_values = evaluate_director_at_points(np.array(interpolation_points),
                                               interp_Qxx, interp_Qxy, interp_Qxz,
                                               interp_Qyy, interp_Qyz)

    # -----------------------------------------------------------------------
    # STAGE 6 — Build base Michel-Levy color map
    # -----------------------------------------------------------------------
    print("\nGENERATING BASE COLOR MAP ---\n")

    retardation_map_nm = np.zeros((RES, RES))  # optical retardation per pixel [nm]
    mask_map           = np.zeros((RES, RES), dtype=bool)  # True = valid pixel
    valid_columns      = {}  # {(i, j): n_valid array} — reused in stage 7

    current_idx = 0
    for (i, j, surface_z) in ProgIter(pixel_indices, desc="Processing columns"):

        # Extract this column's director samples from the flat interpolated array
        n_vals = all_n_values[current_idx: current_idx + params.n_z_samples]
        current_idx += params.n_z_samples

        # Discard columns with fewer than 2 valid (non-NaN) samples
        valid_mask = ~np.isnan(n_vals[:, 0])
        if np.sum(valid_mask) < 2:
            continue

        mask_map[i, j] = True
        n_valid = n_vals[valid_mask].copy()   # shape (M, 3), M ≤ n_z_samples

        # --- Sign-consistency fix ---
        # The principal eigenvector of Q has an arbitrary ±1 sign per layer.
        # To represent the twist correctly in the Jones matrix product, we flip
        # each layer's director so it points into the same hemisphere as the
        # layer immediately below it (greedy sign-consistency walk).
        for k in range(1, len(n_valid)):
            if np.dot(n_valid[k], n_valid[k - 1]) < 0:
                n_valid[k] = -n_valid[k]

        # Cache the sign-corrected column for the POM image loop in stage 7
        valid_columns[(i, j)] = n_valid

        # Quick retardation estimate using column-averaged Δn (for color map only)
        avg_dn    = get_avg_neff_column(n_valid[:, 2], params.N_E, params.N_O)
        thickness = surface_z - z_min   # physical column height [m]
        retardation_map_nm[i, j] = avg_dn * thickness * 1e9   # convert to nm

    # Convert retardation map to sRGB Michel-Levy colors
    base_color_image = compute_michel_levy_color(retardation_map_nm, gamma=2.2)
    base_color_image[~mask_map] = 0.0   # zero out pixels outside the droplet

    # --- Save base color map (retardation-based, no polarizer dependence) ---
    extent = [x_min * 1e6, x_max * 1e6, y_min * 1e6, y_max * 1e6]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.flipud(base_color_image), origin='lower', extent=extent)
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('y (μm)', fontsize=12)
    ax.set_title('Base Color Map (Retardation)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(debug_path, 'base_color_map.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved base color map to: {debug_path}")

    # --- Save height contour map ---
    print("\nGenerating height contour map...")
    height_map = np.zeros((RES, RES))
    for (i, j, surface_z) in pixel_indices:
        height_map[i, j] = (surface_z - z_min) * 1e6   # store in μm

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.flipud(height_map), cmap='viridis', origin='lower', extent=extent)
    X_plot = np.linspace(x_min * 1e6, x_max * 1e6, RES)
    Y_plot = np.linspace(y_min * 1e6, y_max * 1e6, RES)
    contours = ax.contour(X_plot, Y_plot, np.flipud(height_map),
                          levels=10, colors='white', linewidths=0.5, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f μm')
    plt.colorbar(im, ax=ax, label='Height (μm)')
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('y (μm)', fontsize=12)
    ax.set_title('Droplet Height Profile', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    plt.savefig(os.path.join(debug_path, 'height_contour_map.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved height contour map to: {debug_path}")

    # -----------------------------------------------------------------------
    # STAGE 7 — Generate POM images for each polarizer angle
    # -----------------------------------------------------------------------
    print("\nGENERATING POM IMAGES ---\n")
    print(f"Resolution: {RES} x {RES}")
    print(f"Angles: {len(list(params.angles))} frames")
    print(f"Normalize intensity: {params.normalize_intensity}")

    for angle in ProgIter(list(params.angles), desc="Generating POM images"):
        rotation_rad  = np.deg2rad(angle)
        intensity_map = np.zeros((RES, RES))

        # --- Compute Jones-matrix intensity for every valid pixel column ---
        for (i, j), n_valid in valid_columns.items():
            thickness = height_map[i, j] * 1e-6   # convert μm back to metres

            # Full layer-by-layer Jones calculation (Numba JIT — fast)
            scalar_intensity = calculate_intensity_jones_calculus(
                n_valid[:, 0], n_valid[:, 1], n_valid[:, 2],
                thickness, rotation_rad,
                params.N_E, params.N_O,
                params.n_dummy_layers_bottom
            )
            intensity_map[i, j] = scalar_intensity

        # --- Optional per-frame intensity normalisation ---
        # Rescale so the brightest pixel = 1.0.  This keeps dark textures visible
        # when the overall retardation is low, at the cost of losing absolute
        # intensity information between frames.
        if params.normalize_intensity:
            max_intensity = np.max(intensity_map)
            if max_intensity > 0:
                intensity_map = intensity_map / max_intensity

        # --- Modulate base Michel-Levy colors by the Jones intensity ---
        # final_image[i,j] = base_color[i,j] * intensity[i,j]
        # This gives the correct crossed-polarizer POM appearance: the hue comes
        # from the retardation (base color map) and the brightness from the
        # polarizer-angle-dependent Jones intensity.
        final_image = np.zeros((RES, RES, 3))
        for (i, j) in valid_columns.keys():
            final_image[i, j, :] = base_color_image[i, j, :] * intensity_map[i, j]

        # --- Save the frame ---
        final_image = np.clip(final_image, 0, 1)
        fig_pom, ax_pom = plt.subplots(figsize=(10, 10))
        ax_pom.imshow(final_image, origin='lower', extent=extent)
        ax_pom.set_aspect('equal')
        ax_pom.axis('off')   # no axes for publication-style output
        plt.savefig(os.path.join(pom_path, f"frame_{angle:03d}.png"),
                    dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close(fig_pom)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\nCOMPLETE ---\n")
    print(f"Total time: {elapsed:.1f}s")
    print(f"POM images saved to: {pom_path}")
    print(f"Debug plots saved to: {debug_path}")

    return output_base
