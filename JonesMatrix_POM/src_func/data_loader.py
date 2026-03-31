"""
src_func/data_loader.py

Director field I/O and interpolation.

Functions
---------
load_director_field_from_h5
    Read mesh node coordinates and director/polarization field from an
    HDF5 file written by DOLFINx/FEniCSx.

create_director_interpolator
    Build five scipy LinearNDInterpolator objects for the independent
    Q-tensor components (Qxx, Qxy, Qxz, Qyy, Qyz).

evaluate_director_at_points
    Reconstruct unit director vectors at arbitrary (x,y,z) points by
    evaluating the Q-tensor interpolators and extracting the principal
    eigenvector via vectorised np.linalg.eigh.
"""

import numpy as np
import h5py
from scipy.interpolate import LinearNDInterpolator


def load_director_field_from_h5(h5_path, timestep=-1):
    """
    Load director field from a DOLFINx HDF5 checkpoint file.

    Parameters
    ----------
    h5_path : str
        Absolute path to the HDF5 file (typically 'simulation_P.h5').
    timestep : int
        Index into the sorted list of available timesteps.
        -1 → last,  0 → first,  n → n-th entry.

    Returns
    -------
    mesh_coords : ndarray, shape (N, 3)
        (x, y, z) coordinates of the N mesh nodes in metres.
    n_vectors : ndarray, shape (N, 3)
        Unit director vector at each node.  Computed by normalising the
        raw polarization / director field stored in the HDF5 file.
    """
    print(f"\nLoading from HDF5: {h5_path}")

    with h5py.File(h5_path, 'r') as h5f:

        # --- Locate mesh geometry ---
        # DOLFINx has changed the HDF5 layout across versions; try all known paths.
        possible_paths = [
            'Mesh/Cap_Mesh/geometry',    # older DOLFINx cap-mesh checkpoint
            'Mesh/Loaded_Mesh/geometry', # loaded external mesh
            'Mesh/mesh/geometry',        # generic mesh name
            'Mesh/geometry',             # flat layout
            'geometry',                  # top-level (rare)
        ]
        mesh_coords = None
        for path in possible_paths:
            if path in h5f:
                mesh_coords = h5f[path][:]
                print(f"  Found mesh at: {path}")
                break
        if mesh_coords is None:
            raise ValueError("Mesh coordinates not found in HDF5.  "
                             "Tried: " + str(possible_paths))

        # --- Locate the director/polarization field ---
        # Accept either 'Polarization' (P field, used in SmZA sims) or 'Director'.
        field_name = None
        if 'Function' in h5f:
            for candidate in ('Polarization', 'Director'):
                if candidate in h5f['Function']:
                    field_name = candidate
                    break
        if field_name is None:
            raise ValueError("Neither 'Polarization' nor 'Director' found "
                             "under the 'Function' group in the HDF5 file.")

        # --- Select the desired timestep ---
        # Timestep keys are strings; sort lexicographically for consistent ordering.
        timesteps = sorted([k for k in h5f[f'Function/{field_name}'].keys()])
        print(f"  Found field: {field_name}")
        print(f"  Available timesteps: {len(timesteps)} "
              f"({timesteps[0]} to {timesteps[-1]})")

        selected_timestep = timesteps[timestep]   # supports negative indexing
        print(f"  Loading timestep: {selected_timestep} (index {timestep})")

        P_data = h5f[f'Function/{field_name}/{selected_timestep}'][:]

    # --- Reshape if stored as flat array ---
    # DOLFINx may write a 1-D array of length 3N instead of (N, 3).
    if len(P_data.shape) == 1:
        P_values = P_data.reshape(-1, 3)
    else:
        P_values = P_data

    # --- Guard against mesh/field size mismatch ---
    # Take the smaller count to avoid index-out-of-bounds on truncated files.
    n_points = min(mesh_coords.shape[0], P_values.shape[0])
    mesh_coords = mesh_coords[:n_points]
    P_values    = P_values[:n_points]

    # --- Normalise to unit vectors ---
    # The stored field may be a polarization vector with non-unit magnitude;
    # we only need the director (orientation), so we normalise.
    P_mag = np.linalg.norm(P_values, axis=1, keepdims=True)
    P_mag[P_mag == 0] = 1.0   # avoid division by zero at defect cores
    n_vectors = P_values / P_mag

    return mesh_coords, n_vectors


def create_director_interpolator(mesh_coords, n_vectors):
    """
    Build Q-tensor interpolators over the unstructured mesh.

    Parameters
    ----------
    mesh_coords : ndarray, shape (N, 3)
    n_vectors   : ndarray, shape (N, 3)

    Returns
    -------
    tuple of 5 LinearNDInterpolator objects
        (interp_Qxx, interp_Qxy, interp_Qxz, interp_Qyy, interp_Qyz)
    """
    print("Creating Q-tensor interpolators (sign-invariant)...")
    nx, ny, nz = n_vectors[:, 0], n_vectors[:, 1], n_vectors[:, 2]

    # Build one interpolator per independent Q component.
    # Note: Q_zz = 1 - Q_xx - Q_yy is recovered analytically in evaluate_director_at_points.
    return (
        LinearNDInterpolator(mesh_coords, nx * nx),  # Qxx = nx²
        LinearNDInterpolator(mesh_coords, nx * ny),  # Qxy = nx·ny
        LinearNDInterpolator(mesh_coords, nx * nz),  # Qxz = nx·nz
        LinearNDInterpolator(mesh_coords, ny * ny),  # Qyy = ny²
        LinearNDInterpolator(mesh_coords, ny * nz),  # Qyz = ny·nz
    )


def evaluate_director_at_points(points, interp_Qxx, interp_Qxy, interp_Qxz,
                                interp_Qyy, interp_Qyz):
    """
    Reconstruct the unit director field from Q-tensor at arbitrary points.

    Parameters
    ----------
    points : ndarray, shape (M, 3)
        Query coordinates in metres.
    interp_Qxx, interp_Qxy, interp_Qxz, interp_Qyy, interp_Qyz :
        Five LinearNDInterpolator objects from create_director_interpolator.

    Returns
    -------
    n_values : ndarray, shape (M, 3)
        Reconstructed unit director vectors.  NaN for out-of-hull points.
    """
    # Evaluate each Q component at all query points in one call (vectorised)
    Qxx = interp_Qxx(points)
    Qxy = interp_Qxy(points)
    Qxz = interp_Qxz(points)
    Qyy = interp_Qyy(points)
    Qyz = interp_Qyz(points)

    # Recover Q_zz from the unit-vector trace constraint:  Q_xx + Q_yy + Q_zz = 1
    Qzz = 1.0 - Qxx - Qyy

    # Points outside the mesh hull produce NaN from the interpolator
    nan_mask = np.isnan(Qxx)

    # Replace NaN entries with the isotropic tensor (1/3·I) so that
    # np.linalg.eigh doesn't receive NaN and produce garbage eigenvectors.
    # These points are masked to NaN again after eigendecomposition.
    iso = 1.0 / 3.0
    Qxx_c = np.where(nan_mask, iso, Qxx)
    Qxy_c = np.where(nan_mask, 0.0, Qxy)
    Qxz_c = np.where(nan_mask, 0.0, Qxz)
    Qyy_c = np.where(nan_mask, iso, Qyy)
    Qyz_c = np.where(nan_mask, 0.0, Qyz)
    Qzz_c = np.where(nan_mask, iso, Qzz)

    # Assemble symmetric 3×3 matrices stacked as (M, 3, 3)
    row0 = np.stack([Qxx_c, Qxy_c, Qxz_c], axis=1)  # first row of each matrix
    row1 = np.stack([Qxy_c, Qyy_c, Qyz_c], axis=1)  # second row (symmetric)
    row2 = np.stack([Qxz_c, Qyz_c, Qzz_c], axis=1)  # third row
    Q_stack = np.stack([row0, row1, row2], axis=1)    # shape: (M, 3, 3)

    # Vectorised eigendecomposition.
    # eigh (symmetric) returns eigenvalues in ascending order → index 2 = largest.
    _, eigvecs = np.linalg.eigh(Q_stack)
    n_values = eigvecs[:, :, 2].copy()   # principal eigenvector for each point

    # Re-apply the NaN mask for out-of-hull points
    n_values[nan_mask] = np.nan

    return n_values
