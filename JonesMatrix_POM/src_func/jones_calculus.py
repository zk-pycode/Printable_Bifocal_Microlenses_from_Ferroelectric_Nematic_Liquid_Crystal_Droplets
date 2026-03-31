"""
src_func/jones_calculus.py

Low-level optical physics kernels compiled with Numba @njit for speed.

Functions
---------
get_avg_neff_column
    Average effective birefringence Δn_eff along one pixel column.
    Used for the Michel-Levy retardation color map (a quick approximation).

calculate_intensity_jones_calculus
    Full layer-by-layer Jones matrix propagation for one pixel column,
    returning the intensity reaching the crossed analyzer.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Helper: average effective birefringence along a column
# ---------------------------------------------------------------------------

@njit(fastmath=True)   # fastmath=True allows reassociation of FP ops for speed
def get_avg_neff_column(n_z_vals, N_E, N_O):
    """
    Compute the column-average effective birefringence Δn_eff.

    Parameters
    ----------
    n_z_vals : 1-D array
        z-component of the unit director at each layer in the column.
        Clamped internally to [-1, 1] to guard against floating-point drift.
    N_E : float
        Extraordinary refractive index.
    N_O : float
        Ordinary refractive index.

    Returns
    -------
    float
        Mean Δn_eff over the column.  Returns 0.0 for an empty column.
    """
    n = len(n_z_vals)
    if n == 0:
        return 0.0

    total_delta_n = 0.0
    for i in range(n):
        # Clamp nz to valid range for arccos
        nz = n_z_vals[i]
        if nz < -1.0:
            nz = -1.0
        if nz > 1.0:
            nz = 1.0

        # Tilt angle of the director from the z-axis
        theta = np.arccos(nz)

        # Effective extraordinary index at this tilt
        denom = np.sqrt(N_E ** 2 * np.cos(theta) ** 2 + N_O ** 2 * np.sin(theta) ** 2)
        neff = (N_E * N_O) / denom

        # Local birefringence contribution
        total_delta_n += (neff - N_O)

    return total_delta_n / n


# ---------------------------------------------------------------------------
# Main kernel: full Jones matrix propagation for one pixel column
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def calculate_intensity_jones_calculus(n_x_valid, n_y_valid, n_z_valid,
                                       thickness, rotation_rad,
                                       N_E, N_O,
                                       n_dummy_layers_bottom=25):
    """
    Calculate the transmitted intensity through crossed polarizers for one
    pixel column using layer-by-layer Jones matrix multiplication.

    Parameters
    ----------
    n_x_valid, n_y_valid, n_z_valid : 1-D arrays, length n_points
        x, y, z components of the unit director at each layer.
        Must be sign-consistent along the column (see pom_generator.py for
        the flip-to-same-hemisphere step applied before calling this function).
    thickness : float
        Total physical thickness of the droplet column (metres).
    rotation_rad : float
        Polarizer angle in radians.  The analyzer is at rotation_rad + π/2.
    N_E : float
        Extraordinary refractive index.
    N_O : float
        Ordinary refractive index.
    n_dummy_layers_bottom : int
        Number of artificial substrate layers (uniform +x director) prepended
        below the mesh data.

    Returns
    -------
    float
        Spectrum-averaged intensity in [0, 1] (normalised by n_wavelengths).
        Returns 0.0 for an empty column.
    """
    n_points = len(n_x_valid)
    if n_points == 0:
        return 0.0

    # Each layer (real + dummy) gets the same physical thickness
    layer_thickness = thickness / n_points

    # --- Visible spectrum: 20 wavelengths from 380 nm to 750 nm ---
    lambda_min = 380e-9   # metres
    lambda_max = 750e-9   # metres
    n_wavelengths = 20
    dlambda = (lambda_max - lambda_min) / (n_wavelengths - 1)

    intensity_sum = 0.0

    # Outer loop: integrate over wavelengths (spectral averaging)
    for wl_idx in range(n_wavelengths):
        wavelength = lambda_min + wl_idx * dlambda

        # -----------------------------------------------------------------
        # Initialise electric field: linearly polarised at rotation_rad
        # E_in = [cos(θ_pol), sin(θ_pol)]  (unit amplitude, real)
        # -----------------------------------------------------------------
        cos_pol = np.cos(rotation_rad)
        sin_pol = np.sin(rotation_rad)

        Ex_real = cos_pol   # x-component, real part
        Ey_real = sin_pol   # y-component, real part
        Ex_imag = 0.0       # x-component, imaginary part (zero at input)
        Ey_imag = 0.0       # y-component, imaginary part (zero at input)

        # =================================================================
        # STAGE 1: Dummy substrate layers (uniform director along +x)
        # Director = (1, 0, 0)  →  θ_director = 0, θ_tilt = π/2
        # Since θ_director = 0, the rotation matrices R(θ) are identity,
        # so only the retarder phase accumulates.
        # =================================================================
        for dummy_idx in range(n_dummy_layers_bottom):
            # Tilt = π/2 → director fully in-plane (maximum birefringence)
            theta_tilt = np.pi / 2.0
            denom = np.sqrt(N_E ** 2 * np.cos(theta_tilt) ** 2 +
                            N_O ** 2 * np.sin(theta_tilt) ** 2)
            neff = (N_E * N_O) / denom
            delta_n_layer = neff - N_O

            # Phase retardance accumulated by this slab: δ = 2π Δn d / λ
            delta_phase = 2.0 * np.pi * delta_n_layer * layer_thickness / wavelength

            # Half-phase trig values for the retarder Jones matrix
            cos_delta = np.cos(delta_phase / 2.0)
            sin_delta = np.sin(delta_phase / 2.0)

            # Apply retarder (θ_director = 0, so no rotation needed):
            #   Fast axis (x): multiply by exp(-iδ/2) = cos_delta - i·sin_delta
            #   Slow axis (y): multiply by exp(+iδ/2) = cos_delta + i·sin_delta
            Ex_ret_real = Ex_real * cos_delta - Ex_imag * (-sin_delta)
            Ex_ret_imag = Ex_real * (-sin_delta) + Ex_imag * cos_delta
            Ey_ret_real = Ey_real * cos_delta - Ey_imag * sin_delta
            Ey_ret_imag = Ey_real * sin_delta + Ey_imag * cos_delta

            # Update field
            Ex_real = Ex_ret_real
            Ex_imag = Ex_ret_imag
            Ey_real = Ey_ret_real
            Ey_imag = Ey_ret_imag

        # =================================================================
        # STAGE 2: Actual mesh layers from the FEM simulation
        # Each layer has a local director (nx, ny, nz).
        # The Jones matrix is: J = R(θ) · Retarder(δ) · R(-θ)
        # Applied in three sub-steps: rotate in → retard → rotate out.
        # =================================================================
        for k in range(n_points):
            nx = n_x_valid[k]
            ny = n_y_valid[k]
            nz = n_z_valid[k]

            # Clamp nz to [-1, 1] for safe arccos
            if nz < -1.0:
                nz = -1.0
            if nz > 1.0:
                nz = 1.0

            # In-plane director angle (azimuthal), used for R(θ) rotations
            theta_director = np.arctan2(ny, nx)

            # Polar tilt of the director from z-axis, used for n_eff
            theta_tilt = np.arccos(nz)

            # Effective extraordinary index at this tilt
            denom = np.sqrt(N_E ** 2 * np.cos(theta_tilt) ** 2 +
                            N_O ** 2 * np.sin(theta_tilt) ** 2)
            neff = (N_E * N_O) / denom
            delta_n_layer = neff - N_O

            # Phase retardance for this slab
            delta_phase = 2.0 * np.pi * delta_n_layer * layer_thickness / wavelength

            # Skip layers that are effectively isotropic (homeotropic regions,
            # where the director is nearly along z so Δn ≈ 0).
            if abs(delta_phase) < 1e-6:
                continue

            cos_theta = np.cos(theta_director)
            sin_theta = np.sin(theta_director)
            cos_delta = np.cos(delta_phase / 2.0)
            sin_delta = np.sin(delta_phase / 2.0)

            # --- Sub-step 1: Rotate into retarder frame  R(-θ) ---
            # R(-θ) = [[cos θ, sin θ], [-sin θ, cos θ]]
            Ex_rot_real = cos_theta * Ex_real + sin_theta * Ey_real
            Ex_rot_imag = cos_theta * Ex_imag + sin_theta * Ey_imag
            Ey_rot_real = -sin_theta * Ex_real + cos_theta * Ey_real
            Ey_rot_imag = -sin_theta * Ex_imag + cos_theta * Ey_imag

            # --- Sub-step 2: Apply retarder ---
            # Fast axis (x): exp(-iδ/2);  Slow axis (y): exp(+iδ/2)
            Ex_ret_real = Ex_rot_real * cos_delta - Ex_rot_imag * (-sin_delta)
            Ex_ret_imag = Ex_rot_real * (-sin_delta) + Ex_rot_imag * cos_delta
            Ey_ret_real = Ey_rot_real * cos_delta - Ey_rot_imag * sin_delta
            Ey_ret_imag = Ey_rot_real * sin_delta + Ey_rot_imag * cos_delta

            # --- Sub-step 3: Rotate back to lab frame  R(+θ) ---
            # R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
            Ex_real = cos_theta * Ex_ret_real - sin_theta * Ey_ret_real
            Ex_imag = cos_theta * Ex_ret_imag - sin_theta * Ey_ret_imag
            Ey_real = sin_theta * Ex_ret_real + cos_theta * Ey_ret_real
            Ey_imag = sin_theta * Ex_ret_imag + cos_theta * Ey_ret_imag

        # =================================================================
        # STAGE 3: Project onto the crossed analyzer
        # Analyzer axis is perpendicular to the polarizer: θ_A = θ_pol + π/2
        # Transmitted amplitude = E · â_analyzer  (dot product)
        # =================================================================
        analyzer_angle = rotation_rad + np.pi / 2.0
        cos_analyzer = np.cos(analyzer_angle)
        sin_analyzer = np.sin(analyzer_angle)

        # Scalar complex amplitude through the analyzer
        E_transmitted_real = Ex_real * cos_analyzer + Ey_real * sin_analyzer
        E_transmitted_imag = Ex_imag * cos_analyzer + Ey_imag * sin_analyzer

        # Intensity = |E|² = Re² + Im²
        intensity_wl = E_transmitted_real ** 2 + E_transmitted_imag ** 2
        intensity_sum += intensity_wl

    # Average over the visible spectrum (flat white-light weighting)
    intensity = intensity_sum / n_wavelengths
    return intensity
