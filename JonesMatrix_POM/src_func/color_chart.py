"""
src_func/color_chart.py

Michel-Levy color chart: converts optical retardation (nm) → sRGB color.

Background
----------
In polarized optical microscopy, the color seen under crossed polarizers is
determined by the optical path difference (retardation) Γ = Δn · d, where
Δn is the birefringence and d is the sample thickness.

The Michel-Levy method computes the perceived color by:
  1. At each visible wavelength λ, the transmitted intensity through a
     birefringent slab between crossed polarizers is  I(λ) = sin²(π Γ / λ).
  2. The spectrum I(λ) is converted to XYZ tristimulus values using the
     CIE 1931 2° standard observer color-matching functions (read from
     ciexyz31_1.csv in the same directory).
  3. XYZ is transformed to linear sRGB via the standard 3×3 matrix.
  4. An auto-exposure normalisation (peak = 1.0) is applied so colors are
     as vivid as possible.
  5. sRGB gamma (γ = 2.2) is applied for display.

Data file: ciexyz31_1.csv
    Columns: wavelength (nm), x̄(λ), ȳ(λ), z̄(λ)
    Range: 360–830 nm, 1 nm steps.
    Source: CIE 1931 2° standard observer tabulated data.

Functions
---------
read_cie_data
    load the CIE color-matching functions from CSV
compute_michel_levy_color
    map retardation array → sRGB image array
"""

import os
import numpy as np


def read_cie_data(filename='ciexyz31_1.csv'):
    """
    !!! The CSV file must be in the same directory as this Python file !!!

    Parameters
    ----------
    filename : str
        Name of the CIE data file (default: 'ciexyz31_1.csv').

    Returns
    -------
    wl : ndarray, shape (N,)
        Wavelength samples in nm.
    XYZ : ndarray, shape (3, N)
        Color-matching functions [x̄, ȳ, z̄] evaluated at each wavelength.
    """
    # Resolve the path relative to this source file, not the CWD,
    # so the CSV is found regardless of where the script is launched from.
    here = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(here, filename)

    data = np.loadtxt(filepath, delimiter=',')
    # data[:, 0] → wavelengths;  data[:, 1:] → x̄, ȳ, z̄  (shape N×3)
    return data[:, 0], data[:, 1:].T   # transpose → shape (3, N)


def compute_michel_levy_color(retardation_nm, wavelengths=None, gamma=2.2):
    """
    Convert an array of optical retardations to sRGB Michel-Levy colors.

    Parameters
    ----------
    retardation_nm : array-like
        Optical path difference in nanometres.  Can be any shape; the
        output will have the same shape with an extra trailing dimension of 3.
    wavelengths : array-like, optional
        Wavelength sample points in nm for the integration.
        Default: 360–830 nm every 10 nm (48 points) — fast and accurate enough.
    gamma : float
        Gamma exponent for sRGB encoding.  Default 2.2 matches most displays.

    Returns
    -------
    RGB : ndarray, shape (*input_shape, 3)
        Gamma-corrected sRGB values in [0, 1].
        E.g. for a (400, 400) retardation map, output shape is (400, 400, 3).
    """
    if wavelengths is None:
        # Default sampling: 360–830 nm every 10 nm
        wavelengths = np.arange(360, 831, 10)

    # --- Load and interpolate CIE color-matching functions to our wavelength grid ---
    cie_wl, cie_XYZ = read_cie_data()
    XYZ_interpol = np.zeros((3, len(wavelengths)))
    for i in range(3):
        # Linear interpolation of x̄/ȳ/z̄ onto our wavelength samples
        XYZ_interpol[i, :] = np.interp(wavelengths, cie_wl, cie_XYZ[i, :])

    # --- Flatten retardation array for vectorised computation ---
    retardation_nm = np.atleast_1d(retardation_nm)
    original_shape = retardation_nm.shape
    retardation_nm = retardation_nm.flatten()   # shape: (M,)

    # --- Compute spectral intensity matrix L[i, j] = sin²(π Γ_j / λ_i) ---
    # L has shape (n_wavelengths, M): intensity at wavelength λ_i for retardation Γ_j.
    L = np.zeros((len(wavelengths), len(retardation_nm)))
    for i, wl in enumerate(wavelengths):
        # Malus's law for a birefringent slab between crossed polarizers
        val = np.sin(np.pi * retardation_nm / wl)
        L[i, :] = val ** 2   # intensity = sin²(phase/2) where phase = 2π Γ/λ

    # --- Convert spectrum to XYZ tristimulus values ---
    # XYZ_interpol: (3, n_wl);  L: (n_wl, M)  →  L_XYZ: (3, M)
    L_XYZ = np.dot(XYZ_interpol, L)

    # --- XYZ → linear sRGB (IEC 61966-2-1 / sRGB standard matrix) ---
    XYZ_to_RGB = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    RGB = np.dot(XYZ_to_RGB, L_XYZ)   # shape: (3, M)

    # --- Auto-exposure: normalise so the brightest color in the image = 1.0 ---
    # This is the standard Michel-Levy chart presentation: colors are relative
    # to the maximum, so the chart looks as vivid as possible.
    max_val = np.max(RGB)
    if max_val > 0:
        RGB = RGB / max_val

    # --- Clip, apply gamma, and reshape back to the original spatial layout ---
    RGB = np.clip(RGB, 0, 1.0)
    RGB = RGB ** (1.0 / gamma)   # gamma encoding (linear → display)
    RGB = RGB.T.reshape(original_shape + (3,))   # shape: (*original_shape, 3)

    return RGB
