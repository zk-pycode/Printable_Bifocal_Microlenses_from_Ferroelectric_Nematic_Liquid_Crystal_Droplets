## JonesMatrix_POM

Polarized Optical Microscopy (POM) image generator using layer-by-layer
Jones matrix calculus on a 3-D liquid-crystal director field from a
FEniCSx/DOLFINx finite-element simulation.

---

## How it works

1. Reads a DOLFINx HDF5 checkpoint (simulation_P.h5) containing mesh
   coordinates and a 3-component director (or polarization) field.
2. Builds sign-invariant Q-tensor interpolators so antiparallel director
   regions interpolate without artefacts.
3. Models the LC droplet as a spherical cap; determines which output pixels
   fall inside the footprint and how tall each pixel column is.
4. For every pixel column, propagates a Jones electric-field vector through
   the stratified LC stack, accumulating one retarder matrix per layer.
5. Integrates over 20 visible wavelengths (380–750 nm, flat spectrum) and
   maps retardation to Michel-Levy sRGB colors using the CIE 1931 observer.
6. Saves one PNG frame per polarizer angle to POM_output/POM_images/.

---

## Project structure
```bash ...
JonesMatrix_POM/
├── main_JPOM.py          	Entry point — instantiates params, calls pipeline
├── params_JPOM.py        	All user-configurable settings (edit this file)
├── README.md             	This file
└── src_func/
    ├── __init__.py
    ├── ciexyz31_1.csv    	CIE 1931 2° color-matching functions (360–830 nm)
    ├── color_chart.py    	Michel-Levy color computation (retardation → sRGB)
    ├── data_loader.py    	HDF5 I/O + Q-tensor interpolation
    ├── debug_plots.py    	Diagnostic visualisation (thickness profile, director layers)
    ├── jones_calculus.py 	Numba-JIT Jones matrix physics kernels
    └── pom_generator.py  	Pipeline orchestrator (run_pom_pipeline)
```

## Output (written in simulation directory)
```bash ...
<simulation_dir>/POM_output/
├── POM_images/
│   ├── frame_000.png     	POM frame at polarizer angle 0°
│   ├── frame_015.png     	POM frame at polarizer angle 15°
│   └── ...
└── debug_plots/
    ├── base_color_map.png          	Michel-Levy color map (angle-independent)
    ├── height_contour_map.png      	Droplet height profile with contours
    ├── thickness_profile_colored.png   Radial profile colored by retardation
    └── layer_NN_zXX.Xum.png        	Director field at z-slice NN  (if n_debug_layers > 0)
```
---

## Quick start

1. Edit the simulation directory and optical parameters : nano params_JPOM.py
2. Run : python main_JPOM.py

### Dependencies and versions

| Package      | Minimum version | Purpose                                          |
|--------------|-----------------|--------------------------------------------------|
| numpy        | 1.24            | Array operations, eigendecomposition             |
| scipy        | 1.10            | LinearNDInterpolator, ConvexHull, interp1d       |
| matplotlib   | 3.7             | Image saving and debug plot generation           |
| h5py         | 3.8             | Reading DOLFINx HDF5 checkpoint files            |
| numba        | 0.57            | JIT compilation of Jones matrix kernels          |
| progiter     | 2.0             | Progress bars with ETA                           |

### Standard library modules used

| Module     | Used in                            | Purpose                                         |
|------------|------------------------------------|-------------------------------------------------|
| os         | all files                          | Path joining, directory creation                |
| time       | main_JPOM.py, pom_generator.py     | Wall-clock timing                               |
| traceback  | main_JPOM.py                       | Full exception traceback on fatal errors        |

### Data file

| File             | Location  |                                                                         |
|------------------|-----------|-------------------------------------------------------------------------|
| ciexyz31_1.csv   | src_func/ | CIE 1931 2° standard observer color-matching functions (360–830 nm, 1 nm steps). Columns: wavelength, x̄, ȳ, z.	|

---

## Physics notes

### Q-tensor interpolation
The LC director n is headless (n ≡ −n).  Direct interpolation of
the (nx, ny, nz) components between antiparallel nodes collapses to zero at
the midpoint, producing a spurious vertical tilt after normalisation.
The fix: interpolate Q_ij = n_i · n_j (sign-invariant) and recover the
director as the principal eigenvector at each query point.

### Jones matrix formulation
Each LC layer is a uniaxial retarder.  For a layer with in-plane director
angle θ and phase retardance δ = 2π Δn d / λ:

    J = R(θ) · diag(e^{-iδ/2}, e^{+iδ/2}) · R(-θ)

The total matrix is the ordered product J_N · ... · J_1.
Light enters as a linear polarizer state and exits through a crossed analyzer.

### Dummy layers
The FEM mesh is coarsest near the substrate.  n_dummy_layers_bottom uniform
layers with director along +x are prepended below the mesh to compensate for
the missing substrate region.

### Sign-consistency walk
After Q-tensor eigendecomposition, adjacent layers may have flipped signs.
A greedy walk flips each layer to match the hemisphere of the layer below,
preserving the physical twist direction in the Jones product.
