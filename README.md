# Cassegrain Slab Simulation

Numerical simulations and far-field analysis of light transport through local self-uniform (LSU) disordered photonic slabs, using a Cassegrain illumination geometry. The project uses [Tidy3D](https://www.flexcompute.com/tidy3d/) for FDTD simulations and provides post-processing tools to decompose the transmitted far-field into ballistic, co-polarized, and cross-polarized components.

## Authors

- Abraham A. Uribe
- Francisco Hernández Alejandre
- Geoffroy Aubry
- Luis S. Froufe-Pérez
- Marian Florescu
- Frank Scheffold

## Installation

1. **Clone the repository** and navigate into the project directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with your Tidy3D API key:
   ```
   API_TIDY3D_KEY=your_api_key_here
   ```
   You can obtain an API key by registering at [Tidy3D](https://www.flexcompute.com/tidy3d/).

## Notebooks

### 1. `20260310 Numerical Simulations.ipynb`

Prepares and launches Tidy3D FDTD simulations for LSU photonic slab structures under partial Cassegrain-cone illumination. The notebook:

- Loads candidate structure files from the `Structures/` directory.
- Defines angular sampling grids, wavelength sweep ranges, and geometric cut parameters.
- Configures a focused Gaussian beam to model one section of the Cassegrain illuminating cone.
- Iterates over structures and normalized cuts to build and submit far-field simulations.

### 2. `20260310 Get Far Field Data.ipynb`

Reduces heavy Tidy3D simulation results down to the projected far-field quantities needed for analysis. The notebook:

- Loads completed simulation results via the Tidy3D API.
- Extracts only the projected far-field components ($E_\theta$, $E_\varphi$) and coordinate grids.
- Saves the lightweight data into compact HDF5 files in the `data/` directory.

### 3. `20260310_Extract_Transmission_Components.ipynb`

Decomposes the far-field transmitted power through the disordered slab into three physically meaningful components:

- **Ballistic transmission** ($T_\text{ballistic}$) — coherent forward-scattered power preserving both amplitude and phase of the reference beam.
- **Co-polarized transmission** ($T_\text{co}$) — total power projected onto the reference polarization state.
- **Cross-polarized transmission** ($T_\text{cross}$) — scattered power orthogonal to the reference polarization.

The angular integration is restricted to the Cassegrain optical aperture cone ($\theta \in [15°, 30°]$).

## License

This project is licensed under the [MIT License](LICENSE).

## Project Structure

```
├── AutomationModule/       # Helper modules for loading structures, HDF5 I/O, and simulation tools
├── Structures/             # HDF5 files defining the photonic slab geometries
├── data/                   # Stored far-field simulation results (HDF5)
├── requirements.txt        # Python dependencies
└── *.ipynb                 # Jupyter notebooks (see above)
```
