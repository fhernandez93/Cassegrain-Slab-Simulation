import numpy as np
import tidy3d.web as web
import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tidy3d.plugins.dispersion import FastDispersionFitter, AdvancedFastFitterParam
import trimesh as tri


class loadAndRunStructure:
    """
    This class takes a disordered network permitivity distribution file and calculate the transmission coefficients using Tidy3d
    File types can be .stl for structures or .h5 for permittivity distributions. 
    If .stl is in place one must specify if we're working with constant index or specify a link with the refractive index dist. 
    Options for monitors:
    flux_monitor records the transmitted flux through the slab
    flux_time_monitor Records time-dependent transmitted flux through the slab
    field_time_monitors record the E-fields throughout simulation volume at t=run_time/2 and t=run_time
    field_time_monitor_output Records E-fields at the output surface at Nt equally spaced times from 0 to run_time
    Source can be either a planewave, a laser-like beam, or a continuous wave pulse
    """

    def __init__(self, key:str="", file_path:str = "", direction:str="z", 
                 lambda_range: list= [], box_size:float = 0, runtime: int = 0, 
                 width:float=0.4, freqs:int=400,permittivity:float=1, use_permittivity:bool=False,
                 min_steps_per_lambda:int = 20, permittivity_dist:str="", scaling:float=1.0,shuoff_condtion:float=1e-7,
                 sim_mode:str = "transmission", subpixel:bool=True, verbose:bool=False, monitors:list=[], cut_condition:float=1, cut_cell:float=False,cell_size_manual:float=None,
                 source:str="planewave", multiplicate_size:bool=False, tight_percentage:float=None,source_size:float=0, multiplication_factor:int = 1,pol_angle:float = 0,
                 ref_only:bool=False, absorbers:int=40, sim_name:str="",runtime_ps:float=0.0,flux_monitor_position:float=None, h5_bg:float=None,  sim_bg:float=1.0, far_field_settings:any=None,gaussian_params:any=None,
                 boundaries:str="periodic", cut_slab_array_size:dict=None
                 ):
        # =====================================================================
        # Input validation
        # =====================================================================
        if not key:
            raise Exception("No API key was provided")
        else:
            web.configure(key)

        if not file_path and not ref_only:
            raise Exception("No structure was provided")
        
        self.file_format = Path(file_path).suffix
        self.permittivity_dist = permittivity_dist
        
        if self.file_format not in [".h5", ".stl"] and not ref_only:
            raise Exception("No .h5 or .stl structure was provided")
        
        # =====================================================================
        # Load structure file and prepare permittivity data
        # =====================================================================
        self.sim_bg = sim_bg
        self.h5_bg = h5_bg
        self.file = file_path
        self.structure_name = Path(file_path).stem

        if self.file_format == ".h5":
            with h5py.File(self.file, 'r') as f:
                self.permittivity_raw = np.array(f['epsilon'])

                # Trim or extend the permittivity array along the propagation direction
                if cut_condition < 1:
                    # Truncate: keep only a fraction of the slab
                    axis_map = {"x": 0, "y": 1, "z": 2}
                    ax = axis_map[direction]
                    end_idx = int(np.shape(self.permittivity_raw)[ax] * cut_condition - 1)
                    slicing = [slice(None)] * 3
                    slicing[ax] = slice(None, end_idx)
                    self.permittivity_raw = self.permittivity_raw[tuple(slicing)]
                elif cut_condition > 1:
                    # Extend: append a partial copy of the slab along the propagation axis
                    axis_map = {"x": 0, "y": 1, "z": 2}
                    ax = axis_map[direction]
                    extra_len = int(np.shape(self.permittivity_raw)[ax] * (cut_condition - 1))
                    slicing = [slice(None)] * 3
                    slicing[ax] = slice(None, extra_len)
                    extra_slice = self.permittivity_raw[tuple(slicing)]
                    self.permittivity_raw = np.concatenate((self.permittivity_raw, extra_slice), axis=ax)

            # Optional: crop the lateral (x, y) dimensions of the permittivity array
            if cut_slab_array_size:
                self.permittivity_raw = self.permittivity_raw[:int(np.shape(self.permittivity_raw)[0] * cut_slab_array_size["x"] - 1), :, :]
                self.permittivity_raw = self.permittivity_raw[:, :int(np.shape(self.permittivity_raw)[1] * cut_slab_array_size["y"] - 1), :]

            # Replace near-vacuum voxels with a custom background permittivity
            if h5_bg:
                self.permittivity_raw[self.permittivity_raw < 1.0001] = h5_bg

        # Override the structure permittivity to a user-specified value
        if use_permittivity:
            self.permittivity_raw[self.permittivity_raw > 1] += (permittivity - np.max(self.permittivity_raw))
            self.permittivity_raw[self.permittivity_raw < 1] = 1


        # =====================================================================
        # Store simulation parameters
        # =====================================================================
        self.sim_name = sim_name
        self.far_field_settings = far_field_settings
        self.gaussian_params = gaussian_params
        self.flux_monitor_position = flux_monitor_position
        self.absorbers = absorbers
        self.boundaries = boundaries
        self.pol_angle = pol_angle
        self.ref_only = ref_only
        self.source_size = source_size
        self.multiplicate_size = multiplicate_size
        self.multiplication_factor = multiplication_factor
        self.tight_percentage = tight_percentage
        self.source = source
        self.monitors = monitors
        self.min_steps_per_lambda = min_steps_per_lambda

        # For HDF5 files, infer max permittivity from data unless explicitly given
        if self.file_format == ".h5":
            self.permittivity_value = np.max(self.permittivity_raw) if permittivity == 1 else permittivity
        else:
            self.permittivity_value = permittivity
        
        self.sim_mode = sim_mode
        self.subpixel = subpixel
        self.verbose = verbose
        self.scaling = scaling
        self.direction = direction

        # =====================================================================
        # Frequency / wavelength setup
        # =====================================================================
        self.dPML = 1.0
        self.lambda_range = np.array(lambda_range)
        self.freq_range = td.C_0 / self.lambda_range
        self.freq0 = np.sum(self.freq_range) / 2              # Central frequency of source/monitors
        self.lambda0 = td.C_0 / self.freq0
        self.freqw = width * (self.freq_range[1] - self.freq_range[0])  # Gaussian source bandwidth

        # =====================================================================
        # Runtime & temporal parameters
        # =====================================================================
        self.shutoff = shuoff_condtion
        self.runtime = runtime
        self.t_stop = runtime_ps if runtime_ps > 0 else self.runtime / self.freqw
        self.Nfreq = freqs
        self.monitor_freqs = np.linspace(self.freq_range[0], self.freq_range[1], self.Nfreq)
        self.monitor_lambdas = td.constants.C_0 / self.monitor_freqs

        # =====================================================================
        # Spatial grid & domain sizing (all in um)
        # =====================================================================
        # Minimum grid cell size: ensures min_steps_per_lambda points per wavelength inside the medium
        self.dl = (self.lambda_range[1] / self.min_steps_per_lambda) / np.sqrt(self.permittivity_value)

        # PML spacing: gap between slab surface and absorbing boundaries
        self.spacing = self.dPML * self.lambda_range[0]
        self.t_slab = box_size * scaling

        # Slab thickness in each direction (accounts for cut_condition and tiling)
        self.t_slab_x = (self.t_slab * cut_condition if direction == "x" else self.t_slab) * (self.multiplication_factor if self.multiplicate_size else 1)
        self.t_slab_y = (self.t_slab * cut_condition if direction == "y" else self.t_slab) * (self.multiplication_factor if self.multiplicate_size else 1)
        self.t_slab_z = self.t_slab * cut_condition if direction == "z" else self.t_slab

        # Total simulation domain size (slab + spacing for PML on the propagation axis)
        self.sim_size = self.Lx, self.Ly, self.Lz = (
            (self.t_slab_x) / (cut_condition if not cut_cell else 1) + self.spacing * 2 if direction == "x" else self.t_slab_x,
            (self.t_slab_y) / (cut_condition if not cut_cell else 1) + self.spacing * 2 if direction == "y" else self.t_slab_y,
            cell_size_manual if cell_size_manual else (self.t_slab_z) / (cut_condition if not cut_cell else 1) + self.spacing * 2 if direction == "z" else self.t_slab_z,
        )

        # =====================================================================
        # Build and assemble the full Tidy3D simulation
        # =====================================================================
        self.sim_object = self.createSimObjects()
        self.sim = self.simulation_definition()
       


    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
                               
            f'Lx: {self.Lx:.3g} Ly: {self.Ly:.3g} Lz: {self.Lz:.3g}\n'+
            f'lambda_range: {self.lambda_range[1]:.3g} - {self.lambda_range[0]:.3g} um \n'+
            f"lambdaw (pulse) {td.C_0/self.freqw} \n"+
            f"lambda0 {self.lambda0} \n"+
            f"Total runtime <= {self.t_stop*1e12} ps \n"+
            f"dl (Cube Size) = {self.dl*1000} nm \n"
            f"Time Steps = {self.sim.num_time_steps}\n"
            f"Grid Points = {self.sim.num_cells*1e-6} million\n"
            f"eps = {self.permittivity_value}"
        )

        return calculated_data_str


    def checkConnection():
        return web.test()  
    
    def createSimObjects(self):
        """Build all simulation components: source, monitors, structure, and boundaries."""

        # =================================================================
        # Source definition
        # =================================================================
        self.gaussian_pulse = td.GaussianPulse(
            freq0=self.freq0,
            fwidth=self.freqw,
        )

        # Plane wave source (full-domain or tight/truncated)
        if self.source in ["planewave", "tight"]:
            self.source_def = td.PlaneWave(
                source_time = self.gaussian_pulse,
                size= (0 if self.direction == "x" else td.inf, 
                      0 if self.direction == "y" else td.inf, 
                      0 if self.direction == "z" else td.inf
                      ) if self.source == "planewave"
                      else 
                      (
                        0  if self.direction == "x" else (self.t_slab_x*self.tight_percentage if self.tight_percentage else self.source_size),
                        0  if self.direction == "y" else (self.t_slab_y*self.tight_percentage if self.tight_percentage else self.source_size),
                        0  if self.direction == "z" else (self.t_slab_z*self.tight_percentage if self.tight_percentage else self.source_size)

                      )
                      ,
                center=((-self.Lx*0.5+self.spacing*0.1) if self.direction == "x" else 0, 
                        (-self.Ly*0.5+self.spacing*0.1) if self.direction == "y" else 0, 
                        (-self.Lz*0.5+1) if self.direction == "z" else 0) if self.source == "planewave" 


                        else 

                        (
                            -self.t_slab_x/2-self.spacing/2 if self.direction == "x" else 0,
                            -self.t_slab_y/2-self.spacing/2 if self.direction == "y" else 0,
                            -self.t_slab_z/2-self.spacing/2 if self.direction == "z" else 0


                         )

                        ,
                direction='+',
                pol_angle=self.pol_angle,

                name='planewave',
                )
            
        # Focused Gaussian beam source
        if self.source == "gaussian":
                self.source_def = td.GaussianBeam(
                    source_time = td.GaussianPulse(
                        freq0=self.freq0,
                        fwidth=self.freqw
                    ),
                size= (0 if self.direction == "x" else self.gaussian_params.get("size", td.inf), 
                      0 if self.direction == "y" else self.gaussian_params.get("size", td.inf), 
                      0 if self.direction == "z" else self.gaussian_params.get("size", td.inf)
                      ) 
                      ,
                center=((-self.Lx*0.5+self.spacing*0.1) if self.direction == "x" else self.gaussian_params.get("position_x", 0), 
                        (-self.Ly*0.5+self.spacing*0.1) if self.direction == "y" else  self.gaussian_params.get("position_y", 0), 
                        (-self.Lz*0.5+1) if self.direction == "z" else  self.gaussian_params.get("position_z", 0)),

                direction='+',
                pol_angle=self.pol_angle,
                angle_phi=  self.gaussian_params.get("phi", 0),
                angle_theta= self.gaussian_params.get("theta", 0),
                waist_distance=self.gaussian_params["waist_distance"],
                waist_radius=self.gaussian_params["waist_radius"],
                name='gaussian',
                )

        # =================================================================
        # Monitor definitions
        # =================================================================
        self.monitors_names = []

        # Far-field angular projection monitor
        if 'far_field' in self.monitors:
            if self.far_field_settings: 
                self.far_field_monitor =td.FieldProjectionAngleMonitor(
                    center = (
                            0 if self.flux_monitor_position else ((self.Lx - self.spacing)*0.5 if self.direction == "x" else 0), 
                            0 if self.flux_monitor_position else ((self.Ly - self.spacing)*0.5 if self.direction == "y" else 0), 
                            self.flux_monitor_position if self.flux_monitor_position else ((self.Lz - self.spacing)*0.5 if self.direction == "z" else 0)
                            ),  
                    size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf, 
                    ),
                    freqs=self.monitor_freqs,
                    name="far_field",
                    window_size= (self.far_field_settings["window_size"],self.far_field_settings["window_size"]),
                    normal_dir="+",
                    phi=list(self.far_field_settings["phi"]),
                    theta=list(self.far_field_settings["theta"]),
                    proj_distance=self.far_field_settings["r"],
                    far_field_approx=True, 
                )
                self.monitors_names += [self.far_field_monitor]
            else:
                raise Exception("No far field settings were provided")


        # Transmission and reflection flux monitors (frequency-domain)
        if "flux" in self.monitors:
            # Forward flux monitor (downstream of slab)
            self.monitor_1 = td.FluxMonitor(
                center = (
                            0 if self.flux_monitor_position else ((self.Lx - self.spacing)*0.5 if self.direction == "x" else 0), 
                            0 if self.flux_monitor_position else ((self.Ly - self.spacing)*0.5 if self.direction == "y" else 0), 
                            self.flux_monitor_position if self.flux_monitor_position else ((self.Lz - self.spacing)*0.5 if self.direction == "z" else 0)
                            ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf, 
                    ),
                freqs = self.monitor_freqs,
                name='flux1' #To the right 
            )
            # Backward flux monitor (upstream of slab)
            self.monitor_2 = td.FluxMonitor(
                center = (
                        (-self.Lx+self.spacing)*0.5 if self.direction =="x" else 0, 
                        (-self.Ly+self.spacing)*0.5 if self.direction =="y" else 0, 
                        -self.flux_monitor_position if self.flux_monitor_position else ((-self.Lz+self.spacing)*0.5 if self.direction =="z" else 0)
                        ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                freqs = self.monitor_freqs,
                name='flux2'
            )

            self.monitors_names += [self.monitor_1,self.monitor_2]

        # Reflection flux monitor (placed just behind the source)
        if "flux_reflection" in self.monitors:
            self.monitor_reflection = td.FluxMonitor(
               center=((-self.Lx*0.5+self.spacing*0.1) if self.direction == "x" else 0, 
                        (-self.Ly*0.5+self.spacing*0.1) if self.direction == "y" else 0, 
                        (-self.Lz*0.5+0.5) if self.direction == "z" else 0),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                freqs = self.monitor_freqs,
                name='flux_reflection'
            )

            self.monitors_names += [self.monitor_reflection]


        # Records time-dependent transmitted flux through the slab
        if "time_monitor" in self.monitors:
            self.time_monitorT = td.FluxTimeMonitor(
                        center = (
                            0 if self.flux_monitor_position else ((self.Lx - self.spacing)*0.5 if self.direction == "x" else 0), 
                            0 if self.flux_monitor_position else ((self.Ly - self.spacing)*0.5 if self.direction == "y" else 0), 
                            self.flux_monitor_position if self.flux_monitor_position else ((self.Lz - self.spacing)*0.5 if self.direction == "z" else 0)
                            ),
                        size = (
                            0 if self.direction == "x" else td.inf, 
                            0 if self.direction == "y" else td.inf, 
                            0 if self.direction == "z" else td.inf, 
                            ),
                            interval = 50,
                            name="time_monitorT",

                )
            self.monitors_names += [self.time_monitorT]

        # Records E-fields throughout simulation volume at t=run_time/2
        if "field_time_domain" in self.monitors:
            time_monitorH = td.FieldTimeMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    start=self.t_stop / 2.0,
                    stop=self.t_stop / 2.0,
                    #interval_space=(10,10,10),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorH",
                    
                )

            # Records E-fields throughout simulation volume at t=run_time
            time_monitorFinal = td.FieldTimeMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    start=self.t_stop,
                    stop=self.t_stop,
                    #interval_space=(10,10,10),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFinal",
                )
            self.monitors_names+=[time_monitorH,time_monitorFinal]




        # WARNING: frequency-domain field monitor — records at every freq; can be very large
        if "field_monitor" in self.monitors:

            field_monitor = td.FieldMonitor(
                    center=[0.0, 0.0, 0.0],
                    size=[
                            self.t_slab_x+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                    fields=["Ex","Ey","Ez"],
                    name="field_monitor",
                    freqs =self.monitor_freqs,
                    interval_space=(5,5,5)
                )
            self.monitors_names+=[field_monitor]


        # Permittivity distribution monitor (records eps at a single frequency)
        if "permittivity_monitor" in self.monitors:
            eps_monitor = td.PermittivityMonitor(
                center=[0.0, 0.0, 0.0],
                size=[
                            self.t_slab_x+self.spacing if self.direction == "x" else td.inf,
                            self.t_slab_y+self.spacing if self.direction == "y" else td.inf,
                            self.t_slab_z+self.spacing if self.direction == "z" else td.inf
                        ],
                freqs=[self.freq0],
                name="eps_monitor",
            )
            self.monitors_names+=[eps_monitor]


        # =====================================================================
        # Define permittivity distribution / geometry for the slab structure
        # =====================================================================

        # --- Build structure from HDF5 permittivity data ---
        if self.file_format == ".h5":
            x_size, y_size, z_size = self.t_slab_x / self.multiplication_factor, self.t_slab_y / self.multiplication_factor, self.t_slab_z

            if not self.multiplicate_size:
                # Single slab: map the raw permittivity array onto the spatial domain
                Nx, Ny, Nz = np.shape(self.permittivity_raw)
                X = np.linspace(-x_size / 2, x_size / 2, Nx)
                Y = np.linspace(-y_size / 2, y_size / 2, Ny)
                Z = np.linspace(-z_size / 2, z_size / 2, Nz)
                coords = dict(x=X, y=Y, z=Z)

                permittivity_data = SpatialDataArray(self.permittivity_raw, coords=coords)
                dielectric = td.CustomMedium(permittivity=permittivity_data)

                # Size extends to infinity in periodic directions, or matches slab size for absorbers
                slab = td.Structure(
                    geometry=td.Box(
                        center=(0, 0, 0),
                        size=(
                            x_size if self.direction == "x" else (td.inf if self.boundaries == "periodic" else self.t_slab_x),
                            y_size if self.direction == "y" else (td.inf if self.boundaries == "periodic" else self.t_slab_y),
                            z_size if self.direction == "z" else (td.inf if self.boundaries == "periodic" else self.t_slab_z),
                        ),
                    ),
                    medium=dielectric,
                    name='slab',
                )

            else:
                # Multiplicated slab: tile the unit cell in a grid of (multiplication_factor x multiplication_factor)
                slabs = []
                slabs_ref = []
                self.coordinates_slabs = []

                # Compute grid positions for each tile
                for i in range(self.multiplication_factor):
                    for j in range(self.multiplication_factor):
                        center_x = (i - (self.multiplication_factor / 2) + 0.5) * x_size
                        center_y = (j - (self.multiplication_factor / 2) + 0.5) * y_size
                        center_z = 0
                        self.coordinates_slabs.append({
                            "X": (center_x - x_size / 2, center_x + x_size / 2),
                            "Y": (center_y - y_size / 2, center_y + y_size / 2),
                            "Z": (-z_size / 2, z_size / 2),
                            "center": (center_x, center_y, center_z)
                        })

                # Create a structure + reference structure for each tile
                for i, item in enumerate(self.coordinates_slabs):
                    Nx, Ny, Nz = np.shape(self.permittivity_raw)
                    X = np.linspace(item["X"][0], item["X"][1], Nx)
                    Y = np.linspace(item["Y"][0], item["Y"][1], Ny)
                    Z = np.linspace(-z_size / 2, z_size / 2, Nz)
                    coords = dict(x=X, y=Y, z=Z)

                    permittivity_data = SpatialDataArray(self.permittivity_raw, coords=coords)
                    dielectric = td.CustomMedium(permittivity=permittivity_data)

                    slab_i = td.Structure(
                        geometry=td.Box(
                            center=item["center"],
                            size=(x_size, y_size, z_size),
                        ),
                        medium=dielectric,
                        name=f'slab{i}',
                    )

                    # Reference structure uses uniform background permittivity (for normalization)
                    slab_ref_i = td.Structure(
                        geometry=td.Box(
                            center=item["center"],
                            size=(x_size, y_size, z_size),
                        ),
                        medium=td.Medium(permittivity=self.h5_bg if self.h5_bg else 1),
                        name=f'slab_ref_{i}',
                    )

                    slabs.append(slab_i)
                    slabs_ref.append(slab_ref_i)

        # --- Build structure from STL triangle mesh ---
        elif self.file_format == ".stl":
            triangles = tri.load_mesh(self.file)
            triangles.remove_degenerate_faces()
            tri.repair.broken_faces(triangles)
            triangles.apply_scale(self.scaling)
            tri.repair.broken_faces(triangles)
            box = td.TriangleMesh.from_trimesh(triangles)

            if self.permittivity_dist != "":
                # Fit a dispersive material model from an online refractive index database
                fitter = FastDispersionFitter.from_url(self.permittivity_dist)
                fitter = fitter.copy(update={"wvl_range": (self.lambda_range[1], self.lambda_range[0])})
                advanced_param = AdvancedFastFitterParam(weights=(1, 1))
                medium, rms_error = fitter.fit(max_num_poles=10, advanced_param=advanced_param, tolerance_rms=2e-2)
            else:
                # Use a constant (non-dispersive) permittivity
                medium = td.Medium(permittivity=self.permittivity_value)

            slab = td.Structure(geometry=box, medium=medium)

        # =====================================================================
        # Boundary conditions
        # =====================================================================

        # Absorbers on the propagation axis; periodic on transverse axes (unless overridden)
        boundaries = td.BoundarySpec(
            x=td.Boundary(plus=td.Absorber(num_layers=self.absorbers), minus=td.Absorber(num_layers=self.absorbers)) if (self.direction == "x" or self.boundaries == "absorbers") else td.Boundary.periodic(),
            y=td.Boundary(plus=td.Absorber(num_layers=self.absorbers), minus=td.Absorber(num_layers=self.absorbers)) if (self.direction == "y" or self.boundaries == "absorbers") else td.Boundary.periodic(),
            z=td.Boundary(plus=td.Absorber(num_layers=self.absorbers), minus=td.Absorber(num_layers=self.absorbers)) if (self.direction == "z" or self.boundaries == "absorbers") else td.Boundary.periodic(),
        )

        # Mesh override: enforce fine grid resolution inside the slab region
        mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(center=(0,0,0), size=(
                      self.t_slab_x if self.direction == "x"  else td.inf, 
                      self.t_slab_y if self.direction == "y"  else td.inf, 
                      self.t_slab_z if self.direction == "z"  else td.inf
                      )),
            dl=( (self.lambda_range[1] / (self.min_steps_per_lambda)) / np.sqrt(self.permittivity_value) #  grids per smallest wavelength in medium
            ,)*3
        )

        # Assemble all simulation components into a dictionary
        return {
            "size": self.sim_size,
            "grid_spec": td.GridSpec.auto(
                min_steps_per_wvl=self.min_steps_per_lambda,
                wavelength=self.lambda0,
                dl_min=self.dl,
                max_scale=1.2,
            ),
            "sources": [self.source_def],
            "monitors": self.monitors_names,
            "run_time": self.t_stop,
            "boundary_spec": boundaries,
            "normalize_index": None,
            "structures": [] if self.ref_only else ([slab] if not self.multiplicate_size else slabs),
            "ref_structure": [] if self.ref_only else ([slab] if not self.multiplicate_size else slabs_ref),
        }
    
    def simulation_definition(self):
        """Assemble the Tidy3D Simulation object and attach monitors that depend on num_time_steps."""

        definitions = self.sim_object
        sim = td.Simulation(
            center=(0, 0, 0),
            size=definitions['size'],
            grid_spec=definitions['grid_spec'],
            sources=definitions['sources'],
            monitors=definitions['monitors'],
            run_time=definitions['run_time'],
            shutoff=self.shutoff,  # Simulation stops when field decays below this threshold
            boundary_spec=definitions['boundary_spec'],
            normalize_index=None,
            structures=definitions['structures'],
            subpixel=self.subpixel,
            medium=td.Medium(permittivity=self.sim_bg),
        )
        
        # --- Monitors that need num_time_steps (only available after Simulation is created) ---

        # E-field time monitor at the output surface of the slab (100 snapshots)
        if "time_monitorFieldOut" in self.monitors:
            time_monitorFieldOut = td.FieldTimeMonitor(
                center = (
                            (self.t_slab_x)*0.5 if self.direction == "x" else 0, 
                            (self.t_slab_y)*0.5 if self.direction == "y" else 0, 
                            (self.t_slab_z)*0.5 if self.direction == "z" else 0
                            ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                    start=0,
                    stop=self.t_stop,
                    interval=int(sim.num_time_steps/100),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldOut",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldOut]})

        # E-field time monitor on a lateral cross-section through the center
        if "time_monitorFieldLateral" in self.monitors:
            time_monitorFieldLateral = td.FieldTimeMonitor(
                center = (
                           0,0,0
                            ),
                size = (
                    0 if self.direction == "z" else self.Lx, 
                    0 if self.direction == "y" else self.Ly, 
                    0 if self.direction == "x" else self.Lz
                    ),
                    start=0,
                    stop=20e-12,
                    interval=200,
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldLateral",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldLateral]})

        # Frequency-domain E-field monitor at the output surface
        if "freq_monitorFieldOut" in self.monitors:
            freq_monitorFieldOut = td.FieldMonitor(
                center = (
                            (self.t_slab_x)*0.5 if self.direction == "x" else 0, 
                            (self.t_slab_y)*0.5 if self.direction == "y" else 0, 
                            (self.t_slab_z)*0.5 if self.direction == "z" else 0
                            ),
                size = (
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ),
                    fields=["Ex","Ey","Ez"],
                    freqs =self.monitor_freqs,
                    name="freq_monitorFieldOut",
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[freq_monitorFieldOut]})

        # E-field time monitor at the center plane of the slab
        if "time_monitorFieldCenter" in self.monitors:
            time_monitorFieldCenter = td.FieldTimeMonitor(
                center = [0,0,0],
                size = [
                    0 if self.direction == "x" else td.inf, 
                    0 if self.direction == "y" else td.inf, 
                    0 if self.direction == "z" else td.inf
                    ],
                    start=0,
                    stop=self.t_stop,
                    interval=int(sim.num_time_steps/100),
                    fields=["Ex", "Ey", "Ez"],
                    name="time_monitorFieldCenter"
                    
                )
            
            sim = sim.copy(update={"monitors":list(sim.monitors)+[time_monitorFieldCenter]})

        
        return sim 
    
    
    def plot_sim_layout(self):
        sim = self.sim
        plt.figure(dpi=200)
        freqs_plot = (self.freq_range[0], self.freq_range[1])
        fig, ax = plt.subplots(1, tight_layout=True, figsize=(16, 8))
        if self.direction == "x":
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
        elif self.direction == "z":
            sim.plot_eps(x=0, freq=freqs_plot[0], ax=ax)
            plt.show()
            plt.figure(dpi=250)
            freqs_plot = (self.freq_range[0], self.freq_range[1])
            fig, ax = plt.subplots(1, tight_layout=True, figsize=(16, 8))
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
            plt.show()
        elif self.direction == "y":
            sim.plot_eps(z=0, freq=freqs_plot[0], ax=ax)
            
        plt.tight_layout()
        plt.show()

    def estimate_cost(self):
        sim = self.sim
        id =web.upload(sim,task_name="test_net")
        cost = web.estimate_cost(id)
        web.delete(id)
        return cost
    
    def run_sim(self,run_free:bool = True,folder_description:str="",max_grid_size:int = 100,max_time_steps:int=50e3, 
                load:bool=True, run:bool=True,add_ref:bool=True,monitor:bool=False,name_sim:str=""):
        """
        If run for free is set to True the simulation won't be executed if the predefined max grid size or time step values are surpassed. 
        To fix this, reduce the min_steps_per_wvl on the class definition, decrease run time, or set run_free to False.
        Submits a Simulation to server, starts running, monitors progress, downloads, and loads results as a SimulationData object.
        Pushes taskid into task_name_def.txt
        """

        sim = self.sim

        time_steps = sim.num_time_steps
        grid_size = sim.num_cells*1e-6

        if (time_steps < max_time_steps and grid_size < max_grid_size) or not run_free:

            size = self.t_slab_x if self.direction == "x" else self.t_slab
            size = self.t_slab_y if self.direction == "y" else self.t_slab
            size = self.t_slab_z if self.direction == "z" else self.t_slab

            folder_name = folder_description
            task_name_def = f'{self.structure_name}_eps_{self.permittivity_value}_size_{size:.3g}_runtime_{self.runtime:.3g}_lambdaRange_{self.lambda_range[0]:.3g}-{self.lambda_range[1]:.3g}_incidence_{self.direction}' if not self.sim_name else self.sim_name 
            if name_sim:
                task_name_def=name_sim
            #Normalization task
            if add_ref:
                sim0 = sim.copy(update={'structures':[] if not self.h5_bg else self.sim_object["ref_structure"]})
                id_0 =web.upload(sim0, folder_name=folder_name,task_name=task_name_def+'_0', verbose=self.verbose)
                if run:
                    web.start(task_id = id_0)
                    if monitor:
                        web.monitor(task_id=id_0,verbose=self.verbose)

            
            id =web.upload(sim, folder_name=folder_name,task_name=task_name_def, verbose=self.verbose)
            if run:
                web.start(task_id = id)
                if monitor:
                    web.monitor(task_id=id,verbose=self.verbose)

            #Store ids in an file 

            if run:
                ids = (id_0 if add_ref else "") + '\n' + id
                incidence_folder = self.direction+"_incidence"
                file_path = rf"./data/{folder_name}/{incidence_folder}/{task_name_def}.txt"
                # Check if the folder exists
                if not os.path.exists( rf"./data/{folder_name}/{incidence_folder}"):
                    os.makedirs(rf"./data/{folder_name}/{incidence_folder}")
                    print(f"Folder '{folder_name}/{incidence_folder}' created successfully.")

                # Open file in write mode
                with open(file_path, "w") as file:
                    # Write the string to the file
                    file.write(ids)

            
        else: 
            raise Exception("Reduce time steps or grid size")
        
        if load:
                if add_ref:
                 sim_data0=web.load(id_0)
                else: 
                    sim_data0=""

                sim_data=web.load(id)
                return (sim_data, sim_data0,task_name_def)
        else:
            return False
        
       