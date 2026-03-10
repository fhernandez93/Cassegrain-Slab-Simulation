import numpy as np
import tidy3d.web as web
import tidy3d as td
from tidy3d.components.data.data_array import SpatialDataArray
import h5py
import matplotlib.pyplot as plt
from pathlib import Path


class loadFromFile:
    """
    Load results from txt or hdf5 data files
    """
    def __init__(self, key:str="", file_path:str = "",only_download:bool=False, save_path:str="", get_ref:bool=True, verbose:bool=True,configured:bool=True):
        if not key:
            raise Exception("No API key was provided")
        elif configured:
            web.configure(key)

        if not file_path:
            raise Exception("No file was provided")
        
        self.list_id = []


        if (Path(file_path).suffix==".txt"):
            with open(file_path, 'r') as file:
                for line in file:
                    self.list_id+=[line.strip()] 

            self.structure_name = Path(file_path).stem
            if save_path=="":
                store_path =  fr"H:\phd stuff\tidy3d/output/{file_path[file_path.find("data"):]}/Data"
            else:
                store_path =  save_path + rf"/{file_path[file_path.find("2D SHU Project")+len("2D SHU Project")+1:]}/Data"
            self.only_download = only_download
            # raise Exception({store_path})
            if only_download:
                if Path(store_path+".hdf5").is_file():
                    print("File already exists!")
                else:
                    if get_ref:
                        try:
                            web.download(self.list_id[0] ,path=store_path+"_0.hdf5", verbose=verbose)
                        except:
                            print("No Reference Simulation was downloaded")

                    web.download(self.list_id[1],path=store_path+".hdf5", verbose=verbose)
                return None
            
            


        if  Path(store_path+".hdf5").is_file():
            if get_ref:
                try:
                    self.sim_data0 =  td.SimulationData.from_hdf5(store_path+"_0.hdf5")
                except:
                    print("No Reference Simulation was loaded")
            self.sim_data =  td.SimulationData.from_hdf5(store_path+".hdf5")
        else:
            if get_ref:
                try:
                        self.sim_data0=web.load(self.list_id[0] ,path=store_path+"_0.hdf5")
                except:
                        print("No Reference Simulation was found")

            self.sim_data = web.load(self.list_id[1],path=store_path+".hdf5")

        
        try:
            self.cost = web.real_cost(self.list_id[1])
        except:
            self.cost = "No cost was found"

        self.run_time = self.sim_data.simulation.run_time
        self.fwidth=self.sim_data.simulation.sources[0].source_time.fwidth
        self.freq0=self.sim_data.simulation.sources[0].source_time.freq0
        self.run_time = self.sim_data.simulation.run_time*1e12
        try:
            self.monitor_lambdas = td.C_0/np.array(np.array(self.sim_data.simulation.monitors).freqs)
        except:
            self.monitor_lambdas =np.array([td.C_0/((self.freq0) - ((self.fwidth*2))),td.C_0/((self.freq0) +((self.fwidth*2)))])



        self.final_decay = self.sim_data.final_decay_value
        self.description = Path(file_path).stem
        try:
            self.resolution= self.sim_data.simulation.grid_spec.grid_x.min_steps_per_wvl
        except:
            self.resolution= ""
        self.time_per_fwidth = self.sim_data.simulation.run_time*self.fwidth


        
    def __str__(self):

        calculated_data_str = ('Simulation Parameters (wavelengths are expressed in um):\n' +
            f'lambda_range: {self.monitor_lambdas[-1]:.3g} - {self.monitor_lambdas[0]:.3g} um \n'+
            f"lambdaw (pulse) {td.C_0/self.fwidth} \n"+
            f"lambda0 {td.C_0/self.freq0} \n"+
            f"Runtime = {self.run_time} \n"+
            f"resolution = {self.resolution} \n"+
            f"time_per_fwidth = {self.time_per_fwidth} \n"+
            f"final decay value = {self.final_decay} \n"
        ) if not self.only_download else "Info was only downloaded in the specified location"

        return calculated_data_str
    
    def get_sources(self,a=5/3):

        ax1 = self.sim_data.simulation.sources[0].source_time.plot(times=np.linspace(0, self.sim_data.simulation.run_time, 10000))

        # Extract the data from the Axes object
        line = ax1.get_lines()[0]
        times = line.get_xdata()
        amplitude_time = line.get_ydata()

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(times*1e12, amplitude_time)
        ax.set_xlabel(r"t[ps]")
        ax.set_ylabel('Amplitude')
        ax.set_title('Source Amplitude')
        ax.legend(['Source Spectrum'])
        plt.show()


        # Constants
        c = td.C_0  # Speed of light

        # Define the time array
        times = np.linspace(0, self.sim_data.simulation.run_time, 10000)

        # Plot the spectrum and get the Axes object
        ax2 = self.sim_data.simulation.sources[0].source_time.plot_spectrum(times=times)

        # Extract the data from the Axes object
        line = ax2.get_lines()[0]
        freqs = line.get_xdata()
        amplitude_frequ = line.get_ydata()

        # Convert frequency to wavelength (in meters)
        wavelengths = c / freqs

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(a/wavelengths, amplitude_frequ)
        ax.set_xlabel(r"$\nu$'")
        ax.set_ylabel('Amplitude')
        ax.set_title('Source Spectrum')
        ax.legend(['Source Spectrum'])
        plt.show()

        return (times,amplitude_time,wavelengths,amplitude_frequ)
       
        