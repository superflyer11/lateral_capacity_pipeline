from pydantic import BaseModel, FilePath
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut

class PlotConfig(BaseModel):
    csv_files: Dict[str, FilePath]  # Dictionary of labels and valid file paths
    output_dir: str
    suptitle: str
    figsize: tuple = (6, 10)

### **Base Plotter Class with Pydantic Validation**

class BasePlotter:
    def __init__(self, config: PlotConfig):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=self.config.figsize)

    def load_data(self, csv_filepath: FilePath):
        return pd.read_csv(csv_filepath)

    def setup_plot(self, xlabel: str, ylabel: str, title: str, xlim=None, ylim=None):
        self.ax.clear()
        self.ax.margins(x=0, y=0)
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.fig.suptitle(f'{self.config.suptitle}')
        
        # Set the x-axis to have ticks on top
        self.ax.xaxis.set_ticks_position('top')
        self.ax.tick_params(axis='x', labeltop=True)


    def save_plot(self, save_as: str):
        self.ax.legend()
        path = f"{self.config.output_dir}/{save_as}"
        self.fig.savefig(path)
        plt.close(self.fig)
        return path

### **Time Plotter Class**
class TimePlotter(BasePlotter):
    @ut.track_time(f"PLOTTING sig_vM AGAINST e")
    def sigvM_e(self, save_as: str):
        self.setup_plot(r'Equivalent Strain $\varepsilon_{\text{eq}}$ [-]', r'Von Mises Stress $\sigma_{\text{vM}}$ [MPa]', r'$\sigma_{\text{vM}}$ vs $\varepsilon_{\text{eq}}$')
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            # Extract stress components
            stress_xx = df['avg(STRESS (0))']
            stress_xy = df['avg(STRESS (1))']
            stress_xz = df['avg(STRESS (2))']
            stress_yy = df['avg(STRESS (4))']
            stress_yz = df['avg(STRESS (5))']
            stress_zz = df['avg(STRESS (8))']
            
            # Calculate von Mises stress
            von_mises_stress = np.sqrt(0.5 * (
                (stress_xx - stress_yy)**2 + (stress_yy - stress_zz)**2 + (stress_zz - stress_xx)**2 +
                6 * (stress_xy**2 + stress_xz**2 + stress_yz**2)
            ))
            
            # Extract strain components
            strain_xx = df['avg(STRAIN (0))']
            strain_xy = df['avg(STRAIN (1))']
            strain_xz = df['avg(STRAIN (2))']
            strain_yy = df['avg(STRAIN (4))']
            strain_yz = df['avg(STRAIN (5))']
            strain_zz = df['avg(STRAIN (8))']
            
            # Calculate volumetric strain
            e_v = strain_xx + strain_yy + strain_zz
            
            # Calculate deviatoric strains
            dev_strain_xx = strain_xx - e_v / 3
            dev_strain_yy = strain_yy - e_v / 3
            dev_strain_zz = strain_zz - e_v / 3
            dev_strain_xy = strain_xy
            dev_strain_xz = strain_xz
            dev_strain_yz = strain_yz
            
            # Calculate equivalent (von Mises) strain
            von_mises_strain = np.sqrt(2/3 * (
                dev_strain_xx**2 + dev_strain_yy**2 + dev_strain_zz**2 +
                2 * (dev_strain_xy**2 + dev_strain_xz**2 + dev_strain_yz**2)
            ))

            # Plot von Mises stress against equivalent strain
            self.ax.plot(von_mises_strain, von_mises_stress, label=label, marker='o')
        return self.save_plot(save_as)

    @ut.track_time("PLOTTING J against p")
    def J_p(self, save_as: str):
        self.setup_plot(r'Mean Stress p [MPa]', r'Deviatoric Stress J [MPa]', r'J vs p')
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            # Extract stress components
            stress_xx = df['avg(STRESS (0))']
            stress_xy = df['avg(STRESS (1))']
            stress_xz = df['avg(STRESS (2))']
            stress_yx = df['avg(STRESS (3))']
            stress_yy = df['avg(STRESS (4))']
            stress_yz = df['avg(STRESS (5))']
            stress_zx = df['avg(STRESS (6))']
            stress_zy = df['avg(STRESS (7))']
            stress_zz = df['avg(STRESS (8))']
            p = (stress_xx + stress_yy + stress_zz) / 3.0
            J = np.sqrt(1.5 * (
                (stress_xx - p)**2 +
                (stress_yy - p)**2 +
                (stress_zz - p)**2 +
                2 * (stress_xy**2 + stress_yz**2 + stress_zx**2)
            ))

            # Plot von Mises stress against equivalent strain
            self.ax.plot(J, p, label=label, marker='o')
        return self.save_plot(save_as)

### **Depth Plotter Class**
class DepthPlotter(BasePlotter):
    def plot_against_depth(self, x_data, z_data, label):
        self.ax.plot(x_data, z_data, label=label, marker='o')

    @ut.track_time("PLOTTING disp_x AGAINST z")
    def plot_displacement_vs_depth(self, save_as: str):
        self.setup_plot('Displacement in x-direction [m]', 'Depth [m]', 'Displacement [m] against Depth [m]', xlim=(-0.1, 0.1), ylim=(-40, 0))
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            displacement_x = df['DISPLACEMENT:0']
            depth = df['Points:2']
            self.plot_against_depth(displacement_x, depth, label)
        return self.save_plot(save_as)

    @ut.track_time(f"PLOTTING sig_xx AGAINST z")
    def sigxx_z(self, save_as: str):
        self.setup_plot(r'$\sigma$ [MPa]', 'Depth [m]', r'$\sigma$ [MPa] against Depth [m]', ylim=(-40, 0))
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            stress_xx = df['STRESS:0']
            depth = df['Points:2']
            self.plot_against_depth(stress_xx, depth, label)
        return self.save_plot(save_as)

    @ut.track_time("PLOTTING sig_vM AGAINST z")
    def plot_von_mises_vs_depth(self, save_as: str):
        self.setup_plot(r'$\sigma_{\text{eq}}$ [MPa]', 'Depth [m]', r'$\sigma_{\text{vM}}$ against Depth [m]', ylim=(-40, 0))
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            stress_xx = df['STRESS:0']
            stress_yy = df['STRESS:4']
            stress_zz = df['STRESS:8']
            stress_xy = df['STRESS:1']
            stress_xz = df['STRESS:2']
            stress_yz = df['STRESS:5']
            von_mises_stress = np.sqrt(0.5 * (
                (stress_xx - stress_yy)**2 + (stress_yy - stress_zz)**2 + (stress_zz - stress_xx)**2 +
                6 * (stress_xy**2 + stress_xz**2 + stress_yz**2)
            ))
            depth = df['Points:2']
            self.plot_against_depth(von_mises_stress, depth, label)
        return self.save_plot(save_as)
    
    @ut.track_time("PLOTTING E AGAINST z")
    def E_z(self, save_as: str):
        self.setup_plot(r'$E$ [MPa]', 'Depth [m]', 'Stress [MPa] against Depth [m]', ylim=(-40, 0))
        for label, csv_filepath in self.config.csv_files.items():
            df = self.load_data(csv_filepath)
            
            stress_xx = df['STRESS:0']
            stress_yy = df['STRESS:3']
            stress_zz = df['STRESS:6']
            strain_xx = df['STRAIN:0']
            strain_yy = df['STRAIN:3']
            strain_zz = df['STRAIN:6']
            E1 = abs((stress_xx -  0.3 * (stress_yy + stress_zz)) / strain_xx)
            E2 = abs((stress_yy -  0.3 * (stress_xx + stress_zz)) / strain_yy)
            E3 = abs((stress_zz -  0.3 * (stress_xx + stress_yy)) / strain_zz)
            depth = df['Points:2']
            self.plot_against_depth(E1, depth, label)
            self.plot_against_depth(E2, depth, label)
            self.plot_against_depth(E3, depth, label)
        return self.save_plot(save_as)
