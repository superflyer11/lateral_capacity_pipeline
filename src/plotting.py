import os
from pydantic import BaseModel, FilePath
from typing import Dict
import numpy as np
import pandas as pd
import utils as ut
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.ffmpeg_path'] ='/mofem_install/jupyter/thomas/ffmpeg-7.0.2-amd64-static/ffmpeg'

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


# def plot_2d_with_quiver(x, y, xlabel, ylabel, title, color='b', scale=1, linestyle='-', label=None, plastic_cutoff=None, save_as: str = None):
#     plt.figure()
#     tolerance = 1e-6
#     gradient_tolerance = 0.01
#     gradients = np.gradient(y)

#     start_idx = 0
#     for i in range(len(y)-1):
#         if not np.isclose(gradients[i], gradients[i + 1], atol=gradient_tolerance):
#             dx = (x[i] - x[start_idx]) * 0.33
#             dy = (y[i] - y[start_idx]) * 0.33
#             current_color = 'orange' if (plastic_cutoff is not None and np.isclose(y[start_idx], plastic_cutoff[start_idx], atol=tolerance)) else 'b'
#             if i - start_idx > 1:
#                 plt.quiver(x[start_idx], y[start_idx], dx, dy, color=current_color, scale=scale, angles='xy', scale_units='xy', headwidth=5, headlength=4.5,zorder=10)
#             start_idx = i

#     dx = (x[-1] - x[start_idx]) * 0.33
#     dy = (y[-1] - y[start_idx]) * 0.33
#     current_color = 'orange' if (plastic_cutoff is not None and np.isclose(y[start_idx], plastic_cutoff[start_idx], atol=tolerance)) else 'b'
#     if i - start_idx > 1:
#         plt.quiver(x[start_idx], y[start_idx], dx, dy, color=current_color, scale=scale, angles='xy', scale_units='xy', headwidth=5, headlength=4.5,zorder=10)

#     plot_color = []
#     start_idx = 0
#     for i in range(len(y)-1):
#         current_color = 'orange' if (plastic_cutoff is not None and np.isclose(plastic_cutoff[i], y[i], atol=tolerance)) else color
#         if i == 0 or plot_color[-1] != current_color:
#             if i > 0:
#                 plt.scatter(x[start_idx:i], y[start_idx:i], color=plot_color[-1], s=0.5)
#             start_idx = i
#             plot_color.append(current_color)

#     plt.scatter(x[start_idx:], y[start_idx:], color=current_color, s=0.5)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(True,zorder=0)
    
#     if save_as:
#         plt.savefig(save_as)
#         return save_as
#     plt.show()


def plot_2d_with_quiver(x, y, xlabel, ylabel, title, color='b', scale=1, linestyle='-', label=None, plastic_cutoff=None, save_as: str = None):
    plt.figure()
    tolerance = 1e-6
    
    gradient_tolerance = 0.01
    gradients = np.gradient(y)

    start_idx = 0
    for i in range(len(y)-1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        
        plt.quiver(x[i], y[i], dx, dy, color=color, angles='xy', scale_units='xy', scale=1, zorder=10)

    plt.scatter(x, y, color=color, s=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True,zorder=0)
    
    if save_as:
        filepath = os.path.join(PLOT_DIR, save_as)
        plt.savefig(f"{filepath}.png")
        return save_as
    plt.show()

def plot_2d_with_animation(x, y, xlabel, ylabel, title, color='b', scale=1, linestyle='-', label=None, plastic_cutoff=None, save_as: str = None):
    fig, ax = plt.subplots()
    tolerance = 1e-6

    ax.scatter(x, y, color=color, s=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, zorder=0)

    quiver_segments = []

    def init():
        return quiver_segments

    def update(frame):
        # Plot one quiver at a time to animate
        i = frame
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        
        # Add a new quiver to the plot
        quiver = ax.quiver(x[i], y[i], dx, dy, color=color, angles='xy', scale_units='xy', scale=1, zorder=10)
        quiver_segments.append(quiver)

        return quiver_segments

    ani = FuncAnimation(fig, update, frames=len(x) - 1, init_func=init, blit=False, repeat=False, interval=100)

    # Optional: Save the animation as MP4
    if save_as:
        FFwriter = animation.FFMpegWriter(fps=30)
        ani.save(f'{save_as}.mp4', writer = FFwriter)
        # ani.save(f"{filepath}.mp4", writer='ffmpeg', fps=30)  # Save as MP4 using FFmpeg

    plt.show()

def create_plot(data, x_label, y_label, title, save_as, show):
    linestyle = "-"
    fig, ax = plt.subplots()
    max_x, max_y = float('-inf'), float('-inf')
    for x, y, label, color, cutoff in data:
        if x is not None and y is not None:
            if cutoff:
                mask_elastic = abs(y) < abs(cutoff)
                mask_plastic = abs(y) >= abs(cutoff)
                plt.plot(x[mask_elastic], y[mask_elastic], linestyle=linestyle, color='b', label=f"label")
                plt.plot(x [mask_plastic], y[mask_plastic], linestyle=linestyle, color='orange', label=label)
            else:
                plt.plot(x, y, linestyle=linestyle, color=color, label=label)
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
    
    # Add axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    ax.grid(True)
    if save_as:
        plt.savefig(save_as)
        if not show:
            plt.close()
        return save_as
    elif not show:
        plt.close()
def plot_sig_eq_vs_e_zz(sig_eq, e_zz, save_as: str =None):
    return plot_2d_with_quiver(e_zz, sig_eq, 'Axial Strain $\epsilon_{zz}$', 'Equivalent Stress $\sigma_{eq}$', '$\sigma_{eq}$ - Axial Strain',save_as=save_as)



def plot_x_ys(x_array: list, y_arrays, labels: list, cutoffs=None, x_label="", y_label="", title="", save_as: str = None, show=False):
    data = []
    for i in range(len(y_arrays)):
        data.append((x_array, y_arrays[i], labels[i], 'g', None))
    return create_plot(data, x_label, y_label, title, save_as, show)


