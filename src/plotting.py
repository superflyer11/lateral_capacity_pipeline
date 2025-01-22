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

def create_plot(data, x_label, y_label, title, save_as, show, enforce_pass_through_zero, annotate_all_datapoints, annotate_last_datapoint, large_markers, vertical_axis="left", horizontal_axis="bottom", scale_up=1, x_log_scale=False, y_log_scale=False):
    linestyle = "-"
    plt.tight_layout()
    fig, ax = plt.subplots()
    max_x, max_y = float('-inf'), float('-inf')
    min_x, min_y = float('inf'), float('inf')
    for x, y, label, color, cutoff in data:
        if x is not None and y is not None:
            # Convert to NumPy array if data is a Pandas Series
            if isinstance(x, pd.Series):
                x = x.to_numpy()
            if isinstance(y, pd.Series):
                y = y.to_numpy()
                
            if cutoff:
                pass
                # mask_elastic = abs(y) < abs(cutoff)
                # mask_plastic = abs(y) >= abs(cutoff)
                # plt.plot(x[mask_elastic], y[mask_elastic], linestyle=linestyle, color=color, label=f"label")
                # plt.plot(x [mask_plastic], y[mask_plastic], linestyle=linestyle, color='orange', label=label)
            else:
                if label == "":
                    label = None
                plt.plot(x, y, linestyle=linestyle, color=color, label=label)
                if large_markers:
                    plt.scatter(x, y, color=color, marker='x')
                else:
                    plt.scatter(x, y, color=color, marker='x', s = 3)
                if annotate_last_datapoint:
                    ax.annotate(f'{y[-1]:.4g}', xy=(x[-1], y[-1]), xytext=(x[-1], y[-1]*0.92), textcoords='data',
                                fontsize=8,  # Adjust font size
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                                arrowprops=dict(facecolor='black', shrink=0.05, headwidth=5, headlength=5, width=0.5))
                # elif annotate_all_datapoints:
                #     for i in range(len(x)):
                #         ax.annotate(f'{len:.4g}', xy=(x[i], y[i]), xytext=(x[i], y[i]*0.92), textcoords='data',
                #                     fontsize=8,  # Adjust font size
                #                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                #                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=5, headlength=5, width=0.5))
            max_x = max(max_x, max(x))
            max_y = max(max_y, max(y))
            min_x = min(min_x, min(x))
            min_y = min(min_y, min(y))
    
    if enforce_pass_through_zero:
        if max_y < 0:
            ax.set_ylim(top=0)
        elif min_y > 0:
            ax.set_ylim(bottom=0)
        
    if vertical_axis == "right":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['left'].set_visible(False)
        ax.spines['right'].set_position('zero')
        ax.set_xlim(right=max_x)
    else:
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(True)
        
    if horizontal_axis == "top":
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(False)
        ax.spines['top'].set_position('zero')
    else:
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(True)    
    
    if not vertical_axis == "right" and not horizontal_axis == "top":
        plt.axvline(0, color='black', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
    elif vertical_axis == "right" and horizontal_axis == "top":
        plt.axvline(0, color='black', linewidth=2)
        plt.axhline(0, color='black', linewidth=2)
    
    if scale_up > 1:
        ax.set_ylim([min_y * scale_up, max_y * scale_up])
    # ax.set_xlim(left=-5)
    # Add axis labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_log_scale == True:
        ax.set_xscale('log')
    if y_log_scale == True:
        ax.set_yscale('log')
    if not title == "":
        ax.set_title(title)
    # print(data[0][2])
    if not data[0][2] in ["", None]:
        ax.legend(handlelength=0, handletextpad=0)
    ax.grid(False)
    if save_as:
        plt.savefig(save_as,transparent=True)
        if not show:
            plt.close()
        return save_as
    elif not show:
        plt.close()
        
        
def plot_sig_eq_vs_e_zz(sig_eq, e_zz, save_as: str =None):
    return plot_2d_with_quiver(e_zz, sig_eq, 'Axial Strain $\epsilon_{zz}$', 'Equivalent Stress $\sigma_{eq}$', '$\sigma_{eq}$ - Axial Strain',save_as=save_as)



def plot_x_ys(x_array: list, y_arrays, labels: list, colors: list | None = None, cutoffs=None, x_label="", y_label="", title="", save_as: str = None, show=False, enforce_pass_through_zero=False, annotate_all_datapoints=False, annotate_last_datapoint=False, large_markers=False,vertical_axis="left", horizontal_axis="bottom", scale_up=1, x_log_scale=False, y_log_scale=False):
    data = []
    for i in range(len(y_arrays)):
        data.append((x_array, y_arrays[i], labels[i], colors[i] if colors else None, None))
    return create_plot(data, x_label, y_label, title, save_as, show, enforce_pass_through_zero,annotate_all_datapoints,annotate_last_datapoint,large_markers,vertical_axis, horizontal_axis, scale_up, x_log_scale, y_log_scale)

def init_axes_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def plot_cone_and_points(ax, radius, start_height=-50, end_height=100):
    # Step 1: Define the direction vector of the space diagonal
    diagonal_direction = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])

    # Step 2: Generate the cone along the direction of the space diagonal
    height = np.linspace(0, end_height - start_height, 50)
    angle = np.linspace(0, 2 * np.pi, 100)
    Height, Angle = np.meshgrid(height, angle)

    # Define two orthogonal vectors that are perpendicular to the diagonal direction
    orthogonal_vector_1 = np.array([1.0, -1.0, 0.0])
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)
    orthogonal_vector_2 = np.cross(diagonal_direction, orthogonal_vector_1)

    # Compute the radius of the cone at each height (linearly increases from zero)
    cone_radius = radius * (Height / (end_height - start_height))
    # print(cone_radius[-1])
    # Compute the coordinates of the cone
    X = (cone_radius * np.cos(Angle) * orthogonal_vector_1[0] +
         cone_radius * np.sin(Angle) * orthogonal_vector_2[0] +
         (Height + start_height) * diagonal_direction[0])
    Y = (cone_radius * np.cos(Angle) * orthogonal_vector_1[1] +
         cone_radius * np.sin(Angle) * orthogonal_vector_2[1] +
         (Height + start_height) * diagonal_direction[1])
    Z = (cone_radius * np.cos(Angle) * orthogonal_vector_1[2] +
         cone_radius * np.sin(Angle) * orthogonal_vector_2[2] +
         (Height + start_height) * diagonal_direction[2])

    # Step 3: Plot the cone
    ax.plot_surface(X, Y, Z, alpha=0.5, color='m')

    # Step 4: Set the base of the cone for the circle at the end height
    base_point = end_height * diagonal_direction

    # Step 5: Plot a circle around the base point to indicate it lies on the cone plane
    circle_angle = np.linspace(0, 2 * np.pi, 100)
    cone_radius_at_base = radius
    circle_x = (cone_radius_at_base * np.cos(circle_angle) * orthogonal_vector_1[0] +
                cone_radius_at_base * np.sin(circle_angle) * orthogonal_vector_2[0] +
                base_point[0])
    circle_y = (cone_radius_at_base * np.cos(circle_angle) * orthogonal_vector_1[1] +
                cone_radius_at_base * np.sin(circle_angle) * orthogonal_vector_2[1] +
                base_point[1])
    circle_z = (cone_radius_at_base * np.cos(circle_angle) * orthogonal_vector_1[2] +
                cone_radius_at_base * np.sin(circle_angle) * orthogonal_vector_2[2] +
                base_point[2])
    ax.plot(circle_x, circle_y, circle_z, color='b')
    # ax.scatter(*base_point, color='b', s=100)

    # # Step 6: Define a plane that cuts through the cone at the base height
    # plane_normal = diagonal_direction
    # plane_point = base_point  # Plane passes through the base point

    # # Step 7: Find intersection points of the plane with the cone
    # cone_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    # plane_distances = np.dot(cone_points - plane_point, plane_normal)
    # intersection_indices = np.where(np.abs(plane_distances) < 0.05)[0]
    # intersection_points = cone_points[intersection_indices]

    # # Step 8: Pick three random points from the intersection points
    # if intersection_points.shape[0] >= 3:
    #     random_indices = np.random.choice(intersection_points.shape[0], 3, replace=False)
    #     random_points = intersection_points[random_indices]
    #     # Plot the three random points and lines connecting them to the base point
    #     for i in range(3):
    #         ax.scatter(random_points[i, 0], random_points[i, 1], random_points[i, 2], color='g', s=50)
    #         ax.plot([base_point[0], random_points[i, 0]],
    #                 [base_point[1], random_points[i, 1]],
    #                 [base_point[2], random_points[i, 2]], color='k', linestyle='--')
    #         distance = np.linalg.norm(random_points[i] - base_point)
    #         print(f"Distance from base point to point {i+1}: {distance:.2f}")
    # else:
    #     print("Not enough intersection points found to select three random points.")


# Plot stress history with classification based on tau_oct
def plot_stress_history(ax, sig_1, sig_2, sig_3, tau_oct = None, tau_oct_limit = None, save_as: str =None):
    if tau_oct and tau_oct_limit:
        mask_elastic = tau_oct < tau_oct_limit
        mask_plastic = tau_oct >= tau_oct_limit
        if np.any(mask_elastic):
            ax.plot(sig_1[mask_elastic], sig_2[mask_elastic], sig_3[mask_elastic], color='b', label='Elastic', linewidth=2)
        if np.any(mask_plastic):
            ax.plot(sig_1[mask_plastic], sig_2[mask_plastic], sig_3[mask_plastic], color='orange', label='Plastic', linewidth=2)
    else:
        ax.plot(sig_1, sig_2, sig_3, color='orange', linewidth=2)
    # vol_stress_value = (sig_1 + sig_2 + sig_3) / 3
    # diagonal_direction = np.array([1, 1, 1]) / np.linalg.norm([1, 1, 1])
    # vol_stress_x = vol_stress_value * diagonal_direction[0]
    # vol_stress_y = vol_stress_value * diagonal_direction[1]
    # vol_stress_z = vol_stress_value * diagonal_direction[2]
    # ax.plot(vol_stress_x, vol_stress_y, vol_stress_z, color='r', linestyle='--', label='Volumetric Stress')
    # ax.plot([vol_stress_x[-1], sig_1[-1]], [vol_stress_y[-1], sig_2[-1]], [vol_stress_z[-1], sig_3[-1]], color='g', linestyle='--', label='Deviatoric Stress')

# Plot metadata like labels and planes
def plot_meta(ax, elev, azim, roll):
    ax.set_xlabel(r'$\sigma_1$')
    ax.set_ylabel(r'$\sigma_2$')
    ax.set_zlabel(r'$\sigma_3$')
    ax.set_title('3D Plot of Principal Stresses')

    # Plot planes and add arrowheads with labels
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    zlim = ax.get_zlim()
    text_fontsize = 10
    # y-plane
    ax.plot([0, 0], ylim, [0, 0], color='k', linestyle='--', alpha=0.5)
    ax.text(0, ylim[1] + text_fontsize * 2.5, 0, r'$\sigma_2$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    # x-plane
    ax.plot(xlim, [0, 0], [0, 0], color='k', linestyle='--', alpha=0.5)
    ax.text(xlim[1] + text_fontsize * 2.5, 0, 0, r'$\sigma_1$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    # z-plane
    ax.plot([0, 0], [0, 0], zlim, color='k', linestyle='--', alpha=0.5)
    ax.text(0, 0, zlim[1] + text_fontsize * 2.5, r'$\sigma_3$', color='k', fontsize=text_fontsize, verticalalignment='center_baseline', horizontalalignment='center')

    # limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    # ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.view_init(elev=elev, azim=azim, roll=roll)
    # ax.legend()

    # ax.set_axis_off()
    # plt.tight_layout()

def plot_stress_field(sig_1, sig_2, sig_3, elev, azim, roll, save_as: str =None, show = False):
    fig, ax = init_axes_3d()
    # plot_cone_and_points(ax, radius=cone_radius[-1], start_height = cone_tip_p, end_height=max_p)
    plot_stress_history(ax, sig_1, sig_2, sig_3)
    plot_meta(ax, elev, azim, roll)
    
    ax.grid(True)
    if save_as:
        plt.savefig(save_as)
        if not show:
            plt.close()
        return save_as
    elif not show:
        plt.close()
    
