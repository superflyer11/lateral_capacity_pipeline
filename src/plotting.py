import numpy as np
print(np.__file__)
import pandas as pd
import matplotlib.pyplot as plt

import utils as ut

@ut.track_time("PLOTTING DISPLACEMENT AGAINST DEPTH")
def plot_displacement_vs_points(csv_files: dict[str, str], save_as: str, mesh_name: str):
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    plt.ylim(-40,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
        displacement_x = df['DISPLACEMENT:0']
        z = df['Points:2']
        plt.plot(displacement_x, z , label=label, marker='o')
    
    # Label the axes
    plt.xlabel('Displacement in x-direction [m]')
    plt.ylabel('Depth [m]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('Displacement [m] against Depth [m]')
    plt.suptitle(f'{mesh_name}')

    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()
    return save_as

@ut.track_time("PLOTTING STRESS AGAINST DEPTH")
def plot_stress_vs_points(csv_files: dict[str, str], save_as: str, mesh_name: str):
    """
    Load two CSV files, extract DISPLACEMENT:0 and Points:2 columns,
    and generate a plot with Points:2 on the y-axis and DISPLACEMENT:0 on the x-axis.
    Args:
        save_as (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    plt.ylim(-40,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
        stress_xx = df['STRESS:0']
        z = df['Points:2']
        plt.plot(stress_xx, z , label=label, marker='o')
    
    # Label the axes
    plt.xlabel(r'$\sigma_{xx}$ [MPa]')
    plt.ylabel('Depth [m]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('Stress [MPa] against Depth [m]')
    plt.suptitle(f'{mesh_name}')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()
    return save_as


@ut.track_time("PLOTTING STRESS AGAINST DEPTH")
def calculate_and_plot_q_p(csv_files: dict[str, str], save_as: str, mesh_name: str):
    """
    Load two CSV files, extract DISPLACEMENT:0 and Points:2 columns,
    and generate a plot with Points:2 on the y-axis and DISPLACEMENT:0 on the x-axis.
    Args:
        save_as (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    # plt.ylim(,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
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
        q = np.sqrt(1.5 * (
            (stress_xx - p)**2 +
            (stress_yy - p)**2 +
            (stress_zz - p)**2 +
            2 * (stress_xy**2 + stress_yz**2 + stress_zx**2)
        ))
        # z = df['Points:2']
        plt.plot(p, q , label=label, marker='o')
    
    # Label the axes
    plt.xlabel('Mean effective stress p [MPa]')
    plt.ylabel('Deviatoric Stress q [MPa]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('q-p plane')
    plt.suptitle(f'{mesh_name}')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()
    return save_as

@ut.track_time("PLOTTING E AGAINST DEPTH")
def calculate_and_plot_E_vs_depth(csv_files: dict[str, str], save_as: str, mesh_name: str):
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    # plt.ylim(,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
        stress_xx = df['avg(STRESS (0))']
        strain_xx = df['avg(STRAIN (0))']
        E = stress_xx / strain_xx
        z = df['Points:2']
        plt.plot(E, z , label=label, marker='o')
    
    # Label the axes
    plt.xlabel('Equivalent E (assuming no plastic loading) [MPa]')
    plt.ylabel('Depth [m]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('Depth against E')
    plt.suptitle(f'{mesh_name}')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()
    return save_as
    

@ut.track_time("PLOTTING STRESS AGAINST STRAIN")
def calculate_and_plot_stress_strain(csv_files: dict[str, str], save_as: str, mesh_name: str):
    """
    Load two CSV files, extract DISPLACEMENT:0 and Points:2 columns,
    and generate a plot with Points:2 on the y-axis and DISPLACEMENT:0 on the x-axis.
    Args:
        save_as (str): Path to save the generated plot image.
    """
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    # plt.ylim(,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
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
        q = np.sqrt(1.5 * (
            (stress_xx - p)**2 +
            (stress_yy - p)**2 +
            (stress_zz - p)**2 +
            2 * (stress_xy**2 + stress_yz**2 + stress_zx**2)
        ))
        
        strain_xx = df['avg(STRAIN (0))']
        strain_xy = df['avg(STRAIN (1))']
        strain_xz = df['avg(STRAIN (2))']
        strain_yx = df['avg(STRAIN (3))']
        strain_yy = df['avg(STRAIN (4))']
        strain_yz = df['avg(STRAIN (5))']
        strain_zx = df['avg(STRAIN (5))']
        strain_zy = df['avg(STRAIN (5))']
        strain_zz = df['avg(STRAIN (5))']
                
        e_v = strain_xx + strain_yy + strain_zz
        
        dev_strain_xx = strain_xx - e_v/3
        dev_strain_yy = strain_yy - e_v/3
        dev_strain_zz = strain_zz - e_v/3
        dev_strain_xy = strain_xy  # shear components are already deviatoric
        dev_strain_yz = strain_yz
        dev_strain_zx = strain_zx

        
        e_d = np.sqrt(2/3 * (dev_strain_xx**2 + dev_strain_yy**2 + dev_strain_zz**2 +
                       2 * (dev_strain_xy**2 + dev_strain_yz**2 + dev_strain_zx**2)))   
        
        # z = df['Points:2']
        plt.plot(q, e_d , label=label, marker='o')
    
    # Label the axes
    plt.xlabel(r'Deviatoric Strain $\epsilon_d$ [-]')
    plt.ylabel('Deviatoric Stress q [MPa]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title(r'q-$\epsilon_d$')
    plt.suptitle(f'{mesh_name}')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()
    return save_as

