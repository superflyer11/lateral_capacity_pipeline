import pandas as pd
import matplotlib.pyplot as plt

import utils as ut

@ut.track_time("PLOTTING DISPLACEMENT AGAINST DEPTH")
def plot_displacement_vs_points(csv_files: dict[str, str], save_as: str):
    plt.figure(figsize=(6, 6))
    
    plt.margins(x=0,y=0)
    plt.ylim(-40,0)
    for label, csv_filepath in csv_files.items():
        df = pd.read_csv(csv_filepath)
        displacement_x = df['DISPLACEMENT:0']
        z = df['Points:2']
        plt.plot(displacement_x, z , label=label, marker='o')
    
    # Label the axes
    plt.xlabel('Displacement in x-direction [Pa]')
    plt.ylabel('Depth [m]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('Displacement [] against Depth [m]')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()

@ut.track_time("PLOTTING STRESS AGAINST DEPTH")
def plot_stress_vs_points(csv_files: dict[str, str], save_as: str):
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
    plt.xlabel('Stress xx [Pa]')
    plt.ylabel('Depth [m]')
    
    # Add a legend
    plt.legend()
    
    # Add a title (optional)
    plt.title('Stress [Pa] against Depth [m]')
    
    # Save the plot
    plt.savefig(save_as)

    # Close the plot to free memory
    plt.close()

