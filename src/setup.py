import time
import json
import os
from pathlib import Path
import utils as ut


def days_since_epoch():
    epoch_date = time.strptime("2024-08-17", "%Y-%m-%d")
    epoch_seconds = time.mktime(epoch_date)
    # Get the current time
    current_seconds = time.time()
    # Calculate days since the epoch
    days_since_epoch = int((current_seconds - epoch_seconds) // (24 * 3600))
    return days_since_epoch

def log_sim_entry(params):
    params.global_log_file = params.wk_dir / "simulations/simulation_log.json"
    
    # Load existing logs if the log file exists
    if params.global_log_file.exists():
        with open(params.global_log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = {}

    params.new_sim_number = len(logs) + 1
    # Determine simulation number for today prior to this simulation
    params.prior_sims_today = len([log for log in logs.values() if log['date_of_sim'] == params.date_of_sim])
    params.sim_otd = params.prior_sims_today + 1

    # Create the log entry as a dictionary
    log_entry = {
        "days_since_epoch": params.days_since_epoch,
        "sim_otd": params.sim_otd,
        "date_of_sim": params.date_of_sim,
        "time_of_sim": params.time_of_sim,
    }

    # Add the new entry to the logs
    logs[f"{params.new_sim_number}"] = log_entry
    log_dir = params.wk_dir / "simulations"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(params.global_log_file):
        with open(params.global_log_file, 'w'): pass
    # Write the logs back to the JSON file
    with open(params.global_log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    ut.print_message_in_box_plain(f"Simulation #{params.sim_otd} for the day.")
    return params


def initialize_paths(params):
    params.template_sdf_file = params.wk_dir / f"src/template_sdf.py"
    params.sdf_file = params.wk_dir / f"src/sdf.py"

    params.data_dir = params.wk_dir / "simulations" / f"{params.simulation_name}"
    params.data_dir.mkdir(parents=True, exist_ok=True)

    params.vtk_dir = params.data_dir / f"vtks"
    params.vtk_dir.mkdir(parents=True, exist_ok=True)
    
    params.vtk_gauss_dir = params.data_dir / f"vtks_gauss"
    params.vtk_gauss_dir.mkdir(parents=True, exist_ok=True)
    
    params.h5m_dir = params.data_dir / f"h5ms"
    params.h5m_dir.mkdir(parents=True, exist_ok=True)
    
    params.h5m_gauss_dir = params.data_dir / f"h5ms_gauss"
    params.h5m_gauss_dir.mkdir(parents=True, exist_ok=True)
    
    
    params.graph_dir = params.data_dir / f"graphs"
    params.graph_dir.mkdir(parents=True, exist_ok=True)
    
    
    params.med_filepath = params.data_dir / f"{params.mesh_name_appended}.med"
    params.h5m_filepath = params.data_dir / f"{params.mesh_name_appended}.h5m"
    params.finalized_mesh_filepath = params.custom_mesh_filepath if getattr(params, "custom_mesh_filepath", False) else params.h5m_filepath
    

    params.read_med_initial_log_file = params.data_dir / f"{params.mesh_name_appended}_read_med.log"
    params.partition_log_file = params.data_dir / f"{params.mesh_name_appended}_partition.log"
    params.config_file = params.data_dir / "bc.cfg"
    params.log_file = params.data_dir /  f"result_{params.mesh_name_appended}.log"
    params.total_force_log_file = params.data_dir /  f"result_{params.mesh_name_appended}_total_force.log"
    params.FIX_X_1_force_log_file = params.data_dir /  f"result_{params.mesh_name_appended}_FIX_X_1_force.log"
    params.DOFs_log_file = params.data_dir /  f"result_{params.mesh_name_appended}_DOFs.log"
    params.ux_log_file = params.data_dir /  f"result_{params.mesh_name_appended}_ux.log"
    params.snes_log_file = params.data_dir /  f"result_{params.mesh_name_appended}_snes.log"
    
    
    if not os.path.exists(params.log_file):
        with open(params.log_file, 'w'): pass

    params.part_file = os.path.splitext(params.h5m_filepath)[0] + "_" + str(params.nproc) + "p.h5m"
    params.time_history_file = params.data_dir / f"body_force_hist.txt"
    
    params.strain_animation_temp_dir = Path( params.data_dir / "strain_animation_pngs")
    params.strain_animation_temp_dir.mkdir(parents=True, exist_ok=True)
    params.strain_animation_filepath_png =  params.strain_animation_temp_dir / f"{params.mesh_name_appended}.png"
    params.strain_animation_filepath_png_ffmpeg_regex =  params.strain_animation_temp_dir / f"{params.mesh_name_appended}.%04d.png"
    params.strain_animation_filepath_mp4 = params.data_dir /  f"{params.mesh_name_appended}.mp4"
    
    return params

def setup(params):
    if getattr(params, "days_since_epoch", False) and getattr(params, "sim_otd", False) and getattr(params, "FEA_completed", False):
        import re
        # Define the pattern to match the folder name
        pattern = re.compile(rf"{params.case_name}_day_{params.days_since_epoch}_sim_{params.sim_otd}_(\d{{8}}_\d{{6}})_{params.soil_model.name}")
        # Search for the folder in the specified directory
        for folder_name in os.listdir(params.wk_dir / "simulations"):
            match = pattern.match(folder_name)
            if match:
                date_time_str = match.group(1)
                params.time_of_sim = date_time_str
                params.date_of_sim = date_time_str.split('_')[0]
                break
        if not match:
            raise FileNotFoundError(f"No matching folder found for pattern: {pattern.pattern}")
    
    else:
        params.FEA_completed=False
        params.days_since_epoch = days_since_epoch()
        params.time_of_sim = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        params.date_of_sim = time.strftime("%Y%m%d", time.localtime())
        params = log_sim_entry(params)
        
    params.simulation_name = f"{params.case_name}_day_{params.days_since_epoch}_sim_{params.sim_otd}_{params.time_of_sim}_{params.soil_model.name}"
    params.mesh_name_appended = f"{params.case_name}_day_{params.days_since_epoch}_sim_{params.sim_otd}_{params.soil_model.name}"
    params = initialize_paths(params)
    
    return params

def set_display_configurations(): pass
    # import pyvista as pv
    # from matplotlib.colors import ListedColormap
    # pv.set_plot_theme("document")

    # plt.rcParams['figure.figsize'] = [12, 9]
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['font.family'] = "DejaVu Serif"
    # plt.rcParams['font.size'] = 20

    # from pyvirtualdisplay import Display
    # display = Display(backend="xvfb", visible=False, size=(800, 600))
    # display.start()
    
# set_display_configurations()