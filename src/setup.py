import time
import json
import os
from pathlib import Path


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

    # Create the params dictionary for the log entry
    if params.use_case in ["test_2D", "test_3D"]:
        params_dict = {
            "test_volume": params.tester.model_dump(serialize_as_any=True),
            "prescribed_force": params.prescribed_force.model_dump() if getattr(params, 'prescribed_force', None) else None,
            "prescribed_disp": params.prescribed_disp.model_dump() if getattr(params, 'prescribed_disp', None) else None,
        }
    elif params.use_case == "pile_problem":
        params_dict = {
            "mesh": params.mode,
            "pile_manager": params.pile_manager.model_dump(serialize_as_any=True),
            "box_manager": params.box_manager.model_dump(serialize_as_any=True),
            "prescribed_force": params.prescribed_force.model_dump() if getattr(params, 'prescribed_force', None) else None,
            "prescribed_disp": params.prescribed_disp.model_dump() if getattr(params, 'prescribed_disp', None) else None,
        }


    # Filter simulations for today and count simulations with the same parameters
    # params.prior_sims_with_same_params = [log for log in logs.values() if log['params'] == params_dict]
    params.new_sim_number = len(logs) + 1
    # Determine simulation number for today prior to this simulation
    params.prior_sims_today = len([log for log in logs.values() if log['date_of_sim'] == params.date_of_sim])
    params.new_sim_number_today = params.prior_sims_today + 1

    # Create the log entry as a dictionary
    log_entry = {
        "days_since_epoch": params.days_since_epoch,
        "sim_number_of_the_day": params.new_sim_number_today,
        "date_of_sim": params.date_of_sim,
        "time_of_sim": params.time_of_sim,
        "params": params_dict,
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

    print(f"Simulation #{params.new_sim_number_today} for the day.")
    return params

def initialize_paths(params):

    params.simulation_name = f"test_day_{params.days_since_epoch}_sim_{params.new_sim_number_today}_{params.time_of_sim}"
    params.mesh_name = f"test_day_{params.days_since_epoch}_sim_{params.new_sim_number_today}"

    # Continue with the rest of the simulation setup
    params.data_dir = Path(f"/mofem_install/jupyter/thomas/mfront_example_test/simulations/{params.simulation_name}")
    params.data_dir.mkdir(parents=True, exist_ok=True)

    params.template_sdf_file = params.wk_dir / f"src/template_sdf.py"
    params.sdf_file = params.wk_dir / f"src/sdf.py"


    params.med_filepath = params.data_dir / f"{params.mesh_name}.med"
    params.h5m_filepath = params.data_dir / f"{params.mesh_name}.h5m"
    params.vtk_filepath = params.data_dir / f"{params.mesh_name}.vtk"
    params.csv_filepath = params.data_dir / f"{params.mesh_name}.csv"
    params.part_file = os.path.splitext(params.h5m_filepath)[0] + "_" + str(params.nproc) + "p.h5m"
    params.time_history_file = params.data_dir / f"disp_history.txt"

    params.read_med_initial_log_file = params.data_dir / f"{params.mesh_name}_read_med.log"
    params.config_file = params.data_dir / "bc.cfg"
    params.log_file = params.data_dir /  f"result_{params.mesh_name}.log"
    if not os.path.exists(params.log_file):
        with open(params.log_file, 'w'): pass

    return params

def setup(params):
    params.time_history = False # always set as False initilaly, checked at the stage of writing the config files
    params.days_since_epoch = days_since_epoch()
    params.time_of_sim = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    params.date_of_sim = time.strftime("%Y_%m_%d", time.localtime())
    params = log_sim_entry(params)
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