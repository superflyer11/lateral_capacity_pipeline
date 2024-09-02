# 1. Set Up

```python
!spack find -p tfel
```

```python
# %env LD_LIBRARY_PATH=/mofem_install/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-9.4.0/tfel-4.0.0-mvfpqw7u4c23su7hj7g4leuwmykrjmcx/lib
```

```python
!echo "$(spack find -p tfel | awk '/\/mofem_install\// {print $NF "/lib"}')"
```

```python
import math
import os
import re
import sys
import time
import json
from pathlib import Path
import subprocess
import zipfile
import pydantic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import gmsh
from scipy import optimize
from scipy.optimize import curve_fit, least_squares

sys.path.append('/mofem_install/jupyter/thomas/mfront_example_test/src')

import mesh_create as mshcrte
import custom_models as cm
import utils as ut
import plotting


def set_display_configurations(): pass
    # from matplotlib.colors import ListedColormap
    # pv.set_plot_theme("document")

    # plt.rcParams['figure.figsize'] = [12, 9]
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['font.family'] = "DejaVu Serif"
    # plt.rcParams['font.size'] = 20

    # from pyvirtualdisplay import Display
    # display = Display(backend="xvfb", visible=False, size=(800, 600))
    # display.start()
    
set_display_configurations()
os.chdir('/mofem_install/jupyter/thomas/mfront_example_test')
    

```

# 2. Simulation Parameters

```python
def initialize_parameters():
    params = mshcrte.AttrDict()
    params.pile_manager = cm.PileManager(x=0, y=0, z=10, dx=0, dy=0, dz=-20.5, R=1, r=0.975,
                                preferred_model= cm.PropertyTypeEnum.elastic,
                                props = {
                                    cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=200000*(10**6), poisson_ratio=0.3),
                                    cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=200000*10**6, poisson_ratio=0.3),
                                },
                                )

    soil_layer_1 = cm.SoilLayer(
        depth = -2,
        preferred_model= cm.PropertyTypeEnum.elastic,
        props = {
            cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=96*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.isotropic_hardening: cm.IsotropicLinearHardeningPlasticityProperties(youngs_modulus=96*10**6, poisson_ratio=0.3,HardeningSlope = 10.e9, YieldStress = 300.e6),
            cm.PropertyTypeEnum.drucker_prager: cm.ElasticProperties(youngs_modulus=96*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.cam_clay: cm.CamClayProperties(),
            }, 
            )
    soil_layer_2 = cm.SoilLayer(
        depth = -1.4,
        preferred_model= cm.PropertyTypeEnum.elastic,
        props = {
            cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.isotropic_hardening: cm.IsotropicLinearHardeningPlasticityProperties(youngs_modulus=96*10**6, poisson_ratio=0.3,HardeningSlope = 10.e9, YieldStress = 300.e6),
            cm.PropertyTypeEnum.drucker_prager: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.cam_clay: cm.CamClayProperties(),
            }, 
            )
    soil_layer_3 = cm.SoilLayer(
        depth = -7.1,
        preferred_model= cm.PropertyTypeEnum.elastic,
        props = {
            cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=351.3*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.isotropic_hardening: cm.IsotropicLinearHardeningPlasticityProperties(youngs_modulus=96*10**6, poisson_ratio=0.3,HardeningSlope = 10.e9, YieldStress = 300.e6),
            cm.PropertyTypeEnum.drucker_prager: cm.ElasticProperties(youngs_modulus=351.3*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.cam_clay: cm.CamClayProperties(),
                 }, 
            )
    soil_layer_4 = cm.SoilLayer(
        depth = -19,
        preferred_model= cm.PropertyTypeEnum.elastic,
        props = {
            cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=668.4*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.isotropic_hardening: cm.IsotropicLinearHardeningPlasticityProperties(youngs_modulus=96*10**6, poisson_ratio=0.3,HardeningSlope = 10.e9, YieldStress = 300.e6),
            cm.PropertyTypeEnum.drucker_prager: cm.ElasticProperties(youngs_modulus=668.4*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.cam_clay: cm.CamClayProperties(),
            }, 
            )
    params.interface_manager = cm.InterfaceManager(
        preferred_model = cm.PropertyTypeEnum.elastic,
        props = {
            cm.PropertyTypeEnum.elastic: cm.ElasticProperties(youngs_modulus=96*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.saint_venant_kirchhoff: cm.ElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.isotropic_hardening: cm.IsotropicLinearHardeningPlasticityProperties(youngs_modulus=96*10**6, poisson_ratio=0.3,HardeningSlope = 10.e9, YieldStress = 300.e6),
            cm.PropertyTypeEnum.drucker_prager: cm.ElasticProperties(youngs_modulus=96*10**6, poisson_ratio=0.3),
            cm.PropertyTypeEnum.cam_clay: cm.CamClayProperties(),
            },
            )
    # params.prescribed_force = cm.ForceBoundaryCondition(fx=-18*10**6,fy=0,fz=0)
    params.prescribed_disp = cm.SurfaceBoundaryCondition(disp_ux=-1)
    params.box_manager = cm.BoxManager(x=-80, y=-80, z=0, dx=160, dy=80,
        layers=[
        soil_layer_1,
        soil_layer_2,
        soil_layer_3,
        soil_layer_4,
        ],
        far_field_size=10,
        near_field_dist=5,
        near_field_size=0.5,
    )

    params.nproc = 8 # number of processors/cores used
    params.order = 2 #order of approximation functions

    params.time_step = 0.05 # [s]
    params.final_time = 3 # [s]

    return params
```

```python
params = initialize_parameters()
```

# 3. Log paths and meta

```python
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
    params_dict = {
        "pile_manager": params.pile_manager.model_dump(serialize_as_any=True),
        "box_manager": params.box_manager.model_dump(serialize_as_any=True),
        "prescribed_force": params.prescribed_force.model_dump() if getattr(params, 'prescribed_force', None) else None,
        "prescribed_disp": params.prescribed_disp.model_dump() if getattr(params, 'prescribed_disp', None) else None,
    }

    # Filter simulations for today and count simulations with the same parameters
    params.prior_sims_with_same_params = [log for log in logs.values() if log['params'] == params_dict]
    # params.prior_sims_with_same_params_no = len(params.prior_sims_with_same_params)
    # params.new_sim_number_with_same_params = params.prior_sims_with_same_params_no + 1
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
        # "sim_number_with_same_params": params.new_sim_number_with_same_params,
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
    # print(f"Simulation #{params.new_sim_number_with_same_params} with the same parameters.")

    # if params.prior_sims_with_same_params_no > 0:
    #     # Get the datetime of the previous simulation
    #     previous_simulation = params.prior_sims_with_same_params[-1]
    #     print(f"Previous simulation with the same parameters was run on: day {previous_simulation['days_since_epoch']} simulation {previous_simulation['sim_number_today']}")
    return params

def initialize_paths(params):

    params.simulation_name = f"day_{params.days_since_epoch}_sim_{params.new_sim_number_today}_{params.time_of_sim}"
    params.mesh_name = f"day_{params.days_since_epoch}_sim_{params.new_sim_number_today}"

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

    params.read_med_initial_log_file = params.data_dir / f"{params.mesh_name}_read_med.log"

    # params.bc_time_history = params.data_dir / "disp_time.txt"
    params.config_file = params.data_dir / "bc.cfg"
    params.log_file = params.data_dir /  f"result_{params.mesh_name}.log"
    if not os.path.exists(params.log_file):
        with open(params.log_file, 'w'): pass

    return params



# sys.exit()
```

```python
params.days_since_epoch = days_since_epoch()
    
params.time_of_sim = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
params.date_of_sim = time.strftime("%Y_%m_%d", time.localtime())

# File paths and directories
params.wk_dir = Path(f"/mofem_install/jupyter/thomas/mfront_example_test")
params = log_sim_entry(params)
# print(params.pile_manager)
# print(params.pile_manager.model_dump(serialize_as_any=True))
# print(params.pile_manager.model_dump(serialize_as_any=False))
# sys.exit()
params = initialize_paths(params)
params.user_name = !whoami
params.user_name = params.user_name[0]
params.um_view = f"/mofem_install/jupyter/{params.user_name}/um_view"
```

# 4. Generate the mesh

```python
geo = mshcrte.draw_mesh(params)
params.physical_groups = mshcrte.add_physical_groups(params, geo)
params.physical_groups = mshcrte.check_block_ids(params,params.physical_groups)
params.physical_groups = mshcrte.generate_config(params,params.physical_groups)
mshcrte.inject_configs(params)
mshcrte.partition_mesh(params)
```

# 5. Running the analysis and export to .vtk file format

```python
mshcrte.mofem_compute(params) #calling the func from the module
mshcrte.export_to_vtk(params)
```

# 6. Extract data from .vtk file with pvpython, then plotting

```python
!/mofem_install/jupyter/thomas/ParaView-5.13.0-RC1-MPI-Linux-Python3.10-x86_64/bin/pvpython /mofem_install/jupyter/thomas/mfront_example_test/src/paraview_test.py {params.vtk_filepath} {params.data_dir}
```

```python
plotting.plot_displacement_vs_points(
    {"Compression at x = 1m": f'{params.data_dir}/dis_to_depth_compression_x_1.csv', 
     "Compression at x = 1.1m": f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv'}, 
    f"{params.data_dir}/result_disp.png")


plotting.plot_stress_vs_points(
    {"Compression at x = 1m": f'{params.data_dir}/dis_to_depth_compression_x_1.csv', 
     "Compression at x = 1.1m": f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv'}, 
    f"{params.data_dir}/result_stress.png")

plotting.calculate_and_plot_q_p(
    {"Compression at x = 1m": f'{params.data_dir}/dis_to_depth_compression_x_1.csv', 
     "Compression at x = 1.1m": f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv'}, 
    f"{params.data_dir}/result_q_p.png")



```

```python
# raise SystemExit("")
```
