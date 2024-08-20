# Set Up


### Import required Python libraries and set plotting parameters 


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
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time
import os
import os.path
import zipfile
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import sys
import gmsh
import math
import pyvista as pv
import re

from matplotlib.colors import ListedColormap
pv.set_plot_theme("document")

plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = "DejaVu Serif"
plt.rcParams['font.size'] = 20

from pyvirtualdisplay import Display
display = Display(backend="xvfb", visible=False, size=(800, 600))
display.start()
    

#um_view = "/mofem_install/jupyter/callum/mofem_install/mofem-cephas/mofem/users_modules/um-build-Release-7q7t4mo"
```

# Simulation Parameters

```python
import sys

sys.path.append('/mofem_install/jupyter/thomas/mfront_example_test/src')
import mesh_create as mshcrte
import custom_models as cm

os.chdir('/mofem_install/jupyter/thomas/mfront_example_test')
mshcrte.test()
params = mshcrte.AttrDict()
# young modulus in Pa
params.pile_manager = cm.PileManager(x=0, y=0, z=10, dx=0, dy=0, dz=-20.5, R=1, r=0.975,
                              linear_elastic_properties=cm.LinearElasticProperties(youngs_modulus=200000*(10**6), poisson_ratio=0.3)
                              )

soil_layer_1 = cm.SoilLayer(
    depth = -2,
    linear_elastic_properties=cm.LinearElasticProperties(youngs_modulus=96*10**6, poisson_ratio=0.499),
    # cam_cl
    )
soil_layer_2 = cm.SoilLayer(
    depth = -1.4,
    linear_elastic_properties=cm.LinearElasticProperties(youngs_modulus=182.1*10**6, poisson_ratio=0.499),
    )
soil_layer_3 = cm.SoilLayer(
    depth = -7.1,
    linear_elastic_properties=cm.LinearElasticProperties(youngs_modulus=351.3*10**6, poisson_ratio=0.499),
    )
soil_layer_4 = cm.SoilLayer(
    depth = -29.5,
    linear_elastic_properties=cm.LinearElasticProperties(youngs_modulus=668.4*10**6, poisson_ratio=0.499),
    )

# params.prescribed_force = cm.ForceBoundaryCondition(fx=-18*10**6,fy=0,fz=0)
params.prescribed_disp = cm.SurfaceBoundaryCondition(disp_ux=-1)
params.box_manager = cm.BoxManager(x=-80, y=-80, z=0, dx=160, dy=80,
    layers={
        1: soil_layer_1,
        2: soil_layer_2,
        3: soil_layer_3,
        4: soil_layer_4,
    },
    far_field_size=10,
    near_field_dist=5,
    near_field_size=0.5,
)



params.nproc = 8 # number of processors/cores used
params.order = 2 #order of approximation functions

params.final_time = 1 # [s]
params.time_step = 0.5 # [s]

params.material_model = "LinearElasticity"
if params.material_model =="LinearElasticity":
    params.mi_block = cm.PropertyTypeEnum.elastic


```

# Log paths and meta

```python
from pathlib import Path
import time

# Set the epoch date

epoch_date = time.strptime("2024-08-17", "%Y-%m-%d")
epoch_seconds = time.mktime(epoch_date)
# Get the current time
current_seconds = time.time()
# Calculate days since the epoch
days_since_epoch = int((current_seconds - epoch_seconds) // (24 * 3600))


now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
today_str = time.strftime("%Y_%m_%d", time.localtime())

# File paths and directories
params.wk_dir = Path(f"/mofem_install/jupyter/thomas/mfront_example_test")
params.log_file = params.wk_dir / "simulations/simulation_log.txt"

# Read log file to count simulations for today
if not params.log_file.exists():
    logs = []
else:
    with open(params.log_file, 'r') as f:
        logs = f.readlines()



# Filter simulations for today
# Count simulations with the same parameters
if getattr(params, 'prescribed_force', None):
    param_string = f"{params.pile_manager} {params.box_manager} {params.prescribed_force.fx} {params.prescribed_force.fy} {params.prescribed_force.fz} 0 0 0"
elif getattr(params, 'prescribed_disp', None):
    param_string = f"{params.pile_manager} {params.box_manager} 0 0 0 {params.prescribed_disp.disp_ux} {params.prescribed_disp.disp_uy} {params.prescribed_disp.disp_uz}"
else:
    param_string = f"{params.pile_manager} {params.box_manager} 0 0 0 0 0 0"

prior_sims_with_same_params = [log for log in logs if param_string in log]
prior_sims_with_same_params_no = len(prior_sims_with_same_params)
new_sim_number_with_same_params = prior_sims_with_same_params_no + 1

# Determine simulation number for today prior to this simulation
prior_sims_today = len([log for log in logs if today_str in log])
new_sim_number_today = prior_sims_today + 1
# Log the simulation
log_entry = f"{days_since_epoch} {new_sim_number_today} {now} {param_string}\n"

if not params.log_file.exists():
    with open(params.log_file, 'w') as f:
        f.write(log_entry)
else:
    with open(params.log_file, 'a') as f:
        f.write(log_entry)



print(f"Simulation #{new_sim_number_today} for the day.")
print(f"Simulation #{new_sim_number_with_same_params} with the same parameters.")

if len(prior_sims_with_same_params) > 1:
    # Get the datetime of the previous simulation
    previous_simulation = prior_sims_with_same_params[-1].split()
    previous_simulation_day = previous_simulation[0]
    previous_simulation_number = previous_simulation[1]
    print(f"Previous simulation with the same parameters was run on: day {previous_simulation_day} simulation {previous_simulation_number}")
else:
    pass

params.simulation_name = f"day_{days_since_epoch}_sim_{new_sim_number_today}_{now}"
params.mesh_name = params.simulation_name

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
    
user_name=!whoami # get user name
user_name=user_name[0]
params.um_view = "/mofem_install/jupyter/%s/um_view" % user_name

# sys.exit()
```

```python
if not os.path.exists(params.log_file):
    with open(params.log_file, 'w'): pass
    
user_name=!whoami # get user name
user_name=user_name[0]
params.um_view = "/mofem_install/jupyter/%s/um_view" % user_name

# sys.exit()
```

# Generate the mesh

```python
geo = mshcrte.draw_mesh(params)
physical_groups = mshcrte.add_physical_groups(params, geo)
physical_groups = mshcrte.check_block_ids(params,physical_groups)
physical_groups = mshcrte.generate_config(params,physical_groups)
mshcrte.inject_configs(params)
mshcrte.partition_mesh(params)
```

# CUSTOM

```python
def replace_template_sdf(params):
    regex = r"\{(.*?)\}"
    # print(os.getcwd())
    with open(params.template_sdf_file) as infile, open(params.sdf_file, 'w') as outfile:
        for line in infile:
            matches = re.finditer(regex, line, re.DOTALL)
            for match in matches:
                for name in match.groups():
                    src = "{" + name + "}"
                    target = str(1) #1 is a placeholder because it is not being used
                    line = line.replace(src, target)
            outfile.write(line)

def mofem_compute_force_indent(params):
    !rm -rf out*
    replace_template_sdf(params)
    
    mfront_arguments = []
    for physical_group in physical_groups:
        if physical_group.name.startswith("MFRONT_MAT"):
            mfront_block_id = physical_group.meshnet_id
            mi_block = "LinearElasticity"
            mi_param_0 = physical_group.props[cm.PropertyTypeEnum.elastic].youngs_modulus
            mi_param_1 = physical_group.props[cm.PropertyTypeEnum.elastic].poisson_ratio
            mi_param_2 = 0
            mi_param_3 = 0
            mi_param_4 = 0
            
            mfront_arguments.append(
                f"-mi_lib_path_{mfront_block_id} {params.um_view}/mfront_interface/libBehaviour.so "
                f"-mi_block_{mfront_block_id} {mi_block} "
                f"-mi_param_{mfront_block_id}_0 {mi_param_0} "
                f"-mi_param_{mfront_block_id}_1 {mi_param_1} "
                f"-mi_param_{mfront_block_id}_2 {mi_param_2} "
                f"-mi_param_{mfront_block_id}_3 {mi_param_3} "
                f"-mi_param_{mfront_block_id}_4 {mi_param_4} "
            )
    
    # Join mfront_arguments list into a single string
    mfront_arguments_str = ' '.join(mfront_arguments)


    command = (
        f"export OMPI_MCA_btl_vader_single_copy_mechanism=none && "
        f"nice -n 10 mpirun --oversubscribe --allow-run-as-root "
        f"-np {params.nproc} {params.um_view}/tutorials/adv-1/contact_3d "
        f"-file_name {params.part_file} "
        f"-sdf_file {params.sdf_file} "
        f"-order {params.order} "
        f"-contact_order 0 "
        f"-sigma_order 0 " #play around this in the future?
        f"-ts_dt {params.time_step} "
        f"-ts_max_time {params.final_time} "
        f"{mfront_arguments_str} "
        f"-mi_save_volume 1 "
        f"-mi_save_gauss 0 "
        f"2>&1 | tee {params.log_file}"
    )

    import subprocess
    result = subprocess.run(command, shell=True, text=True)
    

def export_to_vtk(params):
    out_to_vtk = !ls -c1 out_*h5m
    print(out_to_vtk)
    last_file=out_to_vtk[0]
    print(last_file)
    !mbconvert {last_file} {params.vtk_filepath}

```

# run the analysis and expor to .vtk file

```python
mofem_compute_force_indent(params)
export_to_vtk(params)
```

# extract data from .vtk file with pvpython

```python
!/mofem_install/jupyter/thomas/ParaView-5.13.0-RC1-MPI-Linux-Python3.10-x86_64/bin/pvpython /mofem_install/jupyter/thomas/mfront_example_test/src/paraview_test.py {params.vtk_filepath} {params.data_dir}
```

```python
import plotting

plotting.plot_displacement_vs_points(f'{params.data_dir}/dis_to_depth_compression_x_1.csv', f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv', f"{params.data_dir}/result_disp.png")

plotting.plot_stress_vs_points(f'{params.data_dir}/dis_to_depth_compression_x_1.csv', f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv', f"{params.data_dir}/result_stress.png")

quit
```

```python
print(indent_list)
print(force_list)
for elem_num, indent, force in zip(elem_num_list, indent_list, force_list):
    plt.plot(indent / params.indenter_radius, force, marker='o', ms=4, lw=1.5, label="MoFEM: {} elements per side".format(elem_num))
    
# plt.plot(indent/params.indenter_radius, hertz_press(indent, params), c='k', ls='--', label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Force, nN")
plt.legend(loc='upper left')
plt.grid(ls=":")
```

# End of custom


### Define utility functions including black-box launch of MoFEM


```python
import gmsh
class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value
        
def replace_template_sdf(params):
    regex = r"\{(.*?)\}"
    # print(os.getcwd())
    with open(params.template_sdf_file) as infile, open(params.sdf_file, 'w') as outfile:
        for line in infile:
            matches = re.finditer(regex, line, re.DOTALL)
            for match in matches:
                for name in match.groups():
                    src = "{" + name + "}"
                    target = str(params[name])
                    line = line.replace(src, target)
            outfile.write(line)

def get_young_modulus(K, G):
    E = 9. * K * G /(3. * K + G)
    return E

def get_poisson_ratio(K, G):
    nu = (3. * K - 2. * G) / 2. / (3. * K + G)
    return nu

def get_bulk_modulus(E, nu):
    K = E / 3. / (1. - 2. * nu)
    return K

def get_shear_modulus(E, nu):
    G = E / 2. / (1. + nu)
    return G

def parse_log_file(filepath):
    force, time, area = [], [], []
    with open(filepath, "r") as log_file:
        for line in log_file:
            line = line.strip()
            if "Contact force:" in line:
                line = line.split()
                time.append(float(line[6]))
                force.append(float(line[10]))
            if "Contact area:" in line:
                line = line.split()
                area.append(float(line[8]))
    return time, force, area

def generate_config(params):
    with open(params.config_file, 'w') as f:
        data = [f"[block_2]", f"id={params.mfront_block_id}", "add=BLOCKSET", f"name=MFRONT_MAT_{params.mfront_block_id}"]
        for line in data:
            f.write(line + '\n')
    return

def mofem_compute_force_indent(params):
    !rm -rf out*
    
    mi_param_2 = 0
    mi_param_3 = 0
    mi_param_4 = 0
    
    if params.material_model == "LinearElasticity":
        mi_block = "LinearElasticity"
        mi_param_0 = params.young_modulus
        mi_param_1 = params.poisson_ratio
    elif params.material_model == "SaintVenantKirchhoffElasticity":
        mi_block = "SaintVenantKirchhoffElasticity"
        mi_param_0 = params.young_modulus
        mi_param_1 = params.poisson_ratio
    elif params.material_model == "NeoHookeanHyperElasticity":
        mi_block = "SignoriniHyperElasticity"
        mi_param_0 = get_bulk_modulus(params.young_modulus, params.poisson_ratio)
        mi_param_1 = 0.5 * get_shear_modulus(params.young_modulus, params.poisson_ratio)
    elif params.material_model == "StandardLinearSolid":
        mi_block = "StandardLinearSolid"
        mi_param_0 = get_bulk_modulus(params.young_modulus_0, params.poisson_ratio_0)
        mi_param_1 = get_shear_modulus(params.young_modulus_0, params.poisson_ratio_0)
        mi_param_2 = get_bulk_modulus(params.young_modulus_1, params.poisson_ratio_1)
        mi_param_3 = get_shear_modulus(params.young_modulus_1, params.poisson_ratio_1)
        mi_param_4 = params.relax_time_1
    else:
        print("Unknown material model: " + params.material_model)
        return
        
    replace_template_sdf(params)
        
    !export OMPI_MCA_btl_vader_single_copy_mechanism=none && \
    nice -n 10 mpirun --oversubscribe --allow-run-as-root \
    -np {params.nproc} {um_view}/tutorials/adv-1/contact_2d \
    -file_name {params.part_file} \
    -sdf_file {params.sdf_file} \
    -order {params.order} \
    -ts_dt {params.time_step} \
    -ts_max_time {params.final_time} \
    -mi_lib_path_{params.mfront_block_id} {um_view}/mfront_interface/libBehaviour.so \
    -mi_block_{params.mfront_block_id} {mi_block} \
    -mi_param_{params.mfront_block_id}_0 {mi_param_0} \
    -mi_param_{params.mfront_block_id}_1 {mi_param_1} \
    -mi_param_{params.mfront_block_id}_2 {mi_param_2} \
    -mi_param_{params.mfront_block_id}_3 {mi_param_3} \
    -mi_param_{params.mfront_block_id}_4 {mi_param_4} \
    -mi_save_volume 1 \
    -mi_save_gauss 0 \
    2>&1 | tee {params.log_file}

    time, react, area = parse_log_file(params.log_file)
    indent = np.asarray(time) * (params.max_indentation / params.final_time)
    force = np.asarray(react)
    
    return indent, force, area

def show_results(params):
    out_to_vtk = !ls -c1 out_*h5m
    print(out_to_vtk)
    last_file=out_to_vtk[0]
    print(last_file)
    !mbconvert {last_file} {last_file[:-3]}vtk
    
    import pyvista as pv
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.image as mpimg
    import re, os

    mesh = pv.read(last_file[:-3] + "vtk")

    mesh=mesh.warp_by_vector('DISPLACEMENT', factor=1)
    if params.show_edges:
        mesh=mesh.shrink(0.95)
    
    if params.show_field == "DISPLACEMENT" or params.show_field == "displacement":
        field = "DISPLACEMENT"
        if params.show_component == "X" or params.show_component == 'x':
            comp = 0
        elif params.show_component == "Y" or params.show_component == 'y':
            comp = 1
        else:
            print("Wrong component {0} of the field {1}".format(params.show_component, params.show_field))
            return
        
    if params.show_field == "STRESS" or params.show_field == "stress":
        field = "STRESS"
        if params.show_component == "X" or params.show_component == "x":
            comp = 0
        elif params.show_component == "Y" or params.show_component == "y":
            comp = 4
        elif params.show_component == "XY" or params.show_component == "xy":
            comp = 1
        else:
            print("Wrong component {0} of the field {1}".format(params.show_component, params.show_field))
            return

    p = pv.Plotter(notebook=True)
    p.add_mesh(mesh, scalars=field, component=comp, show_edges=True, smooth_shading=False, cmap="turbo")
    
    # circle = pv.Circle(radius=params.indenter_radius, resolution=1000)
    # circle = circle.translate((0, params.indenter_radius - params.max_indentation, 0), inplace=False)
    # p.add_mesh(circle, color="grey")
    
    p.camera_position = "xy"
    p.show(jupyter_backend='ipygany')


def generate_mesh(params):
    gmsh.initialize()
    gmsh.model.add("Nanoindentation")
    
    a = params.refine_radius    
    H = params.mesh_height 
    L = params.mesh_length
    R = params.indenter_radius
    
    # Creating points
    tol = 1e-3
    
    print(a, H, R)
    new_model = False
    if new_model:
        point1 = gmsh.model.geo.addPoint(0, 0, 0, tol)
        
    
    
    if a < H / 2 and H > R:
        point1 = gmsh.model.geo.addPoint(0, 0, 0, tol)
        point2 = gmsh.model.geo.addPoint(0, -a, 0, tol)
        point3 = gmsh.model.geo.addPoint(a, 0, 0, tol)
        point4 = gmsh.model.geo.addPoint(0, -H, 0, tol)
        point5 = gmsh.model.geo.addPoint(L, -H, 0, tol)
        point6 = gmsh.model.geo.addPoint(L, 0, 0, tol)

        # Creating connection lines
        arc1 = gmsh.model.geo.addCircleArc(point3, point1, point2)
        line1 = gmsh.model.geo.addLine(point1, point2)
        line2 = gmsh.model.geo.addLine(point2, point4)
        line3 = gmsh.model.geo.addLine(point4, point5)
        line4 = gmsh.model.geo.addLine(point5, point6)
        line5 = gmsh.model.geo.addLine(point6, point3)
        line6 = gmsh.model.geo.addLine(point3, point1)

        loop1 = gmsh.model.geo.addCurveLoop([line1, -arc1, line6])
        surface1 = gmsh.model.geo.addPlaneSurface([loop1])

        loop2 = gmsh.model.geo.addCurveLoop([arc1, line2, line3, line4, line5])
        surface2 = gmsh.model.geo.addPlaneSurface([loop2])

        # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
        gmsh.model.geo.synchronize()

        domain = gmsh.model.addPhysicalGroup(2, [surface1, surface2])
        gmsh.model.setPhysicalName(2, domain, '!_DOMAIN')
        contact = gmsh.model.addPhysicalGroup(1, [line5, line6])
        gmsh.model.setPhysicalName(1, contact, 'CONTACT')
        fix_x = gmsh.model.addPhysicalGroup(1, [line1, line2])
        gmsh.model.setPhysicalName(1, fix_x, 'FIX_X')
        fix_y = gmsh.model.addPhysicalGroup(1, [line3])
        gmsh.model.setPhysicalName(1, fix_y, 'FIX_Y')
        gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(0, -H, 0, L, 0, 0), params.far_field_size)
        gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(0, -a, 0, a, 0, 0), params.near_field_size)
    else:
        point1 = gmsh.model.geo.addPoint(0, 0, 0, tol)
        point2 = gmsh.model.geo.addPoint(0, -H, 0, tol)
        point3 = gmsh.model.geo.addPoint(H, -H, 0, tol)
        point4 = gmsh.model.geo.addPoint(H, 0, 0, tol)
        point5 = gmsh.model.geo.addPoint(L, -H, 0, tol)
        point6 = gmsh.model.geo.addPoint(L, 0, 0, tol)

        # Creating connection lines
        line1 = gmsh.model.geo.addLine(point1, point2)
        line2 = gmsh.model.geo.addLine(point2, point3)
        line3 = gmsh.model.geo.addLine(point3, point4)
        line4 = gmsh.model.geo.addLine(point4, point1)
        line5 = gmsh.model.geo.addLine(point3, point5)
        line6 = gmsh.model.geo.addLine(point5, point6)
        line7 = gmsh.model.geo.addLine(point6, point4)

        loop1 = gmsh.model.geo.addCurveLoop([line1, line2, line3, line4])
        surface1 = gmsh.model.geo.addPlaneSurface([loop1])

        loop2 = gmsh.model.geo.addCurveLoop([-line3, line5, line6, line7])
        surface2 = gmsh.model.geo.addPlaneSurface([loop2])

        # This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
        gmsh.model.geo.synchronize()

        domain = gmsh.model.addPhysicalGroup(2, [surface1, surface2])
        gmsh.model.setPhysicalName(2, domain, '!_DOMAIN')
        contact = gmsh.model.addPhysicalGroup(1, [line7, line4])
        gmsh.model.setPhysicalName(1, contact, 'CONTACT')
        fix_x = gmsh.model.addPhysicalGroup(1, [line1])
        gmsh.model.setPhysicalName(1, fix_x, 'FIX_X')
        fix_y = gmsh.model.addPhysicalGroup(1, [line2, line5])
        gmsh.model.setPhysicalName(1, fix_y, 'FIX_Y1')
        
        gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(0, -H, 0, L, 0, 0), params.far_field_size)
        gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(0, -H, 0, H, 0, 0), params.near_field_size)
        
    gmsh.model.mesh.generate(2)    

    # Save mesh
    gmsh.write(params.med_file)

    # Finalize GMSH = END OF CODE=
    gmsh.finalize()
    
    generate_config(params)
    
    !read_med -med_file {params.med_file} -output_file {params.mesh_file} -meshsets_config {params.config_file} -log_sl inform
    
    params.part_file = os.path.splitext(params.mesh_file)[0] + "_" + str(params.nproc) + "p.h5m"
    #partition the mesh into nproc parts
    !{um_view}/bin/mofem_part \
    -my_file {params.mesh_file} \
    -my_nparts {params.nproc} \
    -output_file {params.part_file} \
    -dim 2 -adj_dim 1
    
    if params.show_mesh:
        !mbconvert {params.mesh_file} {params.vtk_file}

        mesh = pv.read(params.vtk_file )
        mesh = mesh.shrink(0.95)

        p = pv.Plotter(notebook=True)
        p.add_mesh(mesh, smooth_shading=False)

        circle = pv.Circle(radius=params.indenter_radius, resolution=1000)
        circle = circle.translate((0, params.indenter_radius, 0), inplace=False)

        p.add_mesh(circle, color="grey")
        p.camera_position = "xy"
        p.show(jupyter_backend='ipygany')
    
    return

def hertz_press(indent, params):   
    Es = params.young_modulus / (1 - params.poisson_ratio**2)    
    return 4./3. * Es * np.sqrt(params.indenter_radius) * pow(indent, 3./2.)

def hertz_area(indent, params):   
    return np.pi * indent * params.indenter_radius

```

### Sketch of the problem setup


<!-- ![indent.png](attachment:indent.png) -->

<div>
<img src="attachment:indent.png" width="400px"/>
</div>


### Set simulation parameters

```python
import numpy as np
params = AttrDict()

params.med_file = "mesh_2d.med"
params.mesh_file = "mesh_2d.h5m"
params.vtk_file = "mesh_2d.vtk"

params.load_hist = "load.txt"
params.log_file = "log_indent"
# wd = f"/mofem_install/jupyter/{user_name}/mfront_example_test"
# os.chdir(wd)
# params.template_sdf_file = wd + "/" + "template_sdf.py"
# params.sdf_file = wd + "/" + "sdf.py"

params.config_file = "bc.cfg"
params.mfront_block_id = 10

params.nproc = 8 # number of processors/cores used
params.order = 2 #order of approximation functions

params.final_time = 1 # [s]
params.time_step = 0.05 # [s]

params.indenter_radius = 10 
params.max_indentation = 1
params.refine_radius = np.sqrt(params.indenter_radius * params.max_indentation) # a_hertz = sqrt(R * d)

params.mesh_length = params.refine_radius * 50 # L
params.mesh_height = params.mesh_length * 2    # H

params.far_field_size = params.mesh_height / 10
params.near_field_size = params.refine_radius / 10
```

### Generate and visualise the mesh

```python
params.show_mesh = True
generate_mesh(params)
```

### Convergence w.r.t. number of elements per side

```python
# elem_num_list = [2, 5, 10, 20]
elem_num_list = [20]
indent_list = []
force_list = []

params.young_modulus = 100
params.poisson_ratio = 0.45
params.material_model = "LinearElasticity"
params.show_mesh = False

for elem_num in elem_num_list:
    params.far_field_size = params.mesh_height / elem_num
    params.near_field_size = params.refine_radius / elem_num
    generate_mesh(params)
    
    indent, force, area = mofem_compute_force_indent(params)
    indent_list.append(indent)
    force_list.append(force)
```

```python
for elem_num, indent, force in zip(elem_num_list, indent_list, force_list):
    plt.plot(indent / params.indenter_radius, force, marker='o', ms=4, lw=1.5, label="MoFEM: {} elements per side".format(elem_num))
    
plt.plot(indent/params.indenter_radius, hertz_press(indent, params), c='k', ls='--', label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Force, nN")
plt.legend(loc='upper left')
plt.grid(ls=":")
```

### Convergence w.r.t. mesh height
```python
length_mult_list = [5, 10, 20, 40]
indent_list = []
force_list = []

params.young_modulus = 100
params.poisson_ratio = 0.45
params.material_model = "LinearElasticity"

params.show_mesh = False

for length_mult in length_mult_list:
    params.mesh_length = params.refine_radius * length_mult
    params.mesh_height = params.mesh_length * 2
    
    params.far_field_size = params.mesh_height / 10
    params.near_field_size = params.refine_radius / 10

    generate_mesh(params)
    
    indent, force, area = mofem_compute_force_indent(params)
    indent_list.append(indent)
    force_list.append(force)
```

```python
for length_mult, indent, force in zip(length_mult_list, indent_list, force_list):
    plt.plot(indent / params.indenter_radius, force, marker='o', ms=4, lw=1.5, label="MoFEM: mesh length mult {}".format(length_mult))
    
plt.plot(indent/params.indenter_radius, hertz_press(indent, params), c='k', ls ='--', label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Force, nN")
plt.legend(loc='upper left')
plt.grid(ls=":")
```
# Comparison of different elastic models

```python
params.far_field_size = params.mesh_height / 10
params.near_field_size = params.refine_radius / 10

params.mesh_length = params.refine_radius * 40
params.mesh_height = params.mesh_length * 2

params.indenter_radius = 10 
params.max_indentation = 10 

params.refine_radius = np.sqrt(params.indenter_radius * params.max_indentation) # a_hertz = sqrt(R * d)

params.mesh_length = params.refine_radius * 40
params.mesh_height = params.mesh_length * 2

params.far_field_size = params.mesh_height / 10
params.near_field_size = params.refine_radius / 10

params.show_mesh = True
generate_mesh(params)
```

```python
params.young_modulus = 100
params.poisson_ratio = 0.49
params.material_model = "LinearElasticity"

indent_1, force_1, area_1 = mofem_compute_force_indent(params)
```
```python
params.young_modulus = 100
params.poisson_ratio = 0.49
params.material_model = "NeoHookeanHyperElasticity"

indent_2, force_2, area_2 = mofem_compute_force_indent(params)
```

### Comparison of the force evolution

```python
plt.plot(indent_1/params.indenter_radius, force_1, marker='o', ms=6, label="MoFEM LinearElasticity", lw=1.5)
plt.plot(indent_2/params.indenter_radius, force_2, marker='o', ms=6, label="MoFEM NeoHookeanHyperElasticity", lw=1.5)

plt.plot(indent_1/params.indenter_radius, hertz_press(indent_1, params), c='k', ls='--', label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Force, nN")
plt.legend(loc='upper left')
plt.grid()
```

### Comparison of the contact area evolution

```python
plt.plot(indent_1/params.indenter_radius, area_1, marker='o', ms=6, label="MoFEM LinearElasticity", lw=1.5)
plt.plot(indent_2/params.indenter_radius, area_2, marker='o', ms=6, label="MoFEM NeoHookeanHyperElasticity", lw=1.5)

plt.plot(indent_2/params.indenter_radius, hertz_area(indent_2, params), label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Contact area, um2")
plt.legend(loc='upper left')
plt.grid()
```

### Visualisation of the deformation

```python
params.show_field = "DISPLACEMENT"
params.show_component = "Y"
params.show_edges = False

show_results(params)
```

```python
params.show_field = "STRESS"
params.show_component = "Y"
params.show_edges = True

show_results(params)
```

## Indentation of a thin layer

```python
params.max_indentation = params.indenter_radius / 2
params.mesh_height = params.indenter_radius 
params.refine_radius = np.sqrt(params.indenter_radius * params.max_indentation) # a_hertz = sqrt(R * d)

params.mesh_length = params.refine_radius * 10
params.far_field_size = params.mesh_length / 10
params.near_field_size = params.mesh_height / 10

params.show_mesh = True
generate_mesh(params)

params.young_modulus = 100
params.poisson_ratio = 0.49
params.material_model = "NeoHookeanHyperElasticity"

mofem_compute_force_indent(params)
```

```python
params.show_field = "DISPLACEMENT"
params.show_component = "X"
params.show_edges = True

show_results(params)
```

```python
params.show_field = "STRESS"
params.show_component = "X"
params.show_edges = True

show_results(params)
```

```python
height_mult_list = [1, 2, 4, 8, 16]
indent_list = []
force_list = []

params.indenter_radius = 10 
params.max_indentation = 5

params.young_modulus = 100
params.poisson_ratio = 0.49
params.material_model = "NeoHookeanHyperElasticity"

params.refine_radius = np.sqrt(params.indenter_radius * params.max_indentation) # a_hertz = sqrt(R * d)

params.mesh_length = params.refine_radius * 50

    
params.near_field_size = params.refine_radius / 10
params.far_field_size = params.mesh_length / 10

params.show_mesh = False

for height_mult in height_mult_list:
    params.mesh_height = params.indenter_radius * height_mult
    
    generate_mesh(params)
    
    indent, force, area = mofem_compute_force_indent(params)
    indent_list.append(indent)
    force_list.append(force)
```

```python
for elem_num, indent, force in zip(height_mult_list, indent_list, force_list):
    plt.plot(indent / params.indenter_radius, force, marker='o', ms=4, lw=1.5, label="MoFEM: H / R = {}".format(elem_num))
    
plt.plot(indent/params.indenter_radius, hertz_press(indent, params), c='k', ls='--', label="Hertz formula", lw=2)

plt.xlabel("Normalised indentation, δ/R")
plt.ylabel("Force, nN")
plt.legend(loc='upper left')
plt.grid(ls=":")
```
