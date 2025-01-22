import math
import os
import re
import sys
import time
import json
from pathlib import Path
import subprocess
import zipfile
import warnings
import signal

import pydantic
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib

import custom_models as cm
import utils as ut
import calculations as calc
import plotting

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out!")

@ut.track_time("WRITING VTK ZIP")
def zip_vtks(params):
    vtk_files = subprocess.run(
        f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", 
        shell=True, text=True, capture_output=True
    )
    vtk_files_list = vtk_files.stdout.splitlines()
    ut.zip_vtks(vtk_files_list, params.vtk_zip)
    
    if params.save_gauss == 1 or params.convert_gauss == 1:
        vtk_files = subprocess.run(
            f"ls -c1 {params.vtk_gauss_dir}/*.vtk | sort -V", 
            shell=True, text=True, capture_output=True
        )
        vtk_files_list = vtk_files.stdout.splitlines()
        if vtk_files_list:
            ut.zip_vtks(vtk_files_list, params.vtk_zip_gauss)

@ut.track_time("EXTRACTING RESULTS FROM LOG")
def extract_log(params):
    # Extract the total force, force, and DOFs from the log file
    subprocess.run(f"grep 'Total force:' {params.log_file} > {params.total_force_log_file}", shell=True)
    subprocess.run(
        f"echo '{params.prescribed_BC_name}' > {params.PRESCRIBED_BC_force_log_file} && "
        f"grep -A 2 '{params.prescribed_BC_name}' {params.log_file} | awk '/Force/' >> {params.PRESCRIBED_BC_force_log_file}",
        shell=True
    )
    # subprocess.run(f"grep 'Force:' {params.log_file} > {params.force_log_file}", shell=True)
    subprocess.run(f"grep 'nb global dofs' {params.log_file} > {params.DOFs_log_file}", shell=True)
    subprocess.run(f"grep 'Nonlinear solve' {params.log_file} > {params.snes_log_file}", shell=True)
    subprocess.run(f"grep 'TS dt' {params.log_file} > {params.time_step_log_file}", shell=True)

def run_command(command):
    
    # try:
    #     result = subprocess.run(
    #         ["my_long_command", "arg1", "arg2"],
    #         check=True,  # Raise an exception if command fails
    #         text=True,   # Return output as text (if needed)
    #         capture_output=True  # Capture stdout/stderr (if needed)
    #     )
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("Command failed with return code:", e.returncode)
    # except KeyboardInterrupt:
    #     print("Interrupted!")
    
    
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')  # Print to console
    process.stdout.close()
    process.wait()
    return process.returncode

@ut.track_time("PULLING A SELECTED LINE OVER DEPTH AT THE FINAL TIMESTEP WITH pvpython")
def line_to_csv(params, line: cm.Line):
    # command = [
    #     params.paraview_path,
    #     "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/gauss_line_to_depth_csv.py" if params.save_gauss == 1 else "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/line_to_depth_csv.py",
    #     params.vtk_gauss_dir if params.save_gauss == 1 else params.vtk_dir,
    #     # params.vtk_dir,
    #     ,
    #     *line.pt1.flat(),
    #     *line.pt2.flat(),
    # ]
    # # Run the command using subprocess
    # run_command(command)
    vtk_files = subprocess.run(
        f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", 
        shell=True, text=True, capture_output=True
    )
    vtk_files_list = vtk_files.stdout.splitlines()
    vtk_file = vtk_files_list[-1]
    mesh = pv.read(vtk_file)
    data = mesh.sample_over_line(line.pt1.array(),line.pt2.array())
    # print(data)
    # print(data.points)
    # geom = np.array(data['vtkGhostType'])
    # print(geom)
    # sys.exit()
    try:
        geom = np.array(data['GEOMETRY'])
    except KeyError:
        geom = np.array(data.points)
    x = geom[:, 0]  # all rows, column 0
    y = geom[:, 1]  # all rows, column 1
    z = geom[:, 2]  # all rows, column 2
    try:
        disp = np.array(data['U'])
    except KeyError:
        disp = np.array(data['DISPLACEMENT'])
    
    ux = disp[:,0]
    uy = disp[:,1]
    uz = disp[:,2]
    stress = np.array(data['STRESS'])
    sigx = stress[:,0]
    sigy = stress[:,4]
    sigz = stress[:,8]
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ux': ux, 'uy': uy, 'uz': uz, 'sigx': sigx, 'sigy': sigy, 'sigz': sigz})
    df.to_csv(line.line_against_depth_csv_filepath(params), index=False)
    
@ut.track_time("CALCULATE FINAL TOTAL LE STRAIN ENERGY")
def calculate_final_total_strain_energy(params):
    
    vtk_files = subprocess.run(
        f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", 
        shell=True, text=True, capture_output=True
    )
    vtk_files_list = vtk_files.stdout.splitlines()
    final_vtk_file = vtk_files_list[-1]
    mesh = pv.read(final_vtk_file)
    # integrated_volume = mesh.integrate_data()
    # print(integrated_volume)
    # print(integrated_volume.array_names)
    # print(integrated_volume['Area'])
    # print(integrated_volume['Volume'])
    if params.exe == f"/mofem_install/jupyter/thomas/um_view/tutorials/adv-1/contact_3d":
        global_grad_data = mesh.point_data["STRAIN"]
    elif params.exe == f"/mofem_install/jupyter/thomas/um_view_release/adolc_plasticity/adolc_plasticity_3d":
        global_grad_data = mesh.point_data["GRAD"]
    else:
        raise NotImplementedError("This soil model is not implemented in this executable yet.")

    global_stress_data = mesh.point_data["STRESS"]
    # Calculate strain energy for each point
    strain_energy_array = 0.5 * np.sum(global_stress_data * global_grad_data, axis=1)

    # Assign the strain energy array to the mesh point data
    mesh.point_data["STRAIN_ENERGY"] = strain_energy_array

    # Integrate strain energy across the domain
    data = mesh.integrate_data()

    # Retrieve total strain energy from integrated data
    total_strain_energy_from_integration = data["STRAIN_ENERGY"][0]
    print(total_strain_energy_from_integration)
    subprocess.run(
        f"echo '{total_strain_energy_from_integration}' > {params.TOTAL_STRAIN_ENERGY_log_file}",
        shell=True
    )
    return total_strain_energy_from_integration
    # print(total_strain_energy_from_integration)
    # sys.exit()    
    # mesh.point_data["STRAIN_ENERGY"] = 0.5 * global_stress_data * global_grad_data
    # data = mesh.integrate_data()
    # print(data)
    # print(data.array_names)
    # print(data['STRAIN_ENERGY'])
    # print(global_stress_data[0])
    # print(global_grad_data[0])
    # total_strain_energy = 0
    # for i in range(len(global_stress_data)):
    #     strain_energy_node = np.sum(global_stress_data[i] * global_grad_data[i])/2
    #     total_strain_energy += strain_energy_node
    # print(total_strain_energy)
    # sys.exit()
    # total_strain_energy = 0.5 * np.sum(global_stress_data * global_grad_data)
    # print(total_strain_energy)
    # total_strain_energy = 0.5 * np.sum(np.dot(global_grad_data * global_stress_data))
    # print(total_strain_energy)
    # subprocess.run(
    #     f"echo '{total_strain_energy}' > {params.TOTAL_STRAIN_ENERGY_log_file}",
    #     shell=True
    # )
    # print(total_strain_energy)
    # return total_strain_energy
    
    
    # Iterate through the VTK files
    # for i, vtk_file in enumerate(vtk_files_list):
        # print(vtk_file_path)
        # vtk_file = params.vtk_dir / f'out_mi_{i}.vtk'
        # if not os.path.exists(vtk_file):
        #     raise FileNotFoundError(f"File {vtk_file} does not exist.")
        #     continue

        # Read the VTK file

@ut.track_time("EXTRACTING RESULTS FROM .vtk FILES")
def extract(params):
    if params.exe == f"/mofem_install/jupyter/thomas/um_view/tutorials/adv-1/contact_3d":
        fields = ["STRAIN", "STRESS", "DISPLACEMENT"]
        components = {
            "STRAIN": 9,
            "STRESS": 9,
            "DISPLACEMENT": 3
        }
    elif params.exe == f"/mofem_install/jupyter/thomas/um_view_release/adolc_plasticity/adolc_plasticity_3d":
        fields = ["GRAD", "STRESS", "U"]
        components = {
            "GRAD": 9,
            "STRESS": 9,
            "U": 3
        }
    else:
        raise NotImplementedError("This soil model is not implemented in this executable yet.")

    

    # Initialize a dictionary to store DataFrames for each point ID
    point_dataframes = {}

    # Get the list of VTK files
    vtk_files = subprocess.run(
        f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", 
        shell=True, text=True, capture_output=True
    )
    vtk_files_list = vtk_files.stdout.splitlines()

    # Iterate through the VTK files
    for i, vtk_file in enumerate(vtk_files_list):
        # print(vtk_file_path)
        # vtk_file = params.vtk_dir / f'out_mi_{i}.vtk'
        # if not os.path.exists(vtk_file):
        #     raise FileNotFoundError(f"File {vtk_file} does not exist.")
        #     continue

        # Read the VTK file
        mesh = pv.read(vtk_file)
        skip_this = False

        # Iterate through the points of interest
        for point in params.points_of_interest:
            if skip_this:
                break
            point_coords = point.array()

            tolerance = 1e-6
            differences = mesh.points - point_coords  # Element-wise subtraction
            norms = np.linalg.norm(differences, axis=1)  # Compute the norm (magnitude) for each point

            # Find the indices where the norm is within the tolerance
            all_point_ids = np.where(norms <= tolerance)[0]
            # print(all_point_ids)

            # Find all point IDs that match the coordinates
            # all_point_ids = np.where(np.all(mesh.points == point_coords, axis=1))[0]
            if len(all_point_ids) == 0:
                warnings.warn(f"No points found at coordinates {point_coords} in file {vtk_file}")
                if i not in params.skipped_vtks:
                    params.skipped_vtks.append(i)
                    skip_this = True
                    break

            # Initialize DataFrames for each point ID if not already done
            for pid in all_point_ids:
                if pid not in point_dataframes:
                    data = {f"{field}_{comp}": [] for field in fields for comp in range(components[field])}
                    data['x'] = []
                    data['y'] = []
                    data['z'] = []
                    point_dataframes[pid] = pd.DataFrame(data)
                    
            # Append data for each point ID
            for pid in all_point_ids:
                if skip_this:
                    break
                data = {f"{field}_{comp}": [] for field in fields for comp in range(components[field])}
                data['x'] = [point.x]
                data['y'] = [point.y]
                data['z'] = [point.z]
                for field_name in fields:
                    try:
                        point_data = mesh.point_data[field_name][pid]
                        for comp in range(components[field_name]):
                            data[f"{field_name}_{comp}"].append(point_data[comp])
                    except KeyError:
                        if i not in params.skipped_vtks:
                            params.skipped_vtks.append(i)
                        skip_this = True
                        break
                for field_name, field_data in data.items():
                    if len(field_data) == 0:
                        skip_this = True
                        break
                    # print(field_name, field_data)
                    # print(field_name, len(field_data))
                if skip_this == True:
                    break
                df = pd.DataFrame(data)
                point_dataframes[pid] = pd.concat([point_dataframes[pid], df], ignore_index=True)
        ut.print_progress(i + 1, len(vtk_files_list), decimals=1, bar_length=50)
        
        
    with params.skipped_vtks_log_file.open("w") as f:
        f.write(' '.join(str(x) for x in sorted(set(params.skipped_vtks))))
        print('\n')
        print(f"Updated skipped_vtks written to log file: {params.skipped_vtks_log_file}")
        
    print(f"Skipped VTK files: {params.skipped_vtks}")
    # Save the extracted data to a CSV file
    for idx, (pid, df) in enumerate(point_dataframes.items()):
        output_path = cm.Point(x=df["x"].iloc[0], y=df["y"].iloc[0], z=df["z"].iloc[0]).point_dir(params) / f"point_data_{pid}.csv"
        point_dataframes[pid].to_csv(output_path, index=False)
        ut.print_progress(idx + 1, len(point_dataframes), decimals=1, bar_length=50)

@ut.track_time("DOING NECESSARY ADDITIONAL CALCULATIONS")
def calculate(params):
    if not params.use_mfront:
        calculate_final_total_strain_energy(params)
    
    for i, point in enumerate(params.points_of_interest):
        ut.print_progress(i + 1, len(params.points_of_interest), decimals=1, bar_length=50)
        csv_files = [f for f in os.listdir(point.point_dir(params)) if f.startswith("point_data_") and f.endswith(".csv")]
        
        for csv_file in csv_files:
            pid = int(csv_file.split('_')[2].split('.')[0])
            df = pd.read_csv(point.point_dir(params) / csv_file)
            
            df['sig_xx'] = df['STRESS_0']
            df['sig_xy'] = df['STRESS_1']
            df['sig_xz'] = df['STRESS_2']
            df['sig_yy'] = df['STRESS_4']
            df['sig_yz'] = df['STRESS_5']
            df['sig_zz'] = df['STRESS_8']
            
            
            df['SIG_1'], df['SIG_2'], df['SIG_3'] = calc.calculate_principal_stresses(df['sig_xx'], df['sig_yy'], df['sig_zz'], df['sig_xy'], df['sig_yz'], df['sig_xz'])
            df['tau_1'] = (df['SIG_1']-df['SIG_2'])/2
            df['tau_2'] = (df['SIG_2']-df['SIG_3'])/2
            df['tau_3'] = (df['SIG_3']-df['SIG_1'])/2
            
            df['p'] = calc.calculate_p(df['SIG_1'], df['SIG_2'], df['SIG_3'])
            J_2 = calc.calculate_J2(df['SIG_1'], df['SIG_2'], df['SIG_3'])
            J  = np.sqrt(J_2)
            tau_oct = np.sqrt(2 * J_2)
            df["SIG_eq"] = np.sqrt(3 * J_2)
            
            # Check if 'GRAD' columns exist and 'STRAIN' columns do not exist
            if all(f'GRAD_{i}' in df.columns for i in range(9)) and not any(f'STRAIN_{i}' in df.columns for i in [0, 1, 2, 4, 5, 8]):
                # Compute the strain components from the gradient tensor
                df['STRAIN_0'] = df['GRAD_0']
                df['STRAIN_4'] = df['GRAD_4']
                df['STRAIN_8'] = df['GRAD_8']

                df['STRAIN_1'] = 0.5 * (df['GRAD_1'] + df['GRAD_3'])  # Symmetric off-diagonal terms
                df['STRAIN_2'] = 0.5 * (df['GRAD_2'] + df['GRAD_6'])
                df['STRAIN_5'] = 0.5 * (df['GRAD_5'] + df['GRAD_7'])

            # Rename columns with 'U_{number}' to 'DISPLACEMENT_{number}'
            df.columns = [re.sub(r'U_(\d+)', r'DISPLACEMENT_\1', col) for col in df.columns]
            
            df['e_v'], df['e_d'] = calc.calculate_volumetric_and_deviatoric_strain(df['STRAIN_0'], df['STRAIN_4'], df['STRAIN_8'], df['STRAIN_1'], df['STRAIN_2'] , df['STRAIN_5'])
            
            def calculate_stress_magnitude(row):
                stress_tensor = np.array([
                    [row['STRESS_0'], row['STRESS_1'], row['STRESS_2']],
                    [row['STRESS_1'], row['STRESS_4'], row['STRESS_5']],
                    [row['STRESS_2'], row['STRESS_5'], row['STRESS_8']]
                ])
                return np.linalg.norm(stress_tensor)
            
            def calculate_strain_magnitude(row):
                strain_tensor = np.array([
                        [row['STRAIN_0'], row['STRAIN_1'], row['STRAIN_2']],
                        [row['STRAIN_1'], row['STRAIN_4'], row['STRAIN_5']],
                        [row['STRAIN_2'], row['STRAIN_5'], row['STRAIN_8']]
                    ])
                return np.linalg.norm(strain_tensor)
            
            def calculate_displacement_magnitude(row):
                return np.sqrt(
                        row['DISPLACEMENT_0']**2 + 
                        row['DISPLACEMENT_1']**2 + 
                        row['DISPLACEMENT_2']**2
                    )
            
            df['STRESS_magnitude'] = df.apply(calculate_stress_magnitude, axis=1)
            df['STRAIN_magnitude'] = df.apply(calculate_strain_magnitude, axis=1)
            df['DISPLACEMENT_magnitude'] = df.apply(calculate_displacement_magnitude, axis=1)
            
            df.to_csv(point.point_dir(params) /  f"point_data_{pid}.csv", index=False)



def plot_stress_3D(params):
    for point in params.points_of_interest:
        csv_files = [f for f in os.listdir(point.point_dir(params)) if f.startswith("point_data_") and f.endswith(".csv")]
        for csv_file in csv_files:
            pid = int(csv_file.split('_')[2].split('.')[0])
            df = pd.read_csv(point.point_dir(params) / csv_file)
            sig_1 = np.array(df['SIG_1'])
            sig_2 = np.array(df['SIG_2'])
            sig_3 = np.array(df['SIG_3'])

            x_1 = sig_1
            y_1 = sig_2
            z_1 = sig_3
            
            def format_to_decimal(value, decimal_places=4):
                return f"{value:.{decimal_places}f}"

            # Convert Python lists to JavaScript arrays in pure decimal format
            x_1_js = ', '.join(format_to_decimal(v) for v in x_1)
            y_1_js = ', '.join(format_to_decimal(v) for v in y_1)
            z_1_js = ', '.join(format_to_decimal(v) for v in z_1)

            # # Convert Python lists to JavaScript arrays in scientific notation
            # x_1_js = ', '.join(format_to_scientific(v) for v in x_1)
            # y_1_js = ', '.join(format_to_scientific(v) for v in y_1)
            # z_1_js = ', '.join(format_to_scientific(v) for v in z_1)

            # # Convert lists to JavaScript arrays
            # x_1_js = ', '.join(map(str, x_1))
            # y_1_js = ', '.join(map(str, y_1))
            # z_1_js = ', '.join(map(str, z_1))
            interacative_html_js = f"""
    <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Desmos Calculator Debug</title>
                <script src="https://www.desmos.com/api/v1.11/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
            </head>
            <body>
            <div id="calculator" style="width: 1200px; height: 600px;"></div>
            <script>
                var elt = document.getElementById('calculator');
                var calculator = Desmos.Calculator3D(elt);
                calculator.setExpression({{id:'exp1', latex: 'I_1 = x + y + z'}});
                calculator.setExpression({{id:'exp2', latex: 'p = I_1 / 3'}});
                calculator.setExpression({{id:'exp3', latex: 'J_2= \\\\frac{{1}}{{6}} \\\\cdot ((x-y)^2 + (y-z)^2 + (z-x)^2) '}});"""
            if params.soil_model == cm.PropertyTypeEnum.le:
                pass
            elif params.soil_model == cm.PropertyTypeEnum.vM_Default_mfront or params.soil_model == cm.PropertyTypeEnum.vM_Implicit_mfront or params.soil_model == cm.PropertyTypeEnum.vM_adolc:
                sig_y = 10
                H = 0
                interacative_html_js += f"""
                    calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
                    calculator.setExpression({{id:'exp5', latex: 's_{{0}}={sig_y}'}});
                    calculator.setExpression({{id:'exp6', latex: 'H = {H}'}});

                    calculator.setExpression({{id:'exp7', 
                        latex: '0 = q - s_{{0}}',
                        color: Desmos.Colors.RED,
                    }});"""
            elif params.soil_model == cm.PropertyTypeEnum.DP_HYPER or params.soil_model == cm.PropertyTypeEnum.DP:
                c = 10
                # a = 1e-16
                phi = 15 * math.pi / 180
                
                # Combine HTML and JavaScript to create interactive content within the notebook
                interacative_html_js += f"""
                    calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
                    calculator.setExpression({{id:'exp5', latex: 'p_{{hi}} = {phi}'}});
                    calculator.setExpression({{id:'exp6', latex: 'M_{{JP}} = \\\\frac{{2\\\\sqrt{{3}}\\\\sin p_{{hi}}}}{{3-\\\\sin p_{{hi}}}}'}});
                    calculator.setExpression({{id:'exp7', latex: 'a = 10^{{-12}}'}});
                    calculator.setExpression({{id:'exp8', latex: 'c = {c}'}});

                    calculator.setExpression({{id:'exp9', 
                        latex: '0 = + M_{{JP}} p + \\\\sqrt{{a^{{2}} M_{{JP}}^{{2}} + \\\\frac{{q}}{{\\\\sqrt{{3}}}}^{{2}}}} - M_{{JP}} \\\\cdot \\\\frac{{c}}{{\\\\tan p_{{hi}}}}',
                        color: Desmos.Colors.RED,
                    }});"""
            elif params.soil_model == cm.PropertyTypeEnum.Hm_adolc:
                s_yt = 0.2
                s_yc = 0.3
                H_t = 0
                H_c = 0
                a_0 = 1
                a_1 = 1
                a_0bar = s_yt + H_t * a_0
                a_1bar = s_yc + H_c * a_1
                
                # Combine HTML and JavaScript to create interactive content within the notebook
                interacative_html_js += f"""
                    calculator.setExpression({{id:'exp4', latex: 's_{{yt}} = {s_yt}'}});
                    calculator.setExpression({{id:'exp5', latex: 's_{{yc}} = {s_yc}'}});
                    calculator.setExpression({{id:'exp6', latex: 'H_{{t}} = {H_t}'}});
                    calculator.setExpression({{id:'exp7', latex: 'H_{{c}} = {H_c}'}});
                    calculator.setExpression({{id:'exp8', latex: 'a_{{0}} = {a_0}'}});
                    calculator.setExpression({{id:'exp9', latex: 'a_{{1}} = {a_1}'}});
                    calculator.setExpression({{id:'exp10', latex: 'a_{{0bar}} = {a_0bar}'}});
                    calculator.setExpression({{id:'exp11', latex: 'a_{{1bar}} = {a_1bar}'}});

                    calculator.setExpression({{id:'exp12', 
                        latex: '0 = 6 \\\\cdot J_2 + 2 \\\\cdot I_1 \\\\cdot (a_{{1bar}} - a_{{0bar}}) - 2 \\\\cdot a_{{0bar}} \\\\cdot a_{{1bar}}',
                        color: Desmos.Colors.RED,
                    }});"""        
                    
                    
            interacative_html_js += f"""
                    calculator.setExpression({{
                        type: 'table',
                        columns: [
                            {{
                                latex: 'x_1',
                                values: [{x_1_js}]
                            }},
                            {{
                                latex: 'y_1',
                                values: [{y_1_js}],
                            }},
                            {{
                                latex: 'z_1',
                                values: [{z_1_js}],
                            }},
                        ]
                    }});

                    calculator.setExpression({{id:'exp20', 
                        latex: '(x_{{1}},y_{{1}},z_{{1}})',
                        color: Desmos.Colors.BLUE,
                    }});
                    
                    
                    function downloadScreenshot() {{
                        var screenshot = calculator.screenshot();
                        var link = document.createElement('a');
                        link.href = screenshot;
                        link.download = 'screenshot.png';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                    }}
                    
                </script>
                <h2>Interactive Content</h2>
                <button onclick="downloadScreenshot()">Click me to download screenshot!</button>
                </body>
                """
            pid_dir = point.point_dir(params) / f"{pid}"
            pid_dir.mkdir(parents=True, exist_ok=True)
            Html_file= open(f"{pid_dir}/Desmos3D_{pid}.html","w")
            Html_file.write(interacative_html_js)
            Html_file.close()
def sign(num):
    return -1 if num < 0 else 1

@ut.track_time("PLOTTING ALL POINTS")
def plot(params):
    data_force = pd.read_csv(params.PRESCRIBED_BC_force_log_file, sep='\s+', header=None, skiprows=1)
    if params.prescribed_BC_name == 'FIX_Z_1':
        prescribed_load = - data_force[6].values * 2 * (10 ** 6) / 1000
    elif params.prescribed_BC_name == 'FIX_X_1':
        prescribed_load = - data_force[4].values * 2 * (10 ** 6) / 1000
    elif params.prescribed_BC_name == 'FIX_Z_0':
        prescribed_load = - data_force[4].values * (10 ** 6) / 1000
    else:
        raise NotImplementedError("This boundary condition is not implemented yet for plotting.")
    
    data_time = pd.read_csv(params.time_step_log_file, sep='\s+', header=None)
    time = data_time[8].values
    
    if params.skipped_vtks_log_file.exists() and params.skipped_vtks_log_file.stat().st_size > 0:
        with params.skipped_vtks_log_file.open("r") as f:
            params.skipped_vtks = [
                int(val) for line in f for val in line.strip().split() if val.strip().isdigit()
            ]
    else:
        params.skipped_vtks = []
    prescribed_load = np.delete(prescribed_load, [i for i in params.skipped_vtks if i < len(prescribed_load)])
    time = np.delete(time, [i for i in params.skipped_vtks if i < len(time)])
    
    for i, point in enumerate(params.points_of_interest):
        ut.print_progress(i + 1, len(params.points_of_interest), decimals=1, bar_length=50)
        csv_files = [f for f in os.listdir(point.point_dir(params)) if f.startswith("point_data_") and f.endswith(".csv")]
        for csv_file in csv_files:
            pid = int(csv_file.split('_')[2].split('.')[0])
            df = pd.read_csv(point.point_dir(params) / csv_file)
            if params.prescribed_BC_name == 'FIX_Z_1':
                disp = - np.array(df['DISPLACEMENT_2']) * 1000
                symbol = "uz"
                latex_symbol = "$\mu_z$"
            elif params.prescribed_BC_name == 'FIX_X_1':
                disp = - np.array(df['DISPLACEMENT_0']) * 1000
                symbol = "ux"
                latex_symbol = "$\mu_x$"
            elif params.prescribed_BC_name == 'FIX_Z_0':
                disp = - np.array(df['DISPLACEMENT_2']) * 1000
                symbol = "uz"
                latex_symbol = "$\mu_z$"
            else:
                raise NotImplementedError("This boundary condition is not implemented yet for plotting.")
            
            p = np.array(df['p'])
            sig_xx = np.array(df['STRESS_0']) 
            sig_yy = np.array(df['STRESS_4']) 
            sig_zz = np.array(df['STRESS_8']) 
            e_zz = np.array(df['STRAIN_8']) 
            
            sig_eq = np.array(df['SIG_eq']) 
            tau_1 = np.array(df['tau_1']) 
            tau_2 = np.array(df['tau_2']) 
            tau_3 = np.array(df['tau_3']) 
            e_v = np.array(df['e_v']) 
            e_d = np.array(df['e_d']) 
            
            
            if len(prescribed_load) == len(disp) + 1:
                prescribed_load = prescribed_load[:-1]
                
            if len(time) == len(disp) + 1:
                time = time[:-1]
            
            pid_dir = point.point_dir(params) / f"{pid}"
            pid_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a DataFrame from the arrays
            # print(f"Rows in csv: {len(disp)}, rows in log extract: {len(prescribed_load)}")
            df_final = pd.DataFrame({
                'disp': disp,
                'prescribed_load': prescribed_load,
                'time': time,
            })
            # Save the DataFrame to a CSV file
            df_final.to_csv(f"{pid_dir}/disp_{symbol}_vs_prescribed_load_{pid}.csv", index=False)
            
            
            
            plotting.plot_x_ys(disp, [prescribed_load], labels=["FEA"], x_label=f'Displacement {latex_symbol} [mm]', y_label='Prescribed load $H$ [kN]', title=f'Prescribed load $H$ vs {latex_symbol}', save_as = f"{pid_dir}/412_H_{symbol}_{pid}.png", show=False)
            plotting.plot_x_ys(p, [sig_eq], labels=["FEA"], x_label='Hydrostatic stress $p$', y_label='Equivalent stress $sig_{eq}$', title='Equivalent Stress vs Hydrostatic stress', save_as = f"{pid_dir}/111_sigeq_p_{pid}.png", show=False)
            plotting.plot_x_ys(disp, [sig_eq], labels=["FEA"], x_label=f'Displacement {latex_symbol} [mm]', y_label='Equivalent stress $sig_{eq}$', title=f'Equivalent Stress vs Displacement {latex_symbol} [mm]', save_as = f"{pid_dir}/121_sigeq_{symbol}_{pid}.png", show=False)
            plotting.plot_x_ys(time, [np.sign(sig_zz-sig_xx) * sig_eq], labels=["FEA"], x_label='Time $t$', y_label='$sign(\sigma_{zz}-\sigma_{xx}) \cdot$ Equivalent stress $sig_{eq}$', title='Equivalent Stress vs Time', save_as = f"{pid_dir}/131_sigeq_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [tau_1], labels=["FEA"], x_label='Time $t$', y_label='Shear stress $tau_1$', title='Shear stress $tau_1$ vs Time', save_as = f"{pid_dir}/151_tau1_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [tau_2], labels=["FEA"], x_label='Time $t$', y_label='Shear stress $tau_2$', title='Shear stress $tau_2$ vs Time', save_as = f"{pid_dir}/152_tau2_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [tau_3], labels=["FEA"], x_label='Time $t$', y_label='Shear stress $tau_3$', title='Shear stress $tau_3$ vs Time', save_as = f"{pid_dir}/153_tau3_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [sig_xx], labels=["FEA"], x_label='Time $t$', y_label='Stress xx $\sigma_{xx}$', title='Stress $\sigma_{xx}$ vs Time $t$', save_as = f"{pid_dir}/211_sigxx_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [sig_yy], labels=["FEA"], x_label='Time $t$', y_label='Stress yy $\sigma_{yy}$', title='Stress $\sigma_{yy}$ vs Time $t$', save_as = f"{pid_dir}/212_sigyy_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [sig_zz], labels=["FEA"], x_label='Time $t$', y_label='Stress zz $\sigma_{zz}$', title='Stress $\sigma_{zz}$ vs Time $t$', save_as = f"{pid_dir}/213_sigzz_t_{pid}.png", show=False)
            if "test_3D_uniaxial" in params.case_name:
                if params.soil_model == cm.PropertyTypeEnum.vM_Default_mfront:
                    label = f"""von Mises (MFront Default DSL) - FEA
$E = {500}$
$\\nu = {0.3}$
$\\sigma_y$ = {0.25}
H = {2.5}"""
                elif params.soil_model == cm.PropertyTypeEnum.vM_Implicit_mfront:
                    label = f"""von Mises (MFront Implicit DSL) - FEA
$E = {500}$
$\\nu = {0.3}$
$\\sigma_y$ = {0.25}
H = {2.5}"""
                elif params.soil_model == cm.PropertyTypeEnum.vM_adolc:
                    label = f"""von Mises (ADOL-C) - FEA
$E = {500}$
$\\nu = {0.3}$
$\\sigma_y$ = {0.25}
H = {2.5}"""
                elif params.soil_model == cm.PropertyTypeEnum.DP:
                    label = r"""Drucker-Prager (MFront Implicit DSL) - FEA
$E = 500$
$\nu = 0.3$
$\phi = 10\degree$
c = 0.125
$v = 0\degree$"""
                elif params.soil_model == cm.PropertyTypeEnum.Hm_adolc:
                    label = r"""Paraboloidal (ADOL-C) - FEA
$E = 500$
$\nu = 0.3$
$\nu_p = 0.5$
$\sigma_{yc}$ = 0.25
$\sigma_{yt}$ = 0.20
$H_C$ = 0
$H_T$ = 0"""
            else:
                label = ""
            plotting.plot_x_ys(e_zz * 100, [sig_zz], labels=[label], x_label='Axial Strain $\epsilon_{zz}$ [%]', y_label='Uniaxial Stress $\sigma_{zz}$', title='', save_as = f"{pid_dir}/301_sigzz_ezz_{pid}.png", show=False,scale_up=1.2) #vertical_axis="right",
            plotting.plot_x_ys(e_zz * 100, [np.sign(sig_zz-sig_xx) * sig_eq], labels=[""], colors=['r'], x_label='Axial strain $\epsilon_{zz}$ [%]', y_label='$sign(\sigma_{zz}-\sigma_{xx}) \cdot$ Equivalent Stress $\sigma_{eq}$ [MPa]', title='', save_as = f"{pid_dir}/311_sigeq_ezz_{pid}.png", show=False, scale_up=1.2) #, vertical_axis="right", horizontal_axis='top'
            plotting.plot_x_ys(time, [e_zz * 100], labels=["FEA"], x_label='t', y_label='e_{zz}', title='e_{zz} vs t', save_as = f"{pid_dir}/321_ezz_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [e_d], labels=["FEA"], x_label='t', y_label='ed', title='ed vs t', save_as = f"{pid_dir}/331_ed_t_{pid}.png", show=False)
            plotting.plot_x_ys(time, [e_v], labels=["FEA"], x_label='t', y_label='ev', title='ev vs t', save_as = f"{pid_dir}/332_ev_t_{pid}.png", show=False)
            # plotting.plot_x_ys(STRAIN_magnitude, [STRESS_magnitude], labels=["FEA"], x_label='strain', y_label='stress', title='Stress vs strain', save_as = f"{pid_dir}/251_sig_e_{pid}.png", show=False)
    
    plot_stress_3D(params)


def extract_last_column_as_int(file_path, expected_fields):
    last_values = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.split()  # Split the line by whitespace
            if len(fields) == expected_fields:  # Check if the line has the expected number of fields
                try:
                    last_values.append(int(fields[-1]))  # Convert the last field to an integer
                except ValueError:
                    # Skip if the last field cannot be converted to an integer
                    pass
    return pd.Series(last_values)  # Return as a pandas Series



@ut.track_time("PLOTTING ALL POINTS")
def snes_csv(params):
    # data_snes = pd.read_csv(, sep='\s+', header=None)
    # snes_it = data_snes[10].values
    snes_it = extract_last_column_as_int(params.snes_log_file, 11)
    data_ts = pd.read_csv(params.time_step_log_file, sep='\s+', header=None)
    time = data_ts[8].values[:-1]
    dt = data_ts[6].values[:-1]
    norm_snes_it = snes_it/dt
    df_final = pd.DataFrame({
        'snes_it': snes_it,
        'time': time,
        'dt': dt,
        'norm_snes_it': norm_snes_it,
    })
    # Save the DataFrame to a CSV file
    df_final.to_csv(params.snes_csv, index=False)