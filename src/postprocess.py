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
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=(7, 7))

import custom_models as cm
import utils as ut
import calculations as calc
import plotting

@ut.track_time("EXTRACTING RESULTS FROM LOG")
def extract_log(params):
    # Extract the total force, force, and DOFs from the log file
    subprocess.run(f"grep 'Total force:' {params.log_file} > {params.total_force_log_file}", shell=True)
    subprocess.run(
        f"grep -A 2 'FIX_X_1' {params.log_file} | awk '/Force/' > {params.FIX_X_1_force_log_file}",
        shell=True
    )
    # subprocess.run(f"grep 'Force:' {params.log_file} > {params.force_log_file}", shell=True)
    subprocess.run(f"grep 'nb global dofs' {params.log_file} > {params.DOFs_log_file}", shell=True)
    subprocess.run(f"grep 'Nonlinear solve' {params.log_file} > {params.snes_log_file}", shell=True)

@ut.track_time("EXTRACTING RESULTS FROM .vtk FILES")
def extract(params):
    # Example fields and components to extract
    fields = ["STRAIN", "STRESS", "DISPLACEMENT"]
    components = {
        "STRAIN": 9,
        "STRESS": 9,
        "DISPLACEMENT": 3
    }

    # Initialize a dictionary to store DataFrames for each point ID
    point_dataframes = {}

    # Get the list of VTK files
    vtk_files = subprocess.run(
        f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", 
        shell=True, text=True, capture_output=True
    )
    vtk_files_list = vtk_files.stdout.splitlines()

    # Iterate through the VTK files
    for i, vtk_file_path in enumerate(vtk_files_list):
        vtk_file = params.vtk_dir / f'out_mi_{i}.vtk'
        if not os.path.exists(vtk_file):
            raise FileNotFoundError(f"File {vtk_file} does not exist.")
            continue

        # Read the VTK file
        mesh = pv.read(vtk_file)

        # Iterate through the points of interest
        for point in params.points_of_interest:
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
                raise ValueError(f"No points found at coordinates {point_coords} in file {vtk_file}")
                continue

            # Initialize DataFrames for each point ID if not already done
            for pid in all_point_ids:
                if pid not in point_dataframes:
                    data = {f"{field}_{comp}": [] for field in fields for comp in range(components[field])}
                    data['time'] = []
                    data['x'] = []
                    data['y'] = []
                    data['z'] = []
                    point_dataframes[pid] = pd.DataFrame(data)

            # Append data for each point ID
            for pid in all_point_ids:
                data = {f"{field}_{comp}": [] for field in fields for comp in range(components[field])}
                data['time'] = i  # Add a time column
                data['x'] = point.x
                data['y'] = point.y
                data['z'] = point.z
                for field_name in fields:
                    point_data = mesh.point_data[field_name][pid]
                    for comp in range(components[field_name]):
                        data[f"{field_name}_{comp}"].append(point_data[comp])

                df = pd.DataFrame(data)
                point_dataframes[pid] = pd.concat([point_dataframes[pid], df], ignore_index=True)
        ut.print_progress(i + 1, len(vtk_files_list), decimals=1, bar_length=50)
    print('\n')
    # Save the extracted data to a CSV file
    for idx, (pid, df) in enumerate(point_dataframes.items()):
        output_path = cm.Point(x=df["x"].iloc[0], y=df["y"].iloc[0], z=df["z"].iloc[0]).point_dir(params) / f"point_data_{pid}.csv"
        point_dataframes[pid].to_csv(output_path, index=False)
        ut.print_progress(idx + 1, len(point_dataframes), decimals=1, bar_length=50)

@ut.track_time("DOING NECESSARY ADDITIONAL CALCULATIONS")
def calculate(params):
    for i, point in enumerate(params.points_of_interest):
        ut.print_progress(i + 1, len(params.points_of_interest), decimals=1, bar_length=50)
        csv_files = [f for f in os.listdir(point.point_dir(params)) if f.startswith("point_data_") and f.endswith(".csv")]
        
        for csv_file in csv_files:
            pid = int(csv_file.split('_')[2].split('.')[0])
            df = pd.read_csv(point.point_dir(params) / csv_file)
            
            sig_xx = df['STRESS_0']
            sig_xy = df['STRESS_1']
            sig_xz = df['STRESS_2']
            sig_yy = df['STRESS_4']
            sig_yz = df['STRESS_5']
            sig_zz = df['STRESS_8']
            
            sig_1, sig_2, sig_3 = calc.calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_xz)
            
            df['SIG_1'] = sig_1
            df['SIG_2'] = sig_2
            df['SIG_3'] = sig_3
            
            p = calc.calculate_p(sig_1, sig_2, sig_3)
            J_2 = calc.calculate_J2(sig_1, sig_2, sig_3)
            J  = np.sqrt(J_2)
            tau_oct = np.sqrt(2 * J_2)
            df["SIG_eq"] = np.sqrt(3 * J_2)
            
            e_xx = df['STRAIN_0']
            e_xy = df['STRAIN_1']
            e_xz = df['STRAIN_2']
            e_yy = df['STRAIN_4']
            e_yz = df['STRAIN_5']
            e_zz = df['STRAIN_8']
            
            e_v, e_d = calc.calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)
            df['e_v'] = e_v
            df['e_d'] = e_d
            
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

            # Convert lists to JavaScript arrays
            x_1_js = ', '.join(map(str, x_1))
            y_1_js = ', '.join(map(str, y_1))
            z_1_js = ', '.join(map(str, z_1))
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
                var calculator = Desmos.Calculator3D(elt);"""
            if params.soil_model == cm.PropertyTypeEnum.le:
                pass
            elif params.soil_model == cm.PropertyTypeEnum.vM or params.soil_model == cm.PropertyTypeEnum.vMDefault:
                sig_y = 10
                H = 0
                interacative_html_js += f"""
                    calculator.setExpression({{id:'exp1', latex: 'I = x + y + z'}});
                    calculator.setExpression({{id:'exp2', latex: 'p = I / 3'}});
                    calculator.setExpression({{id:'exp3', latex: 'J_2= \\\\frac{{1}}{{6}} \\\\cdot ((x-y)^2 + (y-z)^2 + (z-x)^2) '}});
                    calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
                    calculator.setExpression({{id:'exp5', latex: 's_{{0}}={sig_y}'}});
                    calculator.setExpression({{id:'exp6', latex: 'H = {H}'}});

                    calculator.setExpression({{id:'exp7', 
                        latex: '0 = q - s_{{0}}',
                        color: Desmos.Colors.RED,
                    }});"""
            elif params.soil_model == cm.PropertyTypeEnum.dpHYPER or params.soil_model == cm.PropertyTypeEnum.dp:
                c = 10
                # a = 1e-16
                phi = 15 * math.pi / 180
                
                # Combine HTML and JavaScript to create interactive content within the notebook
                interacative_html_js += f"""
                    calculator.setExpression({{id:'exp1', latex: 'I = x + y + z'}});
                    calculator.setExpression({{id:'exp2', latex: 'p = I / 3'}});
                    calculator.setExpression({{id:'exp3', latex: 'J_2= \\\\frac{{1}}{{6}} \\\\cdot ((x-y)^2 + (y-z)^2 + (z-x)^2) '}});
                    calculator.setExpression({{id:'exp4', latex: 'q = \\\\sqrt{{3 \\\\cdot J_2}}'}});
                    calculator.setExpression({{id:'exp5', latex: 'p_{{hi}} = {phi}'}});
                    calculator.setExpression({{id:'exp6', latex: 'M_{{JP}} = \\\\frac{{2\\\\sqrt{{3}}\\\\sin p_{{hi}}}}{{3-\\\\sin p_{{hi}}}}'}});
                    calculator.setExpression({{id:'exp7', latex: 'a = 10^{{-12}}'}});
                    calculator.setExpression({{id:'exp8', latex: 'c = {c}'}});

                    calculator.setExpression({{id:'exp9', 
                        latex: '0 = + M_{{JP}} p + \\\\sqrt{{a^{{2}} M_{{JP}}^{{2}} + \\\\frac{{q}}{{\\\\sqrt{{3}}}}^{{2}}}} - M_{{JP}} \\\\cdot \\\\frac{{c}}{{\\\\tan p_{{hi}}}}',
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
            Html_file= open(f"{point.point_dir(params)}/Desmos3D_{pid}.html","w")
            Html_file.write(interacative_html_js)
            Html_file.close()

@ut.track_time("PLOTTING ALL POINTS")
def plot(params):
    data_force=pd.read_csv(params.FIX_X_1_force_log_file,sep='\s+',header=None)
    pile_tip_lateral_load = - data_force[4].values * 2 * (10 ** 6) / 1000
    
    for i, point in enumerate(params.points_of_interest):
        ut.print_progress(i + 1, len(params.points_of_interest), decimals=1, bar_length=50)
        csv_files = [f for f in os.listdir(point.point_dir(params)) if f.startswith("point_data_") and f.endswith(".csv")]
        for csv_file in csv_files:
            pid = int(csv_file.split('_')[2].split('.')[0])
            df = pd.read_csv(point.point_dir(params) / csv_file)
            disp_x = - np.array(df['DISPLACEMENT_0']) * 1000
            if len(pile_tip_lateral_load) == len(disp_x) + 1:
                pile_tip_lateral_load = pile_tip_lateral_load[:-1]
            # Create a DataFrame from the arrays
            df_final = pd.DataFrame({
                'disp_x': disp_x,
                'pile_tip_lateral_load': pile_tip_lateral_load
            })
            # Save the DataFrame to a CSV file
            df_final.to_csv(f"{point.point_dir(params)}/disp_x_vs_pile_tip_lateral_load_{pid}.csv", index=False)
            
            plotting.plot_x_ys(disp_x, [pile_tip_lateral_load], labels=["FEA"], x_label='Ground-level displacement$\mu_x$ [mm]', y_label='Lateral load $H$ [kN]', title='Lateral load $H$ vs $\mu_x$', save_as = f"{point.point_dir(params)}/412_H_ux_{pid}.png", show=True)
    
    plot_stress_3D(params)
