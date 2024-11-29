import subprocess
import re
import shutil
import os
import sys
import utils as ut
import time
import resource
import signal

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

# mesh generation logic is coupled with computation logic because the physical_groups list stores the meshsets data
# if to decouple need to persist physical_groups (which is a list of pydantic objects), not difficult to do but have to think of if you are sure that it conforms to the mesh
def generate_mesh(params):
    import mesh_create_common as mshcrte_common
    if getattr(params, "custom_mesh_filepath", False):
        params.physical_groups = params.custom_generate_physical_groups(params)
        partition_mesh(params)
        return
        
    if params.case_name == "test_2D":
        import mesh_create_test_2D as mshcrte
    elif params.case_name == "test_3D":
        import mesh_create_test_3D as mshcrte
    elif params.case_name in ["pile", "pile_manual"]:
        import mesh_create_pile as mshcrte
    else:
        raise NotImplementedError("2024-10-30: no mesh for this use case yet! (Or use case not defined yet)")    
    
    if params.case_name in ["test_2D", "test_3D"]:
        geo = mshcrte.draw_mesh(params)
        params.physical_groups = mshcrte.add_physical_groups(params, geo)
    elif params.case_name == "pile":
        # Thomas Lai 2024, not confirmed the mesh converges yet
        # issues with corner singularities
        # geo = mshcrte.draw_mesh_auto(params)
        # params.physical_groups = mshcrte.generate_physical_groups_auto(params, geo)
        
        geo = mshcrte.draw_mesh_cylinder(params)
        params.physical_groups = mshcrte.generate_physical_groups_cylinder(params, geo)
        params.physical_groups_dimTags = mshcrte.add_physical_groups(params.physical_groups)
        params.physical_groups = mshcrte.finalize_mesh(params, geo, params.physical_groups, params.physical_groups_dimTags)
    elif params.case_name == "pile_manual":
        # Emma Fontaine 2023
        geo = mshcrte.draw_mesh_manual(params)
        params.physical_groups = mshcrte.generate_physical_groups_manual(params, geo)
    else:
        raise NotImplementedError("What?")
        sys.exit()
    params.physical_groups = mshcrte_common.check_block_ids(params,params.physical_groups)
    params.physical_groups = mshcrte_common.generate_config(params,params.physical_groups)
    mshcrte_common.inject_configs(params)
    partition_mesh(params)
    return

@ut.track_time("PARTITIONING MESH with mofem_part")
def partition_mesh(params):
    try:
        with open(params.partition_log_file, 'w') as log_file:
            process = subprocess.Popen(
                [
                    params.partition_exe, 
                    '-my_file', f'{params.finalized_mesh_filepath}',
                    '-my_nparts', f'{params.nproc}',
                    '-output_file', f'{params.part_file}',
                    '-dim', f'{params.dim}',
                    '-adj_dim', f'{params.dim-1}',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line-buffered
            )
            
            # Read the output line by line
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Print to console
                log_file.write(line)  # Write to log file
                log_file.flush()  # Ensure the line is written immediately
            
            process.stdout.close()
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error partitioning mesh: {e}")

@ut.track_time("COMPUTING")
def mofem_compute(params):
    os.chdir(params.data_dir)
    shutil.copy(params.options_file, params.data_dir / "param_file.petsc")
    result = subprocess.run("rm -rf out*", shell=True, text=True)
    
    #not used, but needed for contact exe
    replace_template_sdf(params)
    
    mfront_arguments = []
    for physical_group in params.physical_groups:
        if physical_group.name.startswith("MFRONT_MAT"):
            mfront_block_id = physical_group.meshnet_id
            mi_block = physical_group.preferred_model.value
            mi_param_0 = physical_group.props[physical_group.preferred_model].mi_param_0
            mi_param_1 = physical_group.props[physical_group.preferred_model].mi_param_1
            mi_param_2 = physical_group.props[physical_group.preferred_model].mi_param_2
            mi_param_3 = physical_group.props[physical_group.preferred_model].mi_param_3
            mi_param_4 = physical_group.props[physical_group.preferred_model].mi_param_4
            mi_param_5 = physical_group.props[physical_group.preferred_model].mi_param_5
    
            mfront_arguments.append(
                f"-mi_lib_path_{mfront_block_id} /mofem_install/jupyter/thomas/mfront_interface/src/libBehaviour.so "
                f"-mi_block_{mfront_block_id} {mi_block} "
                f"-mi_param_{mfront_block_id}_0 {mi_param_0} "
                f"-mi_param_{mfront_block_id}_1 {mi_param_1} "
                f"-mi_param_{mfront_block_id}_2 {mi_param_2} "
                f"-mi_param_{mfront_block_id}_3 {mi_param_3} "
                f"-mi_param_{mfront_block_id}_4 {mi_param_4} "
                f"-mi_param_{mfront_block_id}_5 {mi_param_5} "
            )
    
    # Join mfront_arguments list into a single string
    mfront_arguments_str = ' '.join(mfront_arguments)
    
    command = [
        "bash", "-c",
        f"export OMPI_MCA_btl_vader_single_copy_mechanism=none && "
        f"time nice -n 10 mpirun --oversubscribe --allow-run-as-root "
        f"-np {params.nproc} {params.exe} "
        f"-file_name {params.part_file} "
        f"-sdf_file {params.sdf_file} "
        f"-order {params.order} "
        f"-contact_order 0 "
        f"-sigma_order 0 "
        f"{'-base demkowicz ' if (getattr(params, 'base', False) == 'hex') else ''} "
        f"-ts_dt {params.time_step} "
        f"-ts_max_time {params.final_time} "
        f"{mfront_arguments_str} "
        f"-mi_save_volume 1 "
        f"-mi_save_gauss {params.save_gauss} "
        f"{'-time_scalar_file ' + str(params.time_history_file) if getattr(params, 'time_history', False) else ''} "
        # f"-options_file {params.options_file} "
    ]
    
    start_time = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    
    # Log the command
    with open(params.log_file, 'w') as log_file:
        log_file.write(f"================== Mesh: ====================\n")
        log_file.write(f"{params.finalized_mesh_filepath}\n")
        log_file.flush()
    
    # Log the command
    with open(params.log_file, 'a') as log_file:
        log_file.write(f"================ Command: ====================\n")
        log_file.write(f"{' '.join(command)}\n")
        log_file.flush()
        
    # Append the contents of the PETSc params file to the log file
    with open(params.options_file, 'r') as petsc_file:
        petsc_params = petsc_file.read()
    
    with open(params.log_file, 'a') as log_file:
        log_file.write(f"================ Petsc params file: ====================\n")
        log_file.write(petsc_params)
        log_file.write(f"========================================================\n")
        log_file.flush()
    
    try:
        # Open the log file for writing
        with open(params.log_file, 'a') as log_file:
            # Start the subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=4096  # Use a larger buffer for efficiency
            )
            
            # Buffer for batch writing to log file
            log_buffer = []

            # Read the output line by line
            for line in iter(process.stdout.readline, ''):
                # Print to console
                print(line, end='')

                # Append to the log buffer
                log_buffer.append(line)

                # Flush to file every 50 lines
                if len(log_buffer) >= 50:
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()

                # Check for specific error message
                if "Mfront integration failed" in line:
                    print("Error detected: Mfront integration failed")
                    process.kill()
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()

            # Write any remaining lines in the buffer
            if log_buffer:
                log_file.writelines(log_buffer)
                log_file.flush()

            # Wait for the process to complete
            process.wait()

    except KeyboardInterrupt:
        print("Process interrupted by user")
        process.kill()
        process.wait()
    finally:
        end_time = time.time()
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        
        user_cpu_time = usage_end.ru_utime - usage_start.ru_utime
        system_cpu_time = usage_end.ru_stime - usage_start.ru_stime
        params.compute_cpu_time = user_cpu_time + system_cpu_time
        params.compute_wall_time = end_time - start_time
        
        with open(params.log_file, 'a') as log_file:
            log_file.write(f"Total CPU time: {params.compute_cpu_time:.6f} seconds\n")
            log_file.write(f"Wall-clock time: {params.compute_wall_time:.6f} seconds\n")
            log_file.flush()

    subprocess.run(f"grep 'Total force:' {params.log_file} > {params.total_force_log_file}", shell=True)
    subprocess.run(
        f"grep -A 2 'FIX_X_1' {params.log_file} | awk '/Force/' > {params.FIX_X_1_force_log_file}",
        shell=True
    )
    # subprocess.run(f"grep 'Force:' {params.log_file} > {params.force_log_file}", shell=True)
    subprocess.run(f"grep 'nb global dofs' {params.log_file} > {params.DOFs_log_file}", shell=True)
    
    

@ut.track_time("CONVERTING FROM .h5m TO .vtk")
def export_to_vtk(params):
    os.chdir(params.data_dir)
    # Step 1: List all `out_mi*.h5m` files and convert them to `.vtk` using `convert.py`
    out_to_vtk = subprocess.run("ls -c1 out_mi*.h5m", shell=True, text=True, capture_output=True)
    if out_to_vtk.returncode == 0:
        convert_result = subprocess.run(f"{params.h5m_to_vtk_converter} -np 4 out_mi*.h5m", shell=True)
        if convert_result.returncode == 0:
            print("Conversion to VTK successful.")
        else:
            print("Conversion to VTK failed.")
            return
        
    # Step 2: List all `.vtk` files in the current directory
    vtk_files = subprocess.run("ls -c1 *.vtk", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        vtk_files_list = vtk_files.stdout.splitlines()
        if not vtk_files_list:
            print("No .vtk files found.")
            return
        # Step 3: Move each `.vtk` file to the appropriate directory
        for vtk_file in vtk_files_list:
            try:
                if "gauss" in vtk_file:
                    shutil.move(vtk_file, os.path.join(params.vtk_gauss_dir, vtk_file))
                    print(f"Moved {vtk_file} to {params.vtk_gauss_dir}")
                else:
                    shutil.move(vtk_file, os.path.join(params.vtk_dir, vtk_file))
                    print(f"Moved {vtk_file} to {params.vtk_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to move {vtk_file}: {e}")
    else:
        raise RuntimeError(f"Failed to list .vtk files: {vtk_files.stderr}")
    
    # Step 4: List all `.h5m` files in the current directory
    h5m_files = subprocess.run("ls -c1 *.h5m", shell=True, text=True, capture_output=True)
    if h5m_files.returncode == 0:
        h5m_files_list = h5m_files.stdout.splitlines()
        if not h5m_files_list:
            print("No .h5m files found.")
            return
        # Step 5: Move each `.h5m` file to the appropriate directory
        for h5m_file in h5m_files_list:
            try:
                if "gauss" in h5m_file:
                    shutil.move(h5m_file, os.path.join(params.h5m_gauss_dir, h5m_file))
                    print(f"Moved {h5m_file} to {params.h5m_gauss_dir}")
                else:
                    shutil.move(h5m_file, os.path.join(params.h5m_dir, h5m_file))
                    print(f"Moved {h5m_file} to {params.h5m_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to move {h5m_file}: {e}")
    else:
        raise RuntimeError(f"Failed to list .h5m files: {h5m_files.stderr}")

def quick_visualization(params):
    import pyvista as pv
    pv.set_plot_theme("document")

    from pyvirtualdisplay import Display
    display = Display(backend="xvfb", visible=False, size=(1024, 768))
    display.start()
    vtk_files = subprocess.run(f"ls -c1 {params.vtk_dir}/*.vtk | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        final_file = files[-1]
        mesh = pv.read(final_file)
        mesh=mesh.shrink(0.95) 
        warp_factor = 1.0
        # mesh = mesh.warp_by_vector(vectors="U", factor = warp_factor)
        # show_field = "STRESS"
        show_field = "STRAIN" # U: displacement
        # show_field = "STRAIN" # U: displacement
        # print(mesh.point_data)
        # if mesh.point_data[show_field].shape[1] > 3:
            # cmap = "Spectral"
        p = pv.Plotter(notebook=True)
        p.add_mesh(mesh, scalars=show_field)
        # p.camera_position = [(-10, 0, 10), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        p.camera_position = 'xz'
        p.show(jupyter_backend='ipygany')