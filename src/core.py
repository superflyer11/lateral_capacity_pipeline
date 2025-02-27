import subprocess
import re
import shutil
import os
import sys
import utils as ut
import time
import resource
import signal
import custom_models as cm

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
        # params.physical_groups = mshcrte_common.generate_config(params,params.physical_groups)
        partition_mesh(params)
        return
        
    if "test_2D" in params.case_name:
        import mesh_create_test_2D as mshcrte
    elif "test_3D" in params.case_name:
        import mesh_create_test_3D as mshcrte
    elif "pile" in params.case_name:
        import mesh_create_pile as mshcrte
    else:
        raise NotImplementedError("2024-10-30: no mesh for this use case yet! (Or use case not defined yet)")    
    
    if params.case_name.startswith("test"):
        geo = mshcrte.draw_mesh(params)
        params.physical_groups = mshcrte.add_physical_groups(params, geo)
    elif params.case_name.startswith("pile_manual"):
        # Emma Fontaine 2023
        geo = mshcrte.draw_mesh_manual(params)
        params.physical_groups = mshcrte.generate_physical_groups_manual(params, geo)
    elif params.case_name.startswith("pile"):
        # Thomas Lai 2024, not confirmed the mesh converges yet
        # issues with corner singularities
        # geo = mshcrte.draw_mesh_auto(params)
        # params.physical_groups = mshcrte.generate_physical_groups_auto(params, geo)
        
        geo = mshcrte.draw_mesh_cylinder(params)
        params.physical_groups = mshcrte.generate_physical_groups_cylinder(params, geo)
        params.physical_groups_dimTags = mshcrte.add_physical_groups(params.physical_groups)
        params.physical_groups = mshcrte.finalize_mesh(params, geo, params.physical_groups, params.physical_groups_dimTags)
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
    shutil.copy(params.sdf_file, params.data_dir / "sdf.py")
    result = subprocess.run("rm -rf out*", shell=True, text=True)
    if getattr(params, "time_history", False):
        params.time_history.write(params.time_history_file)
    if getattr(params, "body_time_history", False):
        params.body_time_history.write(params.body_time_history_file)
    if getattr(params, "force_time_history", False):
        params.force_time_history.write(params.force_time_history_file)
    if getattr(params, "secondary_force_time_history", False):
        params.secondary_force_time_history.write(params.secondary_force_time_history_file)
    if getattr(params, "displacement_time_history", False):
        params.displacement_time_history.write(params.displacement_time_history_file)
    
    #not used, but needed for contact exe
    replace_template_sdf(params)
    
    if params.use_mfront:
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
                    f"-mi_lib_path_{mfront_block_id} /mofem_install/jupyter/thomas/mfront_modules/src/libBehaviour.so "
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
    else:
        adolc_arguments = []
        adolc_arguments.append(f"-b_bar {1 if getattr(params, 'b_bar', False) else 0}")
        if params.soil_model == cm.PropertyTypeEnum.le_adolc:
            adolc_arguments.append("-material VonMisses")
        elif params.soil_model == cm.PropertyTypeEnum.vM_adolc:
            adolc_arguments.append("-material VonMisses")
        elif params.soil_model == cm.PropertyTypeEnum.Hm_adolc:
            adolc_arguments.append("-material Paraboloidal")
        adolc_arguments_str = ' '.join(adolc_arguments)
    
    if params.exe == "/mofem_install/jupyter/thomas/um_view_release/adolc_plasticity/adolc_plasticity_3d":
        if params.case_name.startswith("pile"):
            if params.soil_model == cm.PropertyTypeEnum.le_adolc:
                additional_arguments = [
                    f"-ts_adapt_type none "
                ]
            elif params.soil_model == cm.PropertyTypeEnum.vM_adolc:
                additional_arguments = [
                    f"-ts_adapt_type TSMoFEMAdapt "
                    f"-ts_mofem_adapt_desired_it 11 "
                    # f"-ts_adapt_type none "
                ]
            elif params.soil_model == cm.PropertyTypeEnum.Hm_adolc:
                additional_arguments = [
                    # f"-ts_adapt_type TSMoFEMAdapt "
                    # f"-ts_mofem_adapt_desired_it 9 "
                    f"-ts_adapt_type none "
                ]
        else:
            additional_arguments = [
                f"-ts_adapt_type none "
            ]
    elif params.exe == "/mofem_install/jupyter/thomas/um_view/tutorials/adv-1/contact_3d":
        additional_arguments = [
            f"-sdf_file {params.sdf_file} -sdf_file {params.sdf_file} "
            f"-sigma_order 0 "
            f"-ts_adapt_type none "
            ]
    else:
        additional_arguments = [
                f"-ts_adapt_type none "
            ]
    additional_arguments_str = ' '.join(additional_arguments)
    command = [
        "bash", "-c",
        f"export OMPI_MCA_btl_vader_single_copy_mechanism=none && "
        f"time nice -n 10 mpirun --oversubscribe --allow-run-as-root "
        f"-np {params.nproc} {params.exe} "
        f"-file_name {params.part_file} "
        f"-order {params.order} "
        f"-contact_order 0 "
        f"-geom_order {1 if params.use_mfront else 0} "
        f"{'-base demkowicz ' if (getattr(params, 'base', False) == 'hex') else ''} "
        f"-ts_dt {params.time_step} "
        f"-ts_max_time {params.final_time} "
        f"{mfront_arguments_str if params.use_mfront else adolc_arguments_str} "
        f"{additional_arguments_str} "
        f"-use_mfront {'1' if params.use_mfront else '0'} "
        f"-mi_save_volume 1 "
        f"-mi_save_gauss {params.save_gauss} "
        f"{'-time_scalar_file ' + str(params.time_history_file) if getattr(params, 'time_history', False) else ''} "
        # f"-options_file {params.options_file} "
    ]
    
    # start_time = time.time()
    # usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    
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
        print(os.getcwd())
        # Open the log file for writing
        with open(params.log_file, 'a') as log_file:
            # Start the subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=256  # Use a larger buffer for efficiency
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
                if len(log_buffer) >= 1:
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()

                # Check for specific error message
                if "Mfront integration failed" in line or "Mfront integration succeeded but results are unreliable" in line:
                    print("Error detected: Mfront integration failed or is unreliable")
                    process.terminate()
                    process.wait()
                    log_file.writelines(log_buffer)
                    log_file.flush()
                    log_buffer.clear()
                    return

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

    params.FEA_completed = True
    # finally:
    #     end_time = time.time()
    #     usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        
    #     user_cpu_time = usage_end.ru_utime - usage_start.ru_utime
    #     system_cpu_time = usage_end.ru_stime - usage_start.ru_stime
    #     params.compute_cpu_time = user_cpu_time + system_cpu_time
    #     params.compute_wall_time = end_time - start_time
        
    #     with open(params.log_file, 'a') as log_file:
    #         log_file.write(f"Total CPU time: {params.compute_cpu_time:.6f} seconds\n")
    #         log_file.write(f"Wall-clock time: {params.compute_wall_time:.6f} seconds\n")
    #         log_file.flush()

    
    

@ut.track_time("CONVERTING FROM .h5m TO .vtk")
def export_to_vtk(params):
    os.chdir(params.data_dir)
    # Step 1: List all `out_mi*.h5m` files and convert them to `.vtk` using `convert.py`
    h5m_files = subprocess.run("ls -c1 out_*.h5m", shell=True, text=True, capture_output=True)
    h5m_files_list = h5m_files.stdout.splitlines()
    
    if h5m_files_list:
        print(f"Moving h5m files from working directory")
        
        for i, h5m_file in enumerate(h5m_files_list):
            try:
                if "gauss" in h5m_file:
                    shutil.move(h5m_file, params.h5m_gauss_dir / h5m_file)
                else:
                    shutil.move(h5m_file, params.h5m_dir / h5m_file)
                ut.print_progress(i + 1, len(h5m_files_list), decimals=1, bar_length=50)
            except Exception as e:
                raise RuntimeError(f"Failed to move {h5m_file}: {e}")
        
        
    print(f"\nConverting h5m files to vtk and moving to vtk dir")
    
    if params.convert_gauss == 1:
        
        h5m_files = subprocess.run(f"ls -c1 {params.h5m_gauss_dir}/out_*.h5m", shell=True, text=True, capture_output=True)
        vtk_files = subprocess.run(f"ls -c1 {params.vtk_gauss_dir}/*.vtk", shell=True, text=True,capture_output=True)
        
        h5m_files_list = h5m_files.stdout.splitlines()
        vtk_files_list = vtk_files.stdout.splitlines()

        # Filter H5M files for which conversion is needed
        h5m_to_convert = []
        for h5m_file in h5m_files_list:
            base_name = h5m_file.split("/")[-1].rsplit(".", 1)[0]
            corresponding_vtk_file = f"{params.vtk_gauss_dir}/{base_name}.vtk"

            if corresponding_vtk_file in vtk_files_list:
                h5m_mtime = os.path.getmtime(h5m_file)
                vtk_mtime = os.path.getmtime(corresponding_vtk_file)
                if vtk_mtime > h5m_mtime:
                    continue  # Skip conversion if VTK is newer than H5M

            h5m_to_convert.append(h5m_file)
        
        # Convert remaining H5M files
        if h5m_to_convert:
            convert_result = subprocess.run(
                f"{params.h5m_to_vtk_converter} -np 4 {' '.join(h5m_to_convert)}", shell=True
            )
            if convert_result.returncode != 0:
                print("Conversion to VTK failed.")
                sys.exit()
            print("Conversion to VTK successful.")
        
        vtk_files = subprocess.run(f"ls -c1 {params.h5m_gauss_dir}/*.vtk", shell=True, text=True,capture_output=True)
        vtk_files_list = vtk_files.stdout.splitlines()
        print(f"Moving vtk files from h5m dir to vtk dir")
        
        for i, vtk_file in enumerate(vtk_files_list):
            shutil.move(vtk_file, params.vtk_gauss_dir / vtk_file.split("/")[-1])
            ut.print_progress(i + 1, len(vtk_files_list), decimals=1, bar_length=50)
            
    h5m_files = subprocess.run(f"ls -c1 {params.h5m_dir}/out_*.h5m", shell=True, text=True, capture_output=True)
    vtk_files = subprocess.run(f"ls -c1 {params.vtk_dir}/*.vtk", shell=True, text=True,capture_output=True)
    
    h5m_files_list = h5m_files.stdout.splitlines()
    vtk_files_list = vtk_files.stdout.splitlines()

    # Filter H5M files for which conversion is needed
    h5m_to_convert = []
    for h5m_file in h5m_files_list:
        base_name = h5m_file.split("/")[-1].rsplit(".", 1)[0]
        corresponding_vtk_file = f"{params.vtk_dir}/{base_name}.vtk"

        if corresponding_vtk_file in vtk_files_list:
            h5m_mtime = os.path.getmtime(h5m_file)
            vtk_mtime = os.path.getmtime(corresponding_vtk_file)
            if vtk_mtime > h5m_mtime:
                continue  # Skip conversion if VTK is newer than H5M

        h5m_to_convert.append(h5m_file)

    # Convert remaining H5M files
    if h5m_to_convert:
        convert_result = subprocess.run(
            f"{params.h5m_to_vtk_converter} -np 4 {' '.join(h5m_to_convert)}", shell=True
        )
        if convert_result.returncode != 0:
            print("Conversion to VTK failed.")
            sys.exit()
        print("Conversion to VTK successful.")
    
    vtk_files = subprocess.run(f"ls -c1 {params.h5m_dir}/*.vtk", shell=True, text=True,capture_output=True)
    vtk_files_list = vtk_files.stdout.splitlines()
    print(f"Moving vtk files from h5m dir to vtk dir")
    for i, vtk_file in enumerate(vtk_files_list):
        shutil.move(vtk_file, params.vtk_dir / vtk_file.split("/")[-1])
        ut.print_progress(i + 1, len(vtk_files_list), decimals=1, bar_length=50)

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