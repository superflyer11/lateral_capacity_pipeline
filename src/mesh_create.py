import os
import sys
import math
import re
import time
import subprocess
from pathlib import Path
from functools import wraps
from typing import List, Optional, Union
from warnings import warn

import gmsh
import pyvista as pv

import custom_models as cm
import utils as ut



@ut.track_time("RUNNING COMMAND...")
def run_command(index, command, log_file):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            log_file.write(output)
            log_file.flush()
            sys.stdout.write(output)
            sys.stdout.flush()
    
    # Capture remaining output after process ends
    for output in process.stdout.readlines():
        log_file.write(output)
        log_file.flush()
        sys.stdout.write(output)
        sys.stdout.flush()
    
    return process


# os.chdir("/mofem_install/um-build-Release-u3my3yi/tutorials/vec-10000")
class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value



@ut.track_time("DRAWING MESH")
def draw_mesh(params) -> cm.GeometryTagManager:
    """
    Draws the mesh using Gmsh based on the provided parameters.

    Args:
        params: Parameters including mesh name and other necessary configurations.

    Returns:
        cm.GeometryTagManager: An object containing tags for soil and pile volumes and surfaces.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    


    gmsh.model.add(f"{params.mesh_name}")

    soil_layer_tags = []
    # Add boxes for the layers
    for no, layer in params.box_manager.layers.items():
        new_tag = params.box_manager.add_layer(layer)
        soil_layer_tags.append(new_tag)

    
    
    outer_cylinder_tag, inner_cylinder_tag = params.pile_manager.addPile()

    # Cut the soil blocks with the pile
    DIM = 3
    cut_soil_tags, _ = gmsh.model.occ.cut(
        [[3, tag] for tag in soil_layer_tags],
        [[3, outer_cylinder_tag]],
        -1,
        removeObject=True,
        removeTool=False,
    )
    cut_soil_boxes = [tag for _, tag in cut_soil_tags]

    symmetry_cutter_tag = gmsh.model.occ.addBox(-200, 0, -200, 400, 400, 400)

    # Cut the outer cylinder with the inner cylinder to form the pile
    cut_pile_tags, _ = gmsh.model.occ.cut(
        [[3, outer_cylinder_tag]],
        [[3, inner_cylinder_tag]],
        -1,
        removeObject=True,
        removeTool=True,
    )
    pile_vols = [tag for dim, tag in cut_pile_tags]

    # Cut the pile in half with the symmetry cutter plane
    cut_pile_tags, _ = gmsh.model.occ.cut(
        [[3, tag] for tag in pile_vols],
        [[3, symmetry_cutter_tag]],
        -1,
        removeObject=True,
        removeTool=True,
    )
    pile_vols = [tag for dim, tag in cut_pile_tags]
    if len(pile_vols) != 1:
        raise ValueError(f"Pile not created correctly : {len(pile_vols)}")
    
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    
    
    
    soil_surface_tags = cm.SurfaceTags()
    for soil_box in cut_soil_boxes:
        surface_data = get_surface_extremes(soil_box)
        update_surface_tags(soil_surface_tags, surface_data)

    pile_surface_tags = cm.SurfaceTags()
    pile_surface_tags = cm.SurfaceTags()
    pile_surface_data = get_surface_extremes(pile_vols[0])
    update_surface_tags(pile_surface_tags, pile_surface_data)

    geometry_tag_manager = cm.GeometryTagManager(
        soil_volumes=cut_soil_boxes,
        pile_volumes=pile_vols,
        soil_surfaces=soil_surface_tags,
        pile_surfaces=pile_surface_tags,
    )
    gmsh.model.occ.synchronize()
    
    return geometry_tag_manager



def get_surface_extremes(volume: int) -> cm.SurfaceTags:
    surfaceLoopTags, surfaceTags = gmsh.model.occ.getSurfaceLoops(volume)
    surfaceTags = list(surfaceTags[0])
    
    extremes = cm.SurfaceTags()

    # Define a small tolerance for floating-point comparison
    tolerance = 1e-5
    
    for surface in surfaceTags:
        x, y, z = gmsh.model.occ.getCenterOfMass(2, surface)
        
        # Update min_x
        if x < extremes.min_x - tolerance:
            extremes.min_x = x
            extremes.min_x_surfaces = [surface]  # Reset the list
        elif math.isclose(x, extremes.min_x, abs_tol=tolerance):
            extremes.min_x_surfaces.append(surface)  # Append to the list

        # Update max_x
        if x > extremes.max_x + tolerance:
            extremes.max_x = x
            extremes.max_x_surfaces = [surface]  # Reset the list
        elif math.isclose(x, extremes.max_x, abs_tol=tolerance):
            extremes.max_x_surfaces.append(surface)  # Append to the list

        # Update min_y
        if y < extremes.min_y - tolerance:
            extremes.min_y = y
            extremes.min_y_surfaces = [surface]  # Reset the list
        elif math.isclose(y, extremes.min_y, abs_tol=tolerance):
            extremes.min_y_surfaces.append(surface)  # Append to the list

        # Update max_y
        if y > extremes.max_y + tolerance:
            extremes.max_y = y
            extremes.max_y_surfaces = [surface]  # Reset the list
        elif math.isclose(y, extremes.max_y, abs_tol=tolerance):
            extremes.max_y_surfaces.append(surface)  # Append to the list

        # Update min_z
        if z < extremes.min_z - tolerance:
            extremes.min_z = z
            extremes.min_z_surfaces = [surface]  # Reset the list
        elif math.isclose(z, extremes.min_z, abs_tol=tolerance):
            extremes.min_z_surfaces.append(surface)  # Append to the list

        # Update max_z
        if z > extremes.max_z + tolerance:
            extremes.max_z = z
            extremes.max_z_surfaces = [surface]  # Reset the list
        elif math.isclose(z, extremes.max_z, abs_tol=tolerance):
            extremes.max_z_surfaces.append(surface)  # Append to the list

    return extremes

def update_surface_tags(global_tags: cm.SurfaceTags, surface_data: cm.SurfaceTags):
    global_tags.min_x_surfaces = [*global_tags.min_x_surfaces,  *surface_data.min_x_surfaces]
    global_tags.max_x_surfaces = [*global_tags.max_x_surfaces, *surface_data.max_x_surfaces]
    global_tags.min_y_surfaces = [*global_tags.min_y_surfaces, *surface_data.min_y_surfaces]
    global_tags.max_y_surfaces = [*global_tags.max_y_surfaces, *surface_data.max_y_surfaces]

    if surface_data.min_z < global_tags.min_z:
        global_tags.min_z_surfaces = surface_data.min_z_surfaces
        global_tags.min_z = surface_data.min_z
    if surface_data.max_z > global_tags.max_z:
        global_tags.max_z_surfaces = surface_data.max_z_surfaces
        global_tags.max_z = surface_data.max_z

@ut.track_time("ADDING PHYSICAL GROUPS TO MESH")
def add_physical_groups(params, geo: cm.GeometryTagManager) -> List[cm.PhysicalGroup]:
    """
    Adds physical groups to the mesh and generates the mesh using Gmsh.
    
    Args:
        params: Parameters including file paths and mode.
        geo (cm.GeometryTagManager): GeometryTagManager containing tags for different volumes and surfaces.
        
    Returns:
        List[cm.PhysicalGroup]: List of physical groups added to the mesh.
    """
    physical_groups = []

    for i in range(1, 5):
        physical_groups.append(cm.PhysicalGroup(
            dim=3, tags=[geo.soil_volumes[i-1]], name=f"SOIL_LAYER_{i}",
            group_type=cm.PhysicalGroupType.MATERIAL, props={cm.PropertyTypeEnum.elastic: params.box_manager.layers[i].linear_elastic_properties}
        ))
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=geo.pile_volumes, name="CYLINDER",
        group_type=cm.PhysicalGroupType.MATERIAL, props={cm.PropertyTypeEnum.elastic: params.pile_manager.linear_elastic_properties},gmsh_tag=5,
    ))

    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.max_x_surfaces, name="FIX_ALL",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # LEFT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_x_surfaces, name="FIX_X_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # RIGHT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_z_surfaces, name="FIX_Z_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # BOTTOM FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=[*geo.soil_surfaces.min_y_surfaces, *geo.soil_surfaces.max_y_surfaces, 
                     *geo.pile_surfaces.max_y_surfaces,
                     ], name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # BACK AND FRONT FACE OF SOIL
    
    if getattr(params, 'prescribed_force', None):
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FORCE",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_force,
        ))  # TOP FACE OF PILE
    elif getattr(params, 'prescribed_disp', None):
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FIX_X_1",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
        ))  # TOP FACE OF PILE
        
        
    # Adding physical groups to Gmsh model
    physical_group_dimtag = {}
    for group in physical_groups:
        physical_group_dimtag[group.name] = (group.dim, gmsh.model.addPhysicalGroup(
            dim=group.dim,
            tags=group.tags,
            name=group.name,
        ))
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.min_x, params.box_manager.min_y, params.box_manager.min_z, params.box_manager.max_x, params.box_manager.max_y, params.box_manager.max_z), params.box_manager.far_field_size)
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.near_field_min_x, params.box_manager.near_field_min_y, params.box_manager.near_field_min_z, params.box_manager.near_field_max_x, params.box_manager.near_field_max_y, params.box_manager.near_field_max_z), params.box_manager.near_field_size)
    # Setting Gmsh options and generating mesh
    try:
        # gmsh.model.occ.synchronize()
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
    except Exception as e:
        print(f"An error occurred during mesh generation: {e}")
        gmsh.write(params.med_filepath.as_posix())
        
    finally:
        gmsh.finalize()

    return physical_groups


def parse_read_med(params):
    """
    Parse the output log from read_med to extract meshset information.

    Args:
        params: Parameters including file paths and other necessary configurations.

    Returns:
        List[cm.MeshsetInfo]: List of MeshsetInfo objects.
    """
    meshsets = []
    meshset_pattern = re.compile(r'\[read_med\] meshset .* msId (\d+) name (\w+)')
    try:
        with open(params.read_med_initial_log_file, 'r') as log_file:
            for line in log_file:
                match = meshset_pattern.search(line.strip())
                if match:
                    meshset_id = int(match.group(1))
                    name = match.group(2)
                    meshsets.append(cm.MeshsetInfo(meshset_id=meshset_id, name=name))
    except IOError as e:
        print(f"Error reading log file: {e}")
    
    return meshsets

@ut.track_time("CHECKING BLOCK IDS")
def check_block_ids(params, physical_groups: List[cm.PhysicalGroup]) -> List[cm.PhysicalGroup]:
    """
    Checks and updates the meshnet IDs of physical groups based on the provided parameters.

    Args:
        params: Parameters including file paths and other necessary configurations.
        physical_groups (List[cm.PhysicalGroup]): List of physical groups to update.

    Returns:
        List[cm.PhysicalGroup]: Updated list of physical groups with corresponding meshnet IDs.
    """

    try:
        with open(params.read_med_initial_log_file, 'w') as log_file:
            subprocess.run(
                ["read_med", "-med_file", params.med_filepath.as_posix()],
                stdout=log_file,
                stderr=log_file,
                check=True
            )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running read_med: {e}")
        return physical_groups
    except IOError as e:
        print(f"An error occurred while writing to the log file: {e}")
        return physical_groups

    meshnets = parse_read_med(params)

    # Create a dictionary mapping name to meshnet_id
    meshnet_mapping = {meshnet.name: meshnet.meshset_id for meshnet in meshnets}

    # Update physical_groups with the corresponding meshnet_id
    for group in physical_groups:
        if group.name in meshnet_mapping:
            group.meshnet_id = meshnet_mapping[group.name]

    return physical_groups

@ut.track_time("GENERATING CONFIG FILES")
def generate_config(params, physical_groups: List[cm.PhysicalGroup]):
    blocks: list[BC_CONFIG_BLOCK | cm.MFRONT_CONFIG_BLOCK] = []
    new_physical_groups: List[cm.PhysicalGroup] = []
    for i in range(len(physical_groups)):
        if physical_groups[i].group_type == cm.PhysicalGroupType.BOUNDARY_CONDITION:
            print(physical_groups[i].bc.dict())
            blocks.append(cm.BC_CONFIG_BLOCK(
                block_name = f"SET_ATTR_{physical_groups[i].name}",
                comment = f"Boundary condition for {physical_groups[i].name}",
                id = physical_groups[i].meshnet_id,
                attributes = list(physical_groups[i].bc.dict().values()),
            ))
            
        #very messy!
        elif physical_groups[i].group_type == cm.PhysicalGroupType.MATERIAL:
            print(physical_groups[i])
            new_physical_group = physical_groups[i].model_copy(deep=True, update={'meshnet_id': physical_groups[i].meshnet_id + 100, 'name': f"MFRONT_MAT_{physical_groups[i].meshnet_id}"})
            new_physical_groups.append(new_physical_group)
            blocks.append(cm.MFRONT_CONFIG_BLOCK(
                block_name = f"block_{physical_groups[i].meshnet_id}",
                id = new_physical_group.meshnet_id,
                comment = f"Material properties for {physical_groups[i].name}",
                name = new_physical_group.name,
            ))
            
    with open(params.config_file, "w") as f:
        for i in range(len(blocks)):
            f.writelines(blocks[i].formatted())
            
    return physical_groups + new_physical_groups
        
@ut.track_time("INJECTING CONFIG FILE with read_med")
def inject_configs(params):
    """
    Inject boundary conditions from a .cfg file into a .med file and convert to a .h5m file.

    Args:
        params: Parameters including file paths and other necessary configurations.
    """
    try:
        subprocess.run(
            ["read_med", 
             "-med_file", f"{params.med_filepath}", 
             "-output_file", f"{params.h5m_filepath}", 
             "-meshsets_config", f"{params.config_file}", #remember it is meshsets not meshnets
             "-log_sl" "inform"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error injecting configs: {e}")


@ut.track_time("PARTITIONING MESH with mofem_part")
def partition_mesh(params):
    try:
        subprocess.run(
                    [
                f'{params.um_view}/bin/mofem_part', 
                '-my_file', f'{params.h5m_filepath}',
                '-my_nparts', f'{params.nproc}',
                '-output_file', f'{params.part_file}',
                '-dim', f'3',
                '-adj_dim', f'2',
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error partitioning mesh: {e}")

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

@ut.track_time("COMPUTING")
def mofem_compute(params):
    result = subprocess.run("rm -rf out*", shell=True, text=True)
    # !rm -rf out*
    replace_template_sdf(params)
    
    mfront_arguments = []
    for physical_group in params.physical_groups:
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

    result = subprocess.run(command, shell=True, text=True)

@ut.track_time("CONVERTING FROM .htm TO .vtk")
def export_to_vtk(params):
    out_to_vtk = subprocess.run("ls -c1 out_*h5m", shell=True, text=True, capture_output=True)
    # Check if the command was successful
    if out_to_vtk.returncode == 0:
        # Split the output by lines to get the list of files
        list_of_files = out_to_vtk.stdout.splitlines()

        # Print the list of .h5m files
        print(f"List of .h5m files: {list_of_files}")

        # Get the last file from the list
        last_file = list_of_files[0]

        # Run mbconvert with the last file
        subprocess.run(f"mbconvert {last_file} {params.vtk_filepath}", shell=True, text=True)
    else:
        print("Error: Could not list files.")



def get_meshset_by_name(meshsets: List[cm.MeshsetInfo], name: str) -> Optional[cm.MeshsetInfo]:
    """
    Get a MeshsetInfo object by name from a list of MeshsetInfo objects.

    Args:
        meshsets (List[cm.MeshsetInfo]): List of MeshsetInfo objects.
        name (str): Name of the meshset to find.

    Returns:
        Optional[cm.MeshsetInfo]: MeshsetInfo object if found, otherwise None.
    """
    for meshset in meshsets:
        if meshset.name == name:
            return meshset
    return None

def parse_log_file(params):
    """
    Parse the log file to extract result variables.

    Args:
        params: Parameters including the log file path.

    Returns:
        AttrDict: Dictionary-like object containing parsed result variables.
    """
    res = AttrDict()
    try:
        with open(params.log_file, 'r') as log_file:
            for line in log_file:
                line = line.strip()
                if "nb global dofs" in line:
                    res.elem_num = int(line.split()[13])
                if "error L2 norm" in line:
                    res.err_l2_norm = float(line.split()[7])
                if "error H1 seminorm" in line:
                    res.err_h1_snorm = float(line.split()[7])
                if "error indicator" in line:
                    res.err_indic_tot = float(line.split()[6])
    except IOError as e:
        print(f"Error reading log file: {e}")
    
    return res


@ut.track_time("CLEARING MAIN LOG FILE BEFORE RUNNING MOFEM")
def clear_main_log(params):
    open(params.log_file, 'w').close() #clear log file
      
def test():
    print('Imported modules')
# if __name__ == "__main__":
#     params = AttrDict()