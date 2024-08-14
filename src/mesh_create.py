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
class AttrDict():
    def _getattr(self,attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"Attribute {attr} not found")
params = AttrDict()
mode = cm.PropertyTypeEnum.elastic



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
    for no, layer in params.properties.layers.items():
        soil_box = cm.BoxLayer(dz=layer.depth)
        new_tag = params.box_manager.add_layer(soil_box)
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
    
    for surface in surfaceTags:
        x, y, z = gmsh.model.occ.getCenterOfMass(2, surface)
        
        # Update min_x
        if x < extremes.min_x:
            extremes.min_x = x
            extremes.min_x_surfaces = [surface]  # Reset the list
        elif x == extremes.min_x:
            extremes.min_x_surfaces.append(surface)  # Add to the list

        # Update max_x
        if x > extremes.max_x:
            extremes.max_x = x
            extremes.max_x_surfaces = [surface]  # Reset the list
        elif x == extremes.max_x:
            extremes.max_x_surfaces.append(surface)  # Add to the list

        # Update min_y
        if y < extremes.min_y:
            extremes.min_y = y
            extremes.min_y_surfaces = [surface]  # Reset the list
        elif y == extremes.min_y:
            extremes.min_y_surfaces.append(surface)  # Add to the list

        # Update max_y
        if y > extremes.max_y:
            extremes.max_y = y
            extremes.max_y_surfaces = [surface]  # Reset the list
        elif y == extremes.max_y:
            extremes.max_y_surfaces.append(surface)  # Add to the list

        # Update min_z
        if z < extremes.min_z:
            extremes.min_z = z
            extremes.min_z_surfaces = [surface]  # Reset the list
        elif z == extremes.min_z:
            extremes.min_z_surfaces.append(surface)  # Add to the list

        # Update max_z
        if z > extremes.max_z:
            extremes.max_z = z
            extremes.max_z_surfaces = [surface]  # Reset the list
        elif z == extremes.max_z:
            extremes.max_z_surfaces.append(surface)  # Add to the list

    return extremes
    
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
            dim=3, tags=[geo.soil_volumes[i-1]], name=f"SOIL_LAYER_1",
            group_type=cm.PhysicalGroupType.MATERIAL, props={cm.PropertyTypeEnum.elastic: params.properties.layers[i].elastic_properties}
        ))
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=geo.pile_volumes, name="CYLINDER",
        group_type=cm.PhysicalGroupType.MATERIAL, props={cm.PropertyTypeEnum.elastic: params.pile_manager.elastic_properties}
    ))

    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_x_surfaces, name="FIX_ALL",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # LEFT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.max_x_surfaces, name="FIX_X_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # RIGHT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_z_surfaces, name="FIX_Z_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # BOTTOM FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=[*geo.soil_surfaces.min_y_surfaces, *geo.soil_surfaces.max_y_surfaces], name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition()
    ))  # BACK AND FRONT FACE OF SOIL
    

    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FIX_X_1",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(f)
    ))  # TOP FACE OF PILE
    # Adding physical groups to Gmsh model
    physical_group_dimtag = {}
    for group in physical_groups:
        physical_group_dimtag[group.name] = (group.dim, gmsh.model.addPhysicalGroup(
            dim=group.dim,
            tags=group.tags,
            name=group.name
        ))
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.min_x, params.box_manager.min_y, params.box_manager.min_z, params.box_manager.max_x, params.box_manager.max_y, params.box_manager.max_z), params.box_manager.far_field_size)
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.near_field_min_x, params.box_manager.near_field_min_y, params.box_manager.near_field_min_z, params.box_manager.near_field_max_x, params.box_manager.near_field_max_y, params.box_manager.near_field_max_z), params.box_manager.near_field_size)
    # Setting Gmsh options and generating mesh
    try:
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)
        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
    except Exception as e:
        print(f"An error occurred during mesh generation: {e}")
        gmsh.write(params.med_filepath.as_posix())
        
    finally:
        gmsh.finalize()

    return physical_groups

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
    log_file_path = 'read_med_initial_log_file.log'

    try:
        with open(log_file_path, 'w') as log_file:
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
    blocks: list[cm.CFGBLOCK2] = []
    for i in range(len(physical_groups)):
        if physical_groups[i].group_type == cm.PhysicalGroupType.BOUNDARY_CONDITION:
            print(physical_groups[i].bc.dict())
            blocks.append(cm.CFGBLOCK2(
                name = f"block_{physical_groups[i].meshnet_id}",
                comment = f"Boundary condition for {physical_groups[i].name}",
                id = physical_groups[i].meshnet_id,
                attributes = physical_groups[i].bc.dict(),
            ))
        elif physical_groups[i].group_type == cm.PhysicalGroupType.MATERIAL:
            blocks.append(cm.CFGBLOCK2(
                name = f"block_{physical_groups[i].meshnet_id}",
                comment = f"Material properties for {physical_groups[i].name}",
                id = physical_groups[i].meshnet_id,
                # attributes = physical_groups[i].props[mode].dict(),
            ))
        

    with open(params.config_file, "w") as f:
        for i in range(len(blocks)):
            f.writelines(blocks[i].formatted())
        

@ut.track_time("INJECTING CONFIG FILE")
def inject_configs(params):
    """
    Inject boundary conditions from a .cfg file into a .med file and convert to a .h5m file.

    Args:
        params: Parameters including file paths and other necessary configurations.
    """
    try:
        subprocess.run(
            ["read_med", 
             "-med_file", params.med_filepath.as_posix(), 
             "-output_file", params.h5m_filepath.as_posix(), 
             "-meshnets_config", params.config_file.as_posix()],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error injecting configs: {e}")
        raise



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
    
    log_file_path = 'read_med_initial_log_file.log'
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                match = meshset_pattern.search(line.strip())
                if match:
                    meshset_id = int(match.group(1))
                    name = match.group(2)
                    meshsets.append(cm.MeshsetInfo(meshset_id=meshset_id, name=name))
    except IOError as e:
        print(f"Error reading log file: {e}")
    
    return meshsets

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
    # show_file = params.show_file
    # if os.path.exists(show_file):
    #     if os.path.isdir(show_file):
    #         # If it is a directory, use shutil.rmtree to remove it and its contents
    #         shutil.rmtree(show_file)
    #     else:
    #         # If it is a file, use os.remove to delete it
    #         os.remove(show_file)
    # else:
    #     print(f"The file or directory {show_file} does not exist")
    open(params.log_file, 'w').close() #clear log file
      
@ut.print_message_in_box("PARTITIONING MESH with mofem_part")
def partition_mesh(params):
    command_1 = [
        '/mofem_install/um_view/bin/mofem_part', 
        '-my_file', f'{params.h5m_filepath.as_posix()}',
        '-output_file', f'{params.h5m_filepath.as_posix()}',
        '-my_nparts', str(params.nproc),
    ]
    with open(params.log_file, 'a') as log_file:  # Open in append mode after clearing
        run_command(1, command_1, log_file)




def test():
    print('Hello world!')
# if __name__ == "__main__":
#     ############################################################################################################
#     # GENERATING MESH FILES
#     ############################################################################################################
#     geo = draw_mesh(params)
#     # geo = discretise_mesh(params)
#     physical_groups = add_physical_groups(params, geo)
#     ############################################################################################################
#     # GENERATING CONFIG FILE
#     ############################################################################################################
#     physical_groups = check_block_ids(params,physical_groups)
#     generate_config(params,physical_groups)
#     # sys.exit()
#     inject_configs(params)
#     ############################################################################################################
#     # RUNNING MOFEM
#     ############################################################################################################
#     clear_main_log(params)
#     partition_mesh(params)
#     set_environment_variables(params)
#     run_mofem(params)
