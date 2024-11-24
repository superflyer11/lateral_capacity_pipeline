import math
import re
import subprocess
from typing import List, Optional, Union

import gmsh

import custom_models as cm
import utils as ut

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

def update_surface_tags(global_tags: cm.SurfaceTags, surface_data: cm.SurfaceTags) -> cm.SurfaceTags:
    global_tags.min_x_surfaces = [*global_tags.min_x_surfaces, *surface_data.min_x_surfaces]
    global_tags.max_x_surfaces = [*global_tags.max_x_surfaces, *surface_data.max_x_surfaces]
    global_tags.min_y_surfaces = [*global_tags.min_y_surfaces, *surface_data.min_y_surfaces]
    global_tags.max_y_surfaces = [*global_tags.max_y_surfaces, *surface_data.max_y_surfaces]

    if surface_data.min_z < global_tags.min_z:
        global_tags.min_z_surfaces = surface_data.min_z_surfaces
        global_tags.min_z = surface_data.min_z
    if surface_data.max_z > global_tags.max_z:
        global_tags.max_z_surfaces = surface_data.max_z_surfaces
        global_tags.max_z = surface_data.max_z
        
    return global_tags

#this is not used
def get_edge_extremes(volume: int) -> cm.CurveTags:
    curveLoopTags, curveTags = gmsh.model.occ.getCurveLoops(volume)
    curveTags = list(curveTags[0])
    
    extremes = cm.CurveTags()

    # Define a small tolerance for floating-point comparison
    tolerance = 1e-5
    
    for curve in curveTags:
        print(curve)
        x, y, z = gmsh.model.occ.getCenterOfMass(1, curve)
        
        # Update min_x
        if x < extremes.min_x - tolerance:
            extremes.min_x = x
            extremes.min_x_curves = [curve]  # Reset the list
        elif math.isclose(x, extremes.min_x, abs_tol=tolerance):
            extremes.min_x_curves.append(curve)  # Append to the list

        # Update max_x
        if x > extremes.max_x + tolerance:
            extremes.max_x = x
            extremes.max_x_curves = [curve]  # Reset the list
        elif math.isclose(x, extremes.max_x, abs_tol=tolerance):
            extremes.max_x_curves.append(curve)  # Append to the list

        # Update min_y
        if y < extremes.min_y - tolerance:
            extremes.min_y = y
            extremes.min_y_curves = [curve]  # Reset the list
        elif math.isclose(y, extremes.min_y, abs_tol=tolerance):
            extremes.min_y_curves.append(curve)  # Append to the list

        # Update max_y
        if y > extremes.max_y + tolerance:
            extremes.max_y = y
            extremes.max_y_curves = [curve]  # Reset the list
        elif math.isclose(y, extremes.max_y, abs_tol=tolerance):
            extremes.max_y_curves.append(curve)  # Append to the list

        # Update min_z
        if z < extremes.min_z - tolerance:
            extremes.min_z = z
            extremes.min_z_curves = [curve]  # Reset the list
        elif math.isclose(z, extremes.min_z, abs_tol=tolerance):
            extremes.min_z_curves.append(curve)  # Append to the list

        # Update max_z
        if z > extremes.max_z + tolerance:
            extremes.max_z = z
            extremes.max_z_curves = [curve]  # Reset the list
        elif math.isclose(z, extremes.max_z, abs_tol=tolerance):
            extremes.max_z_curves.append(curve)  # Append to the list

    return extremes

def update_curve_tags(global_tags: cm.CurveTags, curve_data: cm.CurveTags) -> cm.CurveTags:
    global_tags.min_x_curves = [*global_tags.min_x_curves,  *curve_data.min_x_curves]
    global_tags.max_x_curves = [*global_tags.max_x_curves, *curve_data.max_x_curves]
    global_tags.min_y_curves = [*global_tags.min_y_curves, *curve_data.min_y_curves]
    global_tags.max_y_curves = [*global_tags.max_y_curves, *curve_data.max_y_curves]

    if curve_data.min_z < global_tags.min_z:
        global_tags.min_z_curves = curve_data.min_z_curves
        global_tags.min_z = curve_data.min_z
    if curve_data.max_z > global_tags.max_z:
        global_tags.max_z_curves = curve_data.max_z_curves
        global_tags.max_z = curve_data.max_z
        
    return global_tags

#only works globally
def get_global_surface_extremes_2D() -> cm.SurfaceTags:
    surfaceData = gmsh.model.occ.getEntities(2)
    extremes = cm.SurfaceTags()
    # Define a small tolerance for floating-point comparison
    tolerance = 1e-5
    
    for surface in surfaceData:
        surface = surface[1]
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

#only works globally
def get_global_edge_extremes_2D() -> cm.CurveTags:
    edgeData = gmsh.model.occ.getEntities(1)
    extremes = cm.CurveTags()
    # Define a small tolerance for floating-point comparison
    tolerance = 1e-5
    
    for curve in edgeData:
        curve = curve[1]
        x, y, z = gmsh.model.occ.getCenterOfMass(1, curve)
        
        # Update min_x
        if x < extremes.min_x - tolerance:
            extremes.min_x = x
            extremes.min_x_curves = [curve]  # Reset the list
        elif math.isclose(x, extremes.min_x, abs_tol=tolerance):
            extremes.min_x_curves.append(curve)  # Append to the list

        # Update max_x
        if x > extremes.max_x + tolerance:
            extremes.max_x = x
            extremes.max_x_curves = [curve]  # Reset the list
        elif math.isclose(x, extremes.max_x, abs_tol=tolerance):
            extremes.max_x_curves.append(curve)  # Append to the list

        # Update min_y
        if y < extremes.min_y - tolerance:
            extremes.min_y = y
            extremes.min_y_curves = [curve]  # Reset the list
        elif math.isclose(y, extremes.min_y, abs_tol=tolerance):
            extremes.min_y_curves.append(curve)  # Append to the list

        # Update max_y
        if y > extremes.max_y + tolerance:
            extremes.max_y = y
            extremes.max_y_curves = [curve]  # Reset the list
        elif math.isclose(y, extremes.max_y, abs_tol=tolerance):
            extremes.max_y_curves.append(curve)  # Append to the list

        # Update min_z
        if z < extremes.min_z - tolerance:
            extremes.min_z = z
            extremes.min_z_curves = [curve]  # Reset the list
        elif math.isclose(z, extremes.min_z, abs_tol=tolerance):
            extremes.min_z_curves.append(curve)  # Append to the list

        # Update max_z
        if z > extremes.max_z + tolerance:
            extremes.max_z = z
            extremes.max_z_curves = [curve]  # Reset the list
        elif math.isclose(z, extremes.max_z, abs_tol=tolerance):
            extremes.max_z_curves.append(curve)  # Append to the list

    return extremes

# can only be used after mesh.generate
def get_global_node_extremes_3D() -> cm.NodeTags3D:
    nodeTags, _, __ = gmsh.model.mesh.getNodes(0)
    extremes = cm.NodeTags3D()
    tolerance = 1e-5
    
    for nodeTag in nodeTags:
        coords, _, __, ___ = gmsh.model.mesh.getNode(nodeTag)
        x, y, z = coords

        # Update based on z -> y -> x ordering for non-convex mesh
        if z <= extremes.min_z + tolerance:
            extremes.min_z = z
            if y <= extremes.min_y + tolerance:
                extremes.min_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_min_y_min_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_min_y_min_z_node = nodeTag
            elif y >= extremes.max_y - tolerance:
                extremes.max_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_max_y_min_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_max_y_min_z_node = nodeTag

        elif z >= extremes.max_z - tolerance:
            extremes.max_z = z
            if y <= extremes.min_y + tolerance:
                extremes.min_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_min_y_max_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_min_y_max_z_node = nodeTag
            elif y >= extremes.max_y - tolerance:
                extremes.max_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_max_y_max_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_max_y_max_z_node = nodeTag

    return extremes

def get_node_extremes_3D(array) -> cm.NodeTags3D:
    nodeDimTags = []
    # print(volTags)
    # for _, volTag in volTags:
    #     xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, volTag)
    #     print(gmsh.model.occ.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, 0))
    #     nodeDimTags = [*nodeDimTags, *gmsh.model.occ.getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, 0)]
    #     # print(nodeDimTags)
    # # nodeTags, _, __ = gmsh.model.mesh.getNodes(includeBoundary=True)
    # # print(nodeTags)
    extremes = cm.NodeTags3D()
    tolerance = 0
    for count, nodeTag in enumerate(array[0]):
        # coords, _, __, ___ = gmsh.model.mesh.getNode(nodeDimTag[1])
        x = array[1][3*count]
        y = array[1][3*count+1]
        z = array[1][3*count+2]
        print(f"{nodeTag}")
        print(f"{x} {y} {z}")
        # Update based on z -> y -> x ordering for non-convex mesh
        if z <= extremes.min_z + tolerance:
            extremes.min_z = z
            if y <= extremes.min_y + tolerance:
                extremes.min_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_min_y_min_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_min_y_min_z_node = nodeTag
            elif y >= extremes.max_y - tolerance:
                extremes.max_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_max_y_min_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_max_y_min_z_node = nodeTag

        elif z >= extremes.max_z - tolerance:
            extremes.max_z = z
            if y <= extremes.min_y + tolerance:
                extremes.min_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_min_y_max_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_min_y_max_z_node = nodeTag
            elif y >= extremes.max_y - tolerance:
                extremes.max_y = y
                if x <= extremes.min_x + tolerance:
                    extremes.min_x = x
                    extremes.min_x_max_y_max_z_node = nodeTag
                elif x >= extremes.max_x - tolerance:
                    extremes.max_x = x
                    extremes.max_x_max_y_max_z_node = nodeTag

    return extremes

#only works globally
def get_global_node_extremes_2D(test_surface) -> cm.NodeTags2D:
    nodeData = gmsh.model.occ.getEntities(0)
    # boundary = gmsh.model.getBoundary([(2,test_surface)], recursive=True) this may also work
    extremes = cm.NodeTags2D()
    # Define a small tolerance to compare
    tolerance = 1e-5
    
    for data in nodeData:
        coords, _, __, ___ = gmsh.model.mesh.getNode(data[1])
        x, y, z = coords
        # Update min_x
        if x <= extremes.min_x:
            extremes.min_x = x
            if y <= extremes.min_y:
                extremes.min_y = y
                extremes.min_x_min_y_node = data[1]
                continue
            elif y >= extremes.max_y:
                extremes.max_y = y
                extremes.min_x_max_y_node = data[1]
                continue
        elif x >= extremes.max_x:
            extremes.max_x = x
            if y <= extremes.min_y:
                extremes.min_y = y
                extremes.max_x_min_y_node = data[1]
                continue
            elif y >= extremes.max_y:
                extremes.max_y = y
                extremes.max_x_max_y_node = data[1]
                continue
    return extremes

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
                [params.read_med_exe, "-med_file", params.med_filepath.as_posix()],
                stdout=log_file,
                stderr=log_file,
                check=True
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"An error occurred while running read_med: {e}")
        return physical_groups
    except IOError as e:
        raise RuntimeError(f"An error occurred while writing to the log file: {e}")
        return physical_groups

    meshnets = parse_read_med(params)

    # Create a dictionary mapping name to meshnet_id
    meshnet_mapping = {meshnet.name: meshnet.meshset_id for meshnet in meshnets}
    print(meshnet_mapping)
    # Update physical_groups with the corresponding meshnet_id
    for group in physical_groups:
        if group.name in meshnet_mapping:
            group.meshnet_id = meshnet_mapping[group.name]

    return physical_groups

@ut.track_time("GENERATING CONFIG FILES")
def generate_config(params, physical_groups: List[cm.PhysicalGroup]):
    if getattr(params, "time_history", False):
        params.time_history.write(params.time_history_file)
    print(physical_groups)
    blocks: list[cm.BC_CONFIG_BLOCK | cm.MFRONT_CONFIG_BLOCK] = []
    new_physical_groups: List[cm.PhysicalGroup] = []
    for i in range(len(physical_groups)):
        if physical_groups[i].group_type == cm.PhysicalGroupType.BOUNDARY_CONDITION:
            scale = physical_groups[i].bc.replace_dict_with_value() # replace the dict and returns the dict
            if scale:
                raise NotImplementedError("Inputting the scale directly will have issues with positive and negative signs, directly use params.time_history instead")
                sys.exit()
                # scale.write(params.time_history_file)
            blocks.append(cm.BC_CONFIG_BLOCK(
                block_name = f"SET_ATTR_{physical_groups[i].name}",
                comment = f"Boundary condition for {physical_groups[i].name}",
                id = physical_groups[i].meshnet_id,
                attributes = list(physical_groups[i].bc.model_dump(exclude_none=True).values()),
            ))

            
        elif physical_groups[i].group_type == cm.PhysicalGroupType.MATERIAL:
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
            [params.read_med_exe, 
             "-med_file", f"{params.med_filepath}", 
             "-output_file", f"{params.h5m_filepath}", 
             "-meshsets_config", f"{params.config_file}", #remember it is meshsets not meshnets
             "-log_sl" "inform"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error injecting configs: {e}")

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
    res = cm.AttrDict()
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