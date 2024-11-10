import os
import sys
import math
import re
import time
import subprocess
from pathlib import Path
from functools import wraps
from warnings import warn
from typing import List, Optional, Union

import gmsh
import pyvista as pv

import mesh_create_common as mshcrte_common
import custom_models as cm
import utils as ut

@ut.track_time("DRAWING MESH")
def draw_mesh(params) -> cm.TestTagManager:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)

    gmsh.model.add(f"{params.mesh_name_appended}")

    test_surface = gmsh.model.occ.addRectangle(-params.mesh_size/2,-params.mesh_size/2,0,params.mesh_size,params.mesh_size)
    gmsh.model.occ.synchronize()
    try:
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
    except Exception as e:
        print(f"An error occurred during mesh generation: {e}")
        gmsh.write(params.med_filepath.as_posix())

    test_curve_tags = cm.CurveTags()
    test_curve_data = mshcrte_common.get_edge_extremes(test_surface)
    test_curve_tags = mshcrte_common.update_curve_tags(test_curve_tags, test_curve_data)
    test_node_tags = mshcrte_common.get_global_node_extremes_2D(test_surface)
    test_tag_manager = cm.TestTagManager2D(
        test_surface=[test_surface],
        test_curves=test_curve_tags,
        test_nodes=test_node_tags,
    )
    
    return test_tag_manager

@ut.track_time("ADDING PHYSICAL GROUPS TO MESH")
def add_physical_groups(params, geo: cm.TestTagManager2D) -> List[cm.PhysicalGroup]:
    physical_groups = []
    
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.test_surface, name=params.case_name,
            preferred_model = params.tester.preferred_model,
        group_type=cm.PhysicalGroupType.MATERIAL, props=params.tester.props,
    ))
    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=1, tags=geo.test_curves.min_y_curves, name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.EdgeBoundaryCondition()
    )) 
    physical_groups.append(cm.PhysicalGroup(
        dim=0, tags=[*geo.test_curves.min_x_curves,*geo.test_curves.max_x_curves], name="FIX_Y_5",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.EdgeBoundaryCondition()
    )) 


    if getattr(params, 'prescribed_force', None):
        raise NotImplementedError('2024-09-18: I did not implement this')
    elif getattr(params, 'prescribed_disp', None):
        if getattr(params.prescribed_disp, 'disp_ux', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=1, tags=geo.test_curves.max_y_curves, name="FIX_X_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uy', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=1, tags=geo.test_curves.max_y_curves, name="FIX_Y_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uz', None):
            raise NotImplementedError('2024-09-18: This is a 2D problem disp_uz must be = None')

        
        
    # Adding physical groups to Gmsh model
    physical_group_dimtag = {}
    for group in physical_groups:
        physical_group_dimtag[group.name] = (group.dim, gmsh.model.addPhysicalGroup(
            dim=group.dim,
            tags=group.tags,
            name=group.name,
        ))
    try:
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
    except Exception as e:
        print(f"An error occurred during mesh generation: {e}")
        gmsh.write(params.med_filepath.as_posix())
    finally:
        gmsh.finalize()
    return physical_groups