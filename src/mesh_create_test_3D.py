import os
import sys
import math
import re
import time
import subprocess
from pathlib import Path
import shutil
from functools import wraps
from typing import List, Optional, Union
from warnings import warn

import gmsh
import pyvista as pv

import mesh_create_common as mshcrte_common
import custom_models as cm
import utils as ut


@ut.track_time("DRAWING MESH")
def draw_mesh(params) -> cm.TestTagManager:
    """
    Draws the mesh using Gmsh based on the provided parameters.

    Args:
        params: Parameters including mesh name and other necessary configurations.

    Returns:
        cm.GeometryTagManager: An object containing tags for soil and pile volumes and surfaces.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)

    gmsh.model.add(f"{params.mesh_name_appended}")

    test_volume = gmsh.model.occ.addBox(-params.mesh_size/2,-params.mesh_size/2,-params.mesh_size/2,params.mesh_size,params.mesh_size,params.mesh_size)

    test_surface_tags = cm.SurfaceTags()
    test_surface_data = mshcrte_common.get_surface_extremes(test_volume)
    test_surface_tags = mshcrte_common.update_surface_tags(test_surface_tags, test_surface_data)
    
    test_tag_manager = cm.TestTagManager(
        test_volume=[test_volume],
        test_surfaces=test_surface_tags,
    )
    
    gmsh.model.occ.synchronize()
    
    return test_tag_manager


@ut.track_time("ADDING PHYSICAL GROUPS TO MESH")
def add_physical_groups(params, geo: cm.TestTagManager) -> List[cm.PhysicalGroup]:
    physical_groups = []
    
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=geo.test_volume, name=params.case_name,
            preferred_model = params.tester.preferred_model,
        group_type=cm.PhysicalGroupType.MATERIAL, props=params.tester.props,
    ))
    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=2, 
        tags=geo.test_surfaces.min_z_surfaces, 
        name="FIX_Z_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=None,disp_uy=None,disp_uz=0)
    )) 
    physical_groups.append(cm.PhysicalGroup(
        dim=2, 
        tags=geo.test_surfaces.min_x_surfaces, 
        name="FIX_X_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=None,disp_uz=None)
    ))  
    physical_groups.append(cm.PhysicalGroup(
        dim=2, 
        tags=geo.test_surfaces.min_y_surfaces, 
        name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=None,disp_uy=0,disp_uz=None)
    ))  
    
    if getattr(params, 'prescribed_force', None):
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=[
                *geo.test_surfaces.max_x_surfaces
                ], name="FORCE_1",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.ForceBoundaryCondition(f_x=-50, f_y=0, f_z=0)
        ))  
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=[
                *geo.test_surfaces.max_y_surfaces
                ], name="FORCE_2",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.ForceBoundaryCondition(f_x=0, f_y=-50, f_z=0)
        ))  
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=geo.test_surfaces.max_z_surfaces, name="FORCE",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_force,
        ))  # TOP FACE OF PILE
    elif getattr(params, 'prescribed_disp', None):
        # physical_groups.append(cm.PhysicalGroup(
        #     dim=2, tags=[
        #         *geo.test_surfaces.max_x_surfaces
        #         ], name="FIX_X_1",
        #     group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=-0.3,disp_uy=None,disp_uz=None)
        # ))  
        # physical_groups.append(cm.PhysicalGroup(
        #     dim=2, tags=[
        #         *geo.test_surfaces.max_y_surfaces
        #         ], name="FIX_Y_1",
        #     group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=None,disp_uy=-0.3,disp_uz=None)
        # ))  
        if getattr(params.prescribed_disp, 'disp_ux', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.test_surfaces.max_z_surfaces, name="FIX_X_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uy', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.test_surfaces.max_z_surfaces, name="FIX_Y_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uz', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.test_surfaces.max_z_surfaces, name="FIX_Z_1",
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
    # Setting Gmsh options and generating mesh
    try:
        # gmsh.option.setNumber("Mesh.MeshSizeMax", params.mesh_size/4)
        # gmsh.option.setNumber("Mesh.MeshSizeMin", params.mesh_size/4)
        # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 36)
        for volTag in gmsh.model.getEntitiesForPhysicalGroup(3,geo.test_volume[0]):
            _, surfaceTags = gmsh.model.occ.getSurfaceLoops(volTag)
            for surfaceTag in surfaceTags[0]:
                _, curveTags = gmsh.model.occ.getCurveLoops(surfaceTag)
                for curveTag in curveTags[0]:
                    # length = gmsh.model.occ.getMass(1, curveTag)
                    gmsh.model.mesh.setTransfiniteCurve(curveTag, 4+1)
        surfaces = gmsh.model.getEntities(2)
        for _, tag in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(tag)
        volumes = gmsh.model.getEntities(3)
        for _, tag in volumes:
            gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.model.mesh.recombine()
        # gmsh.option.setNumber("Mesh.SaveAll", 1)

        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
    except Exception as e:
        raise RuntimeError(f"An error occurred during mesh generation: {e}")
        # gmsh.write(params.med_filepath.as_posix())
        
    finally:
        gmsh.finalize()

    return physical_groups