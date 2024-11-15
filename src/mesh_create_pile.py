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

@ut.track_time("(MANUAL) DRAWING MESH ")
def draw_mesh_manual(params) -> cm.GeometryTagManager:
    

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add(f"{params.case_name}")
    gmsh.option.setNumber("General.Verbosity", 3)
    
    # Translation of .geo file 
    box1 = gmsh.model.occ.addBox(-80, -80, 0, 160, 160, -2) # First layer
    box2 = gmsh.model.occ.addBox(-80, -80, -2, 160, 160, -1.4) # Second layer
    box3 = gmsh.model.occ.addBox(-80, -80, -3.4, 160, 160, -7.1) # Third layer
    box4 = gmsh.model.occ.addBox(-80, -80, -10.5, 160, 160, -29.5) # Fourth layer
    cylinder1 = gmsh.model.occ.addCylinder(0, 0, -10.5, 0, 0, 20.5, 1, -1, 2*math.pi) # Two metre diameter pile
    cylinder2 = gmsh.model.occ.addCylinder(0, 0, -10.5, 0, 0, 20.5, 0.975, -1, 2*math.pi) # Hollow out pile
    box5 = gmsh.model.occ.addBox(-40, -40, 0, 80, 80, -20) # Area of refined mesh
    box6 = gmsh.model.occ.addBox(-20, -20, 0, 40, 40, -15) # Area of further refined mesh
    box7 = gmsh.model.occ.addBox(-200, 0, -200, 400, 400, 400) # Box for cutting domain in half
    
    gmsh.model.occ.cut([[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 7], [3, 8]], [[3, 6], [3, 9]], -1, True, True)
    gmsh.model.occ.fragment([[3, 16], [3, 4], [3, 8], [3, 12], [3, 3], [3, 7], [3, 11], [3, 15], [3, 2], 
                             [3, 6], [3, 10], [3, 14], [3, 1], [3, 5], [3, 9], [3, 13]], [], -1, True, False)
    
    # Create the relevant Gmsh data structures from Gmsh model.
    gmsh.model.occ.synchronize()
    
    
    # DISCRETISATION
    # PILE
    # Transfinite curves around pile - LONG LENGTHS
    for n in [150, 154, 151, 155, 115, 120, 117, 122]:
        gmsh.model.mesh.setTransfiniteCurve(n, 125,'Progression', 1.0)
                             
    # Transfinite curves around pile - SHORT LENGTHS
    for n in [82, 87, 80, 85, 42, 50, 52, 44]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)                        
            
    # Transfinite curves around pile - CIRLCES
    for n in [152, 157, 40, 48, 53, 45, 88, 83, 123, 118]:
        gmsh.model.mesh.setTransfiniteCurve(n, 50,'Progression', 1.0)
            
    # Transfinite curves around pile - BASE
    for n in [149]:
        gmsh.model.mesh.setTransfiniteCurve(n, 50,'Progression', 1.0)  
            
    # Transfinite curves around pile BELOW MUDLINE - THICKNESS 
    for n in [124, 121, 86, 89, 54, 51, 49, 47, 156, 153]:
        gmsh.model.mesh.setTransfiniteCurve(n, 1,'Progression', 1.0)  
    
    # Transfinite curves around pile BELOW MUDLINE - THICKNESS 
    for n in [156, 153]:
        gmsh.model.mesh.setTransfiniteCurve(n, 1,'Progression', 1.0) 
        
    # Transfinite surface around thickness
    for n in [72, 73, 70, 49, 47, 71, 67, 50, 48, 44, 27, 26, 24, 23, 25, 20]:
        gmsh.model.mesh.setTransfiniteSurface(n, 'Left')
            
    # Transfinite volume for pile
    for n in [4, 8, 12]:
        gmsh.model.mesh.setTransfiniteVolume(n)
    
    
    # SOIL DOMAIN 
    # Transfinite curves around soil domain - LONG HEIGHTS
    for n in [126, 125, 129, 134]:
        gmsh.model.mesh.setTransfiniteCurve(n, 20,'Progression', 1.0)
        
    # Transfinite curves around soil domain - SHORT HEIGHTS
    for n in [102, 67, 21, 94, 59, 6, 90, 55, 1, 91, 56, 3]:
        gmsh.model.mesh.setTransfiniteCurve(n, 5,'Progression', 1.0)
    
    # Transfinite curves around soil domain - WIDTH
    for n in [135, 101, 66, 20, 13, 127, 92, 57, 4, 2]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)
       
    # Transfinite curves around soil domain - LONG LENGTHS
    for n in [133, 128, 93, 58, 5, 7]:
        gmsh.model.mesh.setTransfiniteCurve(n, 50,'Progression', 1.0)
    
    # Transfinite curves around soil domain - SHORT LENGTHS
    for n in [100, 65, 19, 12, 96, 61, 15, 8]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)
        
        
    # BIG REFINEMENT BOX
    # Transfinite curves around soil domain - LONG HEIGHTS
    for n in [138, 104, 136, 103, 132, 95, 130, 105]:
        gmsh.model.mesh.setTransfiniteCurve(n, 20,'Progression', 1.0)
        
    # Transfinite curves around soil domain - SHORT HEIGHTS
    for n in [70, 24, 69, 23, 60, 14, 68, 22]:
        gmsh.model.mesh.setTransfiniteCurve(n, 5,'Progression', 1.0)
    
    # Transfinite curves around soil domain - WIDTH
    for n in [140, 99, 64, 18, 11, 137, 97, 62, 16, 9]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)
       
    # Transfinite curves around soil domain - LONG LENGTHS
    for n in [139, 98, 63, 17, 10, 131]:
        gmsh.model.mesh.setTransfiniteCurve(n, 50,'Progression', 1.0)
    
    # Transfinite curves around soil domain - SHORT LENGTHS
    for n in [107, 72, 31, 25, 114, 79, 38, 29]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)
        
        
    
    # SMALL REFINEMENT BOX
    # Transfinite curves around soil domain - LONG HEIGHTS
    for n in [141, 112, 146, 110, 143, 106, 144, 108]:
        gmsh.model.mesh.setTransfiniteCurve(n, 20,'Progression', 1.0)
        
    # Transfinite curves around soil domain - SHORT HEIGHTS
    for n in [34, 75, 36, 77, 32, 73, 71, 30]:
        gmsh.model.mesh.setTransfiniteCurve(n, 5,'Progression', 1.0)
    
    # Transfinite curves around soil domain - WIDTH
    for n in [148, 113, 78, 37, 28, 145, 109, 74, 33, 26]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)
       
    # Transfinite curves around soil domain - LONG LENGTHS
    for n in [142, 147, 111, 76, 35, 27]:
        gmsh.model.mesh.setTransfiniteCurve(n, 50,'Progression', 1.0)
    
    # Transfinite curves around soil domain - SHORT LENGTHS
    for n in [119, 116, 84, 81, 46, 43, 41, 39]:
        gmsh.model.mesh.setTransfiniteCurve(n, 25,'Progression', 1.0)

    # gmsh.model.addPhysicalGroup(dimention, [number of element], name="name")
    #gmsh.model.addPhysicalGroup(3, [box1, box2, box3, cylinder1, cylinder2, box4, box5, box6, box7], name="MAT_ELASTIC_")
    all = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    soil_1_vols = [1, 2, 3]
    soil_2_vols = [5, 6, 7]
    soil_3_vols = [9, 10, 11]
    soil_4_vols = [13, 14, 15]
    
    all_soil_vols = soil_1_vols + soil_2_vols + soil_3_vols + soil_4_vols
    
    cylinder_vols = [4,8,12,16]
    FIX_ALL = [78,55,32,6]
    FIX_Y_0 = [76, 59, 36, 10, 53, 30, 4, 83, 64, 41, 16, 60, 37, 12, 88, 68, 45, 21, 66, 43, 19, 72, 70, 49, 47, 24, 26, 91, 92, 75, 52, 29, 2]
    FIX_X_0 = [74, 28, 51, 1]
    FIX_Z_0 = [77]
    FIX_X_1 = [93]
    
    geometry_tag_manager = cm.ManualGeometryTagManager(
        soil_volumes=[soil_1_vols, soil_2_vols, soil_3_vols, soil_4_vols],
        pile_volumes=cylinder_vols,
        FIX_ALL=FIX_ALL,
        FIX_Y_0=FIX_Y_0,
        FIX_X_0=FIX_X_0,
        FIX_Z_0=FIX_Z_0,
        FIX_X_1=FIX_X_1,
    )
    
    
    # gmsh.model.addPhysicalGroup(3, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], name="MAT_ELASTIC_")
    gmsh.model.addPhysicalGroup(3, [1, 2, 3], name="SOIL_LAYER_0") # Volume of soil layer1
    gmsh.model.addPhysicalGroup(3, [5, 6, 7], name="SOIL_LAYER_1") # Volume of soil layer2
    gmsh.model.addPhysicalGroup(3, [9, 10, 11], name="SOIL_LAYER_2") # Volume of soil layer3
    gmsh.model.addPhysicalGroup(3, [13, 14, 15], name="SOIL_LAYER_3") # Volume of soil layer4
    gmsh.model.addPhysicalGroup(3, [4, 8, 12, 16], name="CYLINDER") # Volume of pile
    gmsh.model.addPhysicalGroup(2, [78, 55, 32, 6], name="FIX_ALL") # Boundary condition on plane
    gmsh.model.addPhysicalGroup(2, [76, 59, 36, 10, 53, 30, 4, 83, 64, 41, 16, 60, 37, 12, 88, 68, 45, 21, 66, 43, 19, 72, 70, 49, 47, 24, 26, 91, 92, 75, 52, 29, 2], name="FIX_Y_0") # Boundary condition on plane
    gmsh.model.addPhysicalGroup(2, [74, 28, 51, 1], name="FIX_X_0") # Boundary condition on plane
    gmsh.model.addPhysicalGroup(2, [77], name="FIX_Z_0") # Boundary condition on plane
    gmsh.model.addPhysicalGroup(2, [93], name="FIX_X_1") # Prescribed displacement on plane
    
    
    # Generate a 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Save as a .med file
    gmsh.write(params.med_filepath.as_posix())
    
    # Close gmsh
    gmsh.finalize()
    
    return geometry_tag_manager

@ut.track_time("(MANUAL) ADDING PHYSICAL GROUPS TO MESH")
def generate_physical_groups_manual(params, geo: cm.ManualGeometryTagManager) -> List[cm.PhysicalGroup]:
    physical_groups = []
    
    for i in range(len(geo.soil_volumes)):
        physical_groups.append(cm.PhysicalGroup(
            dim=3, tags=geo.soil_volumes[i], name=f"SOIL_LAYER_{i}",
            preferred_model = params.box_manager.layers[i].preferred_model,
            group_type=cm.PhysicalGroupType.MATERIAL, props=params.box_manager.layers[i].props,
        ))
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=geo.pile_volumes, name="CYLINDER",
            preferred_model = params.pile_manager.preferred_model,
        group_type=cm.PhysicalGroupType.MATERIAL, props=params.pile_manager.props,
    ))

    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.FIX_ALL, name="FIX_ALL",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # LEFT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.FIX_X_0, name="FIX_X_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # RIGHT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.FIX_Z_0, name="FIX_Z_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # BOTTOM FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.FIX_Y_0, name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # BACK AND FRONT FACE OF SOIL
    
    if getattr(params, 'prescribed_force', None):
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=geo.FIX_X_1, name="FORCE",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_force,
        ))  # TOP FACE OF PILE
    elif getattr(params, 'prescribed_disp', None):
        if getattr(params.prescribed_disp, 'disp_ux', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.FIX_X_1, name="FIX_X_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uy', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.FIX_X_1, name="FIX_Y_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uz', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.FIX_X_1, name="FIX_Z_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
    
    return physical_groups

@ut.track_time("DRAWING MESH")
def draw_mesh_auto(params) -> cm.GeometryTagManager:
    """
    Draws the mesh using Gmsh based on the provided parameters.

    Args:
        params: Parameters including mesh name and other necessary configurations.

    Returns:
        cm.GeometryTagManager: An object containing tags for soil and pile volumes and surfaces.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.min_x, params.box_manager.min_y, params.box_manager.min_z, params.box_manager.max_x, params.box_manager.max_y, params.box_manager.max_z), params.box_manager.far_field_size)
    gmsh.model.mesh.setSize(gmsh.model.getEntitiesInBoundingBox(params.box_manager.near_field_min_x, params.box_manager.near_field_min_y, params.box_manager.near_field_min_z, params.box_manager.near_field_max_x, params.box_manager.near_field_max_y, params.box_manager.near_field_max_z), params.box_manager.near_field_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 72)
    # gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
    # gmsh.option.setNumber("Mesh.Algorithm", 11)
    # gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    # gmsh.option.setNumber('Mesh.RecombineAll', 1)
    # gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)


    gmsh.model.add(f"{params.case_name}")

    soil_layer_tags = []
    # Add boxes for the layers
    for layer in params.box_manager.layers:
        new_tag = params.box_manager.add_layer(layer)
        soil_layer_tags.append(new_tag)
        
    if params.interface:
        interface_tag, outer_cylinder_tag, inner_cylinder_tag = params.pile_manager.addPile()
      
        # Cut the soil blocks with the pile
        cut_soil_tags, _ = gmsh.model.occ.cut(
            [[3, tag] for tag in soil_layer_tags],
            [[3, interface_tag]],
            -1,
            removeObject=True,
            removeTool=False,
        )
        cut_soil_boxes = [tag for _, tag in cut_soil_tags]

        symmetry_cutter_tag = gmsh.model.occ.addBox(params.box_manager.min_x-200, 0, params.box_manager.min_z-200, params.box_manager.max_x+400, params.box_manager.max_y+400, params.box_manager.max_z+400)

        interface_tags, _ = gmsh.model.occ.cut(
            [[3, interface_tag]],
            [[3, outer_cylinder_tag]],
            -1,
            removeObject=True,
            removeTool=False,
        )
        interface_vols = [tag for dim, tag in interface_tags]
        
        interface_tags, _ = gmsh.model.occ.cut(
            [[3, tag] for tag in interface_vols],
            [[3, symmetry_cutter_tag]],
            -1,
            removeObject=True,
            removeTool=False,
        )
        
        interface_vols = [tag for dim, tag in interface_tags]
    else:
        outer_cylinder_tag, inner_cylinder_tag = params.pile_manager.addPile()
        # Cut the soil blocks with the pile
        cut_soil_tags, _ = gmsh.model.occ.cut(
            [[3, tag] for tag in soil_layer_tags],
            [[3, outer_cylinder_tag]],
            -1,
            removeObject=True,
            removeTool=False,
        )
        cut_soil_boxes = [tag for _, tag in cut_soil_tags]
        
        symmetry_cutter_tag = gmsh.model.occ.addBox(params.box_manager.min_x-200, 0, params.box_manager.min_z-200, params.box_manager.max_x+400, params.box_manager.max_y+400, params.box_manager.max_z+400)
        
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
        surface_data = mshcrte_common.get_surface_extremes(soil_box)
        soil_surface_tags = mshcrte_common.update_surface_tags(soil_surface_tags, surface_data)

    pile_surface_tags = cm.SurfaceTags()
    pile_surface_data = mshcrte_common.get_surface_extremes(pile_vols[0])
    pile_surface_tags = mshcrte_common.update_surface_tags(pile_surface_tags, pile_surface_data)
    
    interface_surface_tags = cm.SurfaceTags()
    if params.interface:
        interface_surface_data = mshcrte_common.get_surface_extremes(interface_vols[0])
        interface_surface_tags = mshcrte_common.update_surface_tags(interface_surface_tags, interface_surface_data)
    
    geometry_tag_manager = cm.GeometryTagManager(
        soil_volumes=cut_soil_boxes,
        pile_volumes=pile_vols,
        interface_volumes = interface_vols if params.interface else [],
        soil_surfaces=soil_surface_tags,
        pile_surfaces=pile_surface_tags,
        interface_surfaces=interface_surface_tags,
    )
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    
    return geometry_tag_manager

@ut.track_time("CREATING PHYSICAL GROUPS")
def generate_physical_groups_auto(params, geo: cm.GeometryTagManager) -> List[cm.PhysicalGroup]:
    physical_groups = []
    
    for i in range(len(geo.soil_volumes)):
        physical_groups.append(cm.PhysicalGroup(
            dim=3, tags=[geo.soil_volumes[i]], name=f"SOIL_LAYER_{i}",
            preferred_model = params.box_manager.layers[i].preferred_model,
            group_type=cm.PhysicalGroupType.MATERIAL, props=params.box_manager.layers[i].props,
        ))
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=geo.pile_volumes, name="CYLINDER",
            preferred_model = params.pile_manager.preferred_model,
        group_type=cm.PhysicalGroupType.MATERIAL, props=params.pile_manager.props,
    ))
    if params.interface:
        physical_groups.append(cm.PhysicalGroup(
            dim=3, tags=geo.interface_volumes, name="INTERFACE",
                preferred_model = params.interface_manager.preferred_model,
            group_type=cm.PhysicalGroupType.MATERIAL, props=params.interface_manager.props,
        ))

    # Adding boundary condition physical groups
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.max_x_surfaces, name="FIX_ALL",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # LEFT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_x_surfaces, name="FIX_X_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # RIGHT FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.soil_surfaces.min_z_surfaces, name="FIX_Z_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # BOTTOM FACE OF SOIL
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=[*geo.soil_surfaces.min_y_surfaces, *geo.soil_surfaces.max_y_surfaces, 
                     *geo.pile_surfaces.max_y_surfaces, *geo.interface_surfaces.max_y_surfaces,
                     ], name="FIX_Y_0",
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0)
    ))  # BACK AND FRONT FACE OF SOIL
    
    if getattr(params, 'prescribed_force', None):
        physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FORCE",
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_force,
        ))  # TOP FACE OF PILE
    elif getattr(params, 'prescribed_disp', None):
        if getattr(params.prescribed_disp, 'disp_ux', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FIX_X_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uy', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FIX_Y_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
        elif getattr(params.prescribed_disp, 'disp_uz', None):
            physical_groups.append(cm.PhysicalGroup(
                dim=2, tags=geo.pile_surfaces.max_z_surfaces, name="FIX_Z_1",
                group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=params.prescribed_disp,
            ))  # TOP FACE OF PILE
    return physical_groups
        
        

        
        
@ut.track_time("DRAWING MESH")
def draw_mesh_cylinder(params) -> cm.GeometryTagManager:
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 50)

    gmsh.model.add(f"{params.case_name}")

    init_soil_layer_tags = []
    # Add boxes for the layers
    for i in range(len(params.cylinder_manager.layers)):
        new_tags = params.cylinder_manager.add_layer(params.cylinder_manager.layers[i], params.pile_manager.embedded_depth)
        init_soil_layer_tags.append(new_tags)
    outer_cylinder_tag, inner_cylinder_tag = params.pile_manager.addPile()
    
    # Cut the outer cylinder with the inner cylinder to form the pile
    pile_tags, _ = gmsh.model.occ.cut(
        objectDimTags = [[3, outer_cylinder_tag]],
        toolDimTags = [[3, inner_cylinder_tag]],
        removeObject=True,
        removeTool=True,
    )
  
    sliced_soil_tags = []
    # Cut the soil blocks with the pile
    for i in range(len(init_soil_layer_tags)):
        sliced_soil_layer_tags, _ = gmsh.model.occ.cut(
            objectDimTags = init_soil_layer_tags[i],
            toolDimTags = pile_tags,
            removeObject=True,
            removeTool=False,
        )
        sliced_soil_tags.append(sliced_soil_layer_tags)
        
    list_of_flat_surface_z = [params.cylinder_manager.z]
    # Calculate cumulative depths of each layer to determine flat surface z-values
    cumulative_depth = params.cylinder_manager.z
    for layer in params.cylinder_manager.layers:
        cumulative_depth += layer.depth
        list_of_flat_surface_z.append(cumulative_depth)
        
    disp_node = gmsh.model.occ.addPoint(-1,0,10.5)
    origin_node = gmsh.model.occ.addPoint(0,0,0)
    bottom_origin_node = gmsh.model.occ.addPoint(0,0,cumulative_depth)
    centre_line = gmsh.model.occ.addLine(origin_node, bottom_origin_node)
    
    gmsh.model.occ.synchronize()
    soil_vols = []
    all_vols = []
    for i in sliced_soil_tags:
        for j in i:
            soil_vols.append(j)
    all_vols = [*soil_vols, *pile_tags]
    
    # def get_interior_soil_symmetrical_surfaces():
    #     #can either query just soil_vols or globally, here choose to check globally because it is easier to implement
    #     surfaceData = gmsh.model.occ.getEntities(2)
    #     # Define a small tolerance for floating-point comparison
    #     tolerance = 1e-5
    #     query = []
        
    #     for surface in surfaceData:
    #         surface = surface[1]
    #         x, y, z = gmsh.model.occ.getCenterOfMass(2, surface)
    #         print(f"{x} {y} {z}")
            
    #         if math.isclose(x, 0, abs_tol=tolerance) and math.isclose(y, 0, abs_tol=tolerance):
    #             query.append(surface)
    #     return query
    
    # # for i in pile_tags:
    #     # all_vols.append(j)
    # print(get_interior_soil_symmetrical_surfaces())
    
    def get_curved_bounding_y(dimTags, params) -> cm.SurfaceTags:
        # Define a small tolerance for floating-point comparison
        tolerance = 1e-5
        
        # Desired norm (radius) for the side surfaces
        target_radius = 0.5 * params.cylinder_manager.r
        curved_side_surfaces = []
        for dim, surface in dimTags:
            # Get the center of mass for each surface
            x, y, z = gmsh.model.occ.getCenterOfMass(2, surface)
            
            xy_norm = math.sqrt(x**2 + y**2)

            # Only process surfaces with center of mass in the specified z range and y not close to zero
            if (
                not any(math.isclose(z, flat_z, abs_tol=tolerance) for flat_z in list_of_flat_surface_z) and 
                (abs(y) > tolerance) and 
                (xy_norm > target_radius)
            ):
                curved_side_surfaces.append(surface)

        return curved_side_surfaces

    all_curved_side_surfaces = get_curved_bounding_y(gmsh.model.getBoundary(all_vols, combined = True, oriented=False), params)
    pile_side_surfaces = get_curved_bounding_y(gmsh.model.getBoundary(pile_tags, combined = True, oriented=False), params)
    
    soil_curved_side_surfaces = list(set(all_curved_side_surfaces) - set(pile_side_surfaces))
    
    global_surface_tags = mshcrte_common.get_global_surface_extremes_2D()
    
    geometry_tag_manager = cm.GeometryTagManager(
        soil_volumes=sliced_soil_tags,
        pile_volumes=pile_tags,
        global_surfaces=global_surface_tags,
        curved_surfaces = soil_curved_side_surfaces,
        disp_node = disp_node,
        origin_node = origin_node,
        # interface_volumes = interface_vols if params.interface else [],
        # soil_surfaces=soil_surface_tags,
        # pile_surfaces=pile_surface_tags,
        # interface_surfaces=interface_surface_tags,
    )
    # gmsh.model.occ.synchronize()
    
    return geometry_tag_manager

@ut.track_time("CREATING PHYSICAL GROUPS")
def generate_physical_groups_cylinder(params, geo: cm.GeometryTagManager) -> List[cm.PhysicalGroup]:
    physical_groups: List[cm.PhysicalGroup] = []
    
    for i in range(len(geo.soil_volumes)):
        physical_groups.append(cm.PhysicalGroup(
            dim=3, tags=list(k[1] for k in geo.soil_volumes[i]), name=f"SOIL_LAYER_{i}",
            preferred_model = params.cylinder_manager.layers[i].preferred_model,
            group_type=cm.PhysicalGroupType.MATERIAL, props=params.cylinder_manager.layers[i].props,
        ))
    physical_groups.append(cm.PhysicalGroup(
        dim=3, tags=list(k[1] for k in geo.pile_volumes), name="CYLINDER",
            preferred_model = params.pile_manager.preferred_model,
        group_type=cm.PhysicalGroupType.MATERIAL, props=params.pile_manager.props,
    ))
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.global_surfaces.min_z_surfaces, name="FIX_ALL_0",
            preferred_model = None,
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=0,disp_uy=0,disp_uz=0),
    ))
    physical_groups.append(cm.PhysicalGroup(
        dim=2, tags=geo.global_surfaces.min_y_surfaces, name="FIX_Y_0",
            preferred_model = None,
        group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=None,disp_uy=0,disp_uz=None),
    ))
    physical_groups.append(cm.PhysicalGroup(
            dim=0, tags=[geo.disp_node], name="FIX_X_1",
                preferred_model = None,
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=-1,disp_uy=None,disp_uz=None),
        ))
    physical_groups.append(cm.PhysicalGroup(
            dim=2, tags=[*geo.curved_surfaces], name="FIX_ALL_1",
                preferred_model = None,
            group_type=cm.PhysicalGroupType.BOUNDARY_CONDITION, bc=cm.SurfaceBoundaryCondition(disp_ux=None,disp_uy=None,disp_uz=None),
        ))
    
    return physical_groups

@ut.track_time("ADDING PHYSICAL GROUPS TO MESH")
def add_physical_groups(physical_groups: List[cm.PhysicalGroup]) -> None: 
    # Adding physical groups to Gmsh model
    physical_group_dimtag = {}
    for group in physical_groups:
        physical_group_dimtag[group.name] = (group.dim, gmsh.model.addPhysicalGroup(
            dim=group.dim,
            tags=group.tags,
            name=group.name,
        ))
    return physical_group_dimtag

@ut.track_time("GENERATING MESH")
def finalize_mesh(params, geo, physical_groups, physical_groups_dimTags):
    # Setting Gmsh options and generating mesh
    # radial_divisions = 10
    radial_divisions = params.mesh_radial_divisions
    try:
        tolerance = 1e-5
        list_of_flat_surface_z = [params.cylinder_manager.z]
        # Calculate cumulative depths of each layer to determine flat surface z-values
        cumulative_depth = params.cylinder_manager.z
        for layer in params.cylinder_manager.layers:
            cumulative_depth += layer.depth
            list_of_flat_surface_z.append(cumulative_depth)
            
        
        # query = []
        for _, volTag in gmsh.model.getEntitiesForPhysicalName("CYLINDER"):
            _, surfaceTags = gmsh.model.occ.getSurfaceLoops(volTag)
            for surfaceTag in surfaceTags[0]:
                _, curveTags = gmsh.model.occ.getCurveLoops(surfaceTag)
                for curveTag in curveTags[0]:
                    length = gmsh.model.occ.getMass(1, curveTag)
                    if length < 1:
                        gmsh.model.mesh.setTransfiniteCurve(curveTag, 1)
                    else:
                        # the number of nodes has to be set as the same as the nodes in the soil domain final else 
                        gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions)
        # print(4 * params.pile_manager.r/ (3 * math.pi))
        for i in range(len(geo.soil_volumes)):
            volTags = gmsh.model.getEntitiesForPhysicalName(f"SOIL_LAYER_{i}")
            for _, volTag in volTags:
                _, surfaceTags = gmsh.model.occ.getSurfaceLoops(volTag)
                for surfaceTag in surfaceTags[0]:
                    _, curveTags = gmsh.model.occ.getCurveLoops(surfaceTag)
                    for curveTag in curveTags[0]:
                        length = gmsh.model.occ.getMass(1, curveTag)
                        x,y,z = gmsh.model.occ.getCenterOfMass(1, curveTag)
                        
                        # for the short/symmetrical edge of the "virtual" pile underneath the pile
                        if math.isclose(x, params.pile_manager.r + (params.pile_manager.R-params.pile_manager.r)/2, abs_tol=tolerance) and math.isclose(y, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, 1)
                        # for the short edge of the "virtual" pile underneath the pile
                        elif math.isclose(x, -params.pile_manager.r - (params.pile_manager.R-params.pile_manager.r)/2, abs_tol=tolerance) and math.isclose(y, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, 1)
                        
                        # this number is controlled by the final else
                        # for the long/curved edge of the "virtual" pile underneath the pile to be consistent as the one with the pile
                        elif math.isclose(y, params.pile_manager.r*(2/math.pi), abs_tol=tolerance) and math.isclose(x, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions)
                        # for the long/curved edge of the "virtual" pile underneath the pile to be consistent as the one with the pile
                        elif math.isclose(y, params.pile_manager.R*(2/math.pi), abs_tol=tolerance) and math.isclose(x, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions)
                        
                        # this number is controlled by the final else
                        # for symmetrical edge of the the inner soil domain
                        elif math.isclose(x, -params.pile_manager.r/2, abs_tol=tolerance) and math.isclose(y, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions)
                        elif math.isclose(x, params.pile_manager.r/2, abs_tol=tolerance) and math.isclose(y, 0, abs_tol=tolerance):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions)    
                        
                        
                        #radial
                        elif math.isclose(y,0,abs_tol=tolerance) and any(math.isclose(z, flat_z, abs_tol=tolerance) for flat_z in list_of_flat_surface_z):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, 15, 'Progression', params.cylinder_manager.radial_progression)
                        #along the vertical edges
                        elif length > 10 and not(any(math.isclose(z, flat_z, abs_tol=tolerance) for flat_z in list_of_flat_surface_z)):
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, 10, 'Progression', 1.2)
                        #this has to be set equal to the radial number of nodes, but don't understand why?
                        #potentially just add more if else cases to fine tune
                        else:
                            gmsh.model.mesh.setTransfiniteCurve(curveTag, radial_divisions, 'Progression')
        # sys.exit()
        
        surfaces = gmsh.model.getEntities(2)
        # def get_interior_soil_horizontal_surfaces():
        #     #can either query just soil_vols or globally, here choose to check globally because it is easier to implement
        #     surfaceData = gmsh.model.occ.getEntities(2)
        #     # Define a small tolerance for floating-point comparison
        #     tolerance = 1e-5
        #     query = []
        #     for surface in surfaceData:
        #         surface = surface[1]
        #         x, y, z = gmsh.model.occ.getCenterOfMass(2, surface)
        #         if math.isclose(x, 0, abs_tol=tolerance) and math.isclose(y, 4 * params.pile_manager.r/ (3 * math.pi), abs_tol=tolerance):
        #             query.append(surface)
        #     return query
        # interior_soil_horizontal_surfaces = get_interior_soil_horizontal_surfaces()
        
        for _, tag in surfaces:
            # if tag in interior_soil_horizontal_surfaces:
            #     print(tag)
            #     # or figure out how to set as radial triangles
            #     continue
            gmsh.model.mesh.setTransfiniteSurface(tag)
            
        # sys.exit()
        volumes = gmsh.model.getEntities(3)
        for _, tag in volumes:
            _, surfaceTags = gmsh.model.occ.getSurfaceLoops(tag)
            # if any(surface in surfaceTags[0] for surface in interior_soil_horizontal_surfaces):
            #     continue
            gmsh.model.mesh.setTransfiniteVolume(tag)
        gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.option.setNumber('Mesh.Recombine3DAll', 1)
        gmsh.option.setNumber('Mesh.Recombine3DLevel', 1)
        gmsh.model.mesh.recombine()
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(params.med_filepath.as_posix())
        return physical_groups
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during mesh generation: {e}")
        gmsh.write(params.med_filepath.as_posix())
        
    finally:
        gmsh.finalize()