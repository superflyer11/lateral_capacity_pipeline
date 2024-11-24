import sys
import math
import os

original_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = ""
    
from pydantic import BaseModel
# add Coreform Cubit libraries to your path
sys.path.append(r'/mofem_install/jupyter/thomas/Coreform-Cubit-2024.8/bin')
 
import cubit
 
# start cubit - this step is key. Note that:
# cubit.init does not require any arguments.
# If you do want to provide arguments, you must
# provide 2 or more, where the first must
# be "cubit", and user args start 
# as the 2nd argument.
# If only one argument is used, it will be ignored.

cubit.init(['cubit','-nojournal'])
 
height = 1.2
blockHexRadius = 0.1732628
tolerance = 1e-6
stickup_height = 10
embedded_depth = -10.5
depths = [0, -2, -3.4, -10.5, -40]
centres_of_depths = [(depths[i-1]+depths[i])/2 for i in range(1, len(depths))]
FIX_Y_0_depths = centres_of_depths + [(stickup_height+embedded_depth)/2]
total_soil_depth = sum(depths)
body_force = True


#!python
cubit.cmd(f"create Cylinder height {abs(depths[-1])+stickup_height} radius 1 ")
outer_pile = cubit.get_last_id('volume')
cubit.cmd(f"create Cylinder height {abs(depths[-1])+stickup_height} radius 0.975 ")
inner_pile = cubit.get_last_id('volume')
cubit.cmd(f"move Volume {outer_pile} {inner_pile} x 0 y 0 z {(abs(depths[-1])+stickup_height)/2+depths[-1]} include_merged ")
cubit.cmd(f"create Cylinder height {abs(depths[-1])} radius 80 ")
cubit.cmd(f"move Volume {cubit.get_last_id('volume')} x 0 y 0 z {depths[-1]/2} include_merged")
cubit.cmd(f"subtract volume {inner_pile} from volume {outer_pile}")
cubit.cmd(f"webcut volume all with plane yplane offset 0")
entity_id_list = cubit.get_entities("volume")
remaining_entity_id_list = []
for entity_id in entity_id_list:
    center_point = cubit.get_center_point("volume", entity_id)
    if center_point[1] > 0:
        cubit.cmd(f"delete volume {entity_id}")
    else:
        remaining_entity_id_list.append(entity_id)
for entity_id in remaining_entity_id_list:
    center_point = cubit.get_center_point("volume", entity_id)
    if math.isclose(center_point[1], -40, abs_tol=tolerance):
        cubit.cmd(f"vol {entity_id} id 2")
    elif math.isclose(center_point[1], -0.5, abs_tol=tolerance):
        cubit.cmd(f"vol {entity_id} id 1")
        pile = 1
    else:
        raise NotImplementedError(f"Unexpected center point: {center_point}")
entity_id_list = cubit.get_entities("volume")

uncut_soil_layers = [2]
volume = 2
for depth in depths:
    if depth == 0:
        continue
    cubit.cmd(f"webcut volume {volume} with plane zplane offset {depth}")
    volume = cubit.get_last_id('volume')
    uncut_soil_layers.append(volume)
entity_id_list = cubit.get_entities("volume")
uncut_soil_layers_str = " ".join(map(str, uncut_soil_layers))
cubit.cmd(f"subtract volume {pile} from volume {uncut_soil_layers_str} keep_tool")
cut_soil_layers = {}
cut_soil_layers_vertical_curves = {}
for i,depth in enumerate(depths):
    if i == 0:
        continue
    cut_soil_layers[f"{i}"] = []
    cut_soil_layers_vertical_curves[f"{i}"] = []
    
entity_id_list = cubit.get_entities("volume")

for entity_id in entity_id_list:
    center_point = cubit.get_center_point("volume", entity_id)
    for index, depth in enumerate(centres_of_depths):
        if math.isclose(center_point[2], depth, rel_tol=1e-6):  # Using math.isclose with a relative tolerance
            cut_soil_layers[f"{index + 1}"].append(entity_id)
            

cubit.cmd(f"webcut volume {pile} with plane zplane offset {embedded_depth}")

pile_under = cubit.get_last_id('volume')
pile_unders = [pile_under]

volume = pile_under
for depth in depths:
    if depth == 0:
        continue
    cubit.cmd(f"webcut volume {volume} with plane zplane offset {depth}")
    volume = cubit.get_last_id('volume')
    if volume != pile_under:
        pile_unders.append(volume)

for entity_id in pile_unders:
    center_point = cubit.get_center_point("volume", entity_id)
    for index, depth in enumerate(centres_of_depths):
        if math.isclose(center_point[2], depth, rel_tol=1e-6):  # Using math.isclose with a relative tolerance
            cut_soil_layers[f"{index + 1}"].append(entity_id)    

flattened_cut_soil_layers = []
for layer_no, cut_soil_layer in cut_soil_layers.items():
    flattened_cut_soil_layers.extend(cut_soil_layer)
    soil_layers_str = " ".join(map(str, cut_soil_layer))
    new_block = cubit.get_next_block_id()
    cubit.cmd(f"block {new_block} add volume {soil_layers_str}")
    cubit.cmd(f"block {new_block} name 'MFRONT_MAT_{layer_no}'")

new_block = cubit.get_next_block_id()
cubit.cmd(f"block {new_block} add volume {pile}")
cubit.cmd(f"block {new_block} name 'MFRONT_MAT_{new_block}'")

    
cubit.cmd(f"imprint volume all")
cubit.cmd(f"merge volume all")


class BC(BaseModel):
    name: str
    surfaces: list[int]
    attribue_count: int
    attribute_values: list[int]


BC_blocks = {
    "FIX_Y_0": BC(name="FIX_Y_0", surfaces=[], attribue_count=1, attribute_values=[0]),
    "FIX_ALL_0": BC(name="FIX_ALL_0", surfaces=[], attribue_count=3, attribute_values=[0,0,0]),
    "FIX_ALL_1": BC(name="FIX_ALL_1", surfaces=[], attribue_count=3, attribute_values=[0,0,0]),
    "FIX_X_1": BC(name="FIX_X_1", surfaces=[], attribue_count=1, attribute_values=[1]),
}

surfaces = cubit.get_entities("surface")
for surface in surfaces:
    center_point = cubit.get_center_point("surface", surface)
    if math.isclose(center_point[1], 0, rel_tol=1e-6) and any(math.isclose(center_point[2], depth, abs_tol=tolerance) for depth in FIX_Y_0_depths):
        BC_blocks["FIX_Y_0"].surfaces.append(surface)
    elif math.isclose(center_point[2], depths[-1], rel_tol=1e-6):
        BC_blocks["FIX_ALL_0"].surfaces.append(surface)
    elif math.isclose(center_point[1], -80, rel_tol=1e-6):
        BC_blocks["FIX_ALL_1"].surfaces.append(surface)
    elif math.isclose(center_point[2], 10, rel_tol=1e-6):
        BC_blocks["FIX_X_1"].surfaces.append(surface)

for _, bc in BC_blocks.items():
    surface_list_str = " ".join(map(str, bc.surfaces))
    new_block = cubit.get_next_block_id()
    cubit.cmd(f"block {new_block} add surface {surface_list_str}")
    cubit.cmd(f"block {new_block} name '{bc.name}'")
    cubit.cmd(f"block {new_block} attribute count {bc.attribue_count}")
    for i, attribute_value in enumerate(bc.attribute_values):
        cubit.cmd(f"block {new_block} attribute index {i+1} {attribute_value} name 'Attribute {i+1}'")

if body_force:
    cubit.cmd(f"set duplicate block elements on")
    density = 2.16
    acceleration = 9.81
    traction = density * acceleration * 1e-3
    flattened_cut_soil_layers_str = " ".join(map(str, flattened_cut_soil_layers))
    new_block = cubit.get_next_block_id()
    cubit.cmd(f"block {new_block} add volume {flattened_cut_soil_layers_str}")
    cubit.cmd(f"block {new_block} name 'BODY_FORCE_1'")
    cubit.cmd(f"block {new_block} attribute count 3")
    cubit.cmd(f"block {new_block} attribute index 1 0 name 'Attribute 1'")
    cubit.cmd(f"block {new_block} attribute index 2 0 name 'Attribute 2'")
    cubit.cmd(f"block {new_block} attribute index 3 {traction} name 'Attribute 3'")


horizontal_symmetrical_surface_outer_positive_curves = []
horizontal_symmetrical_surface_outer_negative_curves = []
horizontal_symmetrical_surface_inner_curves = []
horizontal_semi_circle_curves = []


curves = cubit.get_entities("curve")
for curve in curves:
    center_point = cubit.get_center_point("curve", curve)
    if math.isclose(center_point[1], 0, rel_tol=1e-6) and any(math.isclose(center_point[2], depth, abs_tol=tolerance) for depth in depths):
        if center_point[0] > 1:
            horizontal_symmetrical_surface_outer_positive_curves.append(curve)
        elif center_point[0] < -1:
            horizontal_symmetrical_surface_outer_negative_curves.append(curve)
        else:
            horizontal_symmetrical_surface_inner_curves.append(curve)
    elif center_point[1] > 0 and any(math.isclose(center_point[2], depth, abs_tol=tolerance) for depth in depths):
        horizontal_semi_circle_curves.append(curve)

horizontal_symmetrical_surface_outer_positive_curves_str = " ".join(map(str,horizontal_symmetrical_surface_outer_positive_curves))
horizontal_symmetrical_surface_outer_negative_curves_str = " ".join(map(str,horizontal_symmetrical_surface_outer_negative_curves))
horizontal_symmetrical_surface_inner_curves_str = " ".join(map(str,horizontal_symmetrical_surface_inner_curves))
horizontal_semi_circle_curves_str = " ".join(map(str, horizontal_semi_circle_curves))

for layer_no, cut_soil_layer in cut_soil_layers.items():
    cut_soil_layer_str = " ".join(map(str, cut_soil_layer))
    curves = cubit.parse_cubit_list('curve', f'in volume {cut_soil_layer_str}')
    for curve in curves:
        center_point = cubit.get_center_point("curve", curve)
        if any(math.isclose(center_point[2], depth, abs_tol=tolerance) for depth in centres_of_depths):
            cut_soil_layers_vertical_curves[f"{layer_no}"].append(curve)

factors = list(range(1, 11))
for factor in factors:
    cubit.cmd(f"delete mesh volume all propagate")
    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_positive_curves_str} orient sense direction x")
    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_positive_curves_str} interval {int(15 - factor)+1}")
    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_positive_curves_str} scheme bias factor 1.5")

    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_negative_curves_str} orient sense direction nx")    
    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_negative_curves_str} interval {int(15 - factor)+1}")
    cubit.cmd(f"curve {horizontal_symmetrical_surface_outer_negative_curves_str} scheme bias factor 1.5")
    cubit.cmd(f"curve {horizontal_semi_circle_curves_str} interval {int(11 - factor)+2}")

    

    for layer_no, curves in cut_soil_layers_vertical_curves.items():
        curves_str = " ".join(map(str, curves))
        cubit.cmd(f"curve {curves_str} orient sense direction nz")
        # layer_depth = cubit.get_arc_length(curves[0]) 
        if layer_no == "1":
            # cubit.cmd(f"curve {curves_str} size {(factor**2.5) *0.1}")
            cubit.cmd(f"curve {curves_str} interval {int(15 - factor)+1}")
            cubit.cmd(f"curve {curves_str} scheme bias factor 1.2")
        elif layer_no == "2":
            cubit.cmd(f"curve {curves_str} size {math.sqrt(factor) * 0.3}")
            cubit.cmd(f"curve {curves_str} scheme equal")
        elif layer_no == "3":
            cubit.cmd(f"curve {curves_str} size {math.sqrt(factor) * 0.5}")
            cubit.cmd(f"curve {curves_str} scheme equal")
        else:
            cubit.cmd(f"curve {curves_str} size {math.sqrt(factor) * 3}")
            cubit.cmd(f"curve {curves_str} scheme equal")

    # cubit.cmd(f"imprint surface all")
    # cubit.cmd(f"merge surface all")

    cubit.cmd(f"volume all size auto factor {factor}")
    cubit.cmd(f"mesh volume all")
    if body_force:
        cubit.cmd(f"save as '/mofem_install/jupyter/thomas/mfront_example_test/user_data/pile_body_force_{factor}.cub' overwrite")
    else:
        cubit.cmd(f"save as '/mofem_install/jupyter/thomas/mfront_example_test/user_data/pile_{factor}.cub' overwrite")
    
os.environ["PYTHONPATH"] = original_pythonpath
