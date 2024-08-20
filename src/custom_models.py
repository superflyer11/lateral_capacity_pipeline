import math
from typing import Any, Dict, List, Optional
from typing_extensions import Self
from pydantic import BaseModel, model_validator
import gmsh
from enum import Enum
    
class SurfaceTags(BaseModel):
    min_x_surfaces: list = []
    max_x_surfaces: list = []
    min_y_surfaces: list = []
    max_y_surfaces: list = []
    min_z_surfaces: list = []
    max_z_surfaces: list = []
    min_x: float = float('inf')
    max_x: float = float('-inf')
    min_y: float = float('inf')
    max_y: float = float('-inf')
    min_z: float = float('inf')
    max_z: float = float('-inf')

class GeometryTagManager(BaseModel):
    soil_volumes: list 
    pile_volumes: list 
    soil_surfaces: SurfaceTags
    pile_surfaces: SurfaceTags


class BoundaryCondition(BaseModel):
    pass

class NodeBoundaryCondition(BoundaryCondition):
    disp_ux: float = 0
    disp_uy: float = 0
    disp_flag3: float = 0

class EdgeBoundaryCondition(BoundaryCondition):
    disp_ux: float = 0
    disp_uy: float = 0
    disp_uz: float = 0

class SurfaceBoundaryCondition(BoundaryCondition):
    disp_ux: float = 0
    disp_uy: float = 0
    disp_uz: float = 0
    
class ForceBoundaryCondition(BoundaryCondition):
    fx: int = 0
    fy: int = 0
    fz: int = 0

class PropertyTypeEnum(Enum):
    elastic = 1
    drucker_prager = 2
    cam_clay = 3

class MaterialProperty(BaseModel):
    pass
    
class LinearElasticProperties(MaterialProperty):
    youngs_modulus: float
    poisson_ratio: float

class DruckerPragerProperties(MaterialProperty):
    pass

class CamClayProperties(MaterialProperty):
    hvorslev_shape_alpha: float = 0
    hvorslev_shape_n: float = 0
    hvorslev_plastic_potential_beta: float = 0
    hvorslev_plastic_potential_m: float = 0
    v0: float = 0
    la: float = 0
    ka: float = 0
    G_0: float = 0
    a: float = 0
    b: float = 0
    RG_min: float = 0
    
    

class SoilLayer(BaseModel):
    depth: float
    linear_elastic_properties: LinearElasticProperties
    mcc_properties: CamClayProperties = CamClayProperties()

class BC_CONFIG_BLOCK(BaseModel):
    block_name: str
    comment: str
    id: int
    attributes: list

    @property
    def number_of_attributes(self) -> int:
        return len(self.attributes)
    
    def formatted(self):
        attrs = ''
        for i in range(len(self.attributes)):
            attrs += f'user{i+1}={self.attributes[i]}\n'       
        block = f"""[{self.block_name}]
# {self.comment}
number_of_attributes={self.number_of_attributes}\n""" + attrs
        return block
    
class MFRONT_CONFIG_BLOCK(BaseModel):
    block_name: str
    comment: str
    id: int
    name: str
    
    def formatted(self):
        block = f"""[{self.block_name}]
# {self.comment}
id={self.id}
add=BLOCKSET
name={self.name}
"""
        return block

class BoxManager(BaseModel):
    x: float
    y: float
    z: float
    dx: float
    dy: float
    new_layer_z: float = 0
    far_field_size: float = 5
    near_field_dist: float = 40
    near_field_size: float = 1
    layers: Dict[int, SoilLayer]
    
    
    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y

    @property
    def min_z(self) -> float:
        total_depth = sum(layer.depth for layer in self.layers.values())
        return total_depth + self.max_z

    @property
    def max_x(self) -> float:
        return self.x + self.dx
    
    @property
    def max_y(self) -> float:
        return self.y + self.dy

    @property
    def max_z(self) -> float:
        return self.z
    
    @property
    def near_field_min_x(self) -> float:
        return (self.min_x + self.max_x) / 2 - self.near_field_dist
    
    @property
    def near_field_min_y(self) -> float:
        return self.max_y - self.near_field_dist

    @property
    def near_field_min_z(self) -> float:
        return self.min_z

    @property
    def near_field_max_x(self) -> float:
        return (self.min_x + self.max_x) / 2 + self.near_field_dist
    
    @property
    def near_field_max_y(self) -> float:
        return self.max_y

    @property
    def near_field_max_z(self) -> float:
        return self.max_z
    
    def add_layer(self, box: SoilLayer):
        layer_tag = gmsh.model.occ.addBox(self.x, self.y, self.new_layer_z, self.dx, self.dy, box.depth)
        self.new_layer_z += box.depth
        return layer_tag
    

class PileManager(BaseModel):
    x: float
    y: float
    z: float
    dx: float 
    dy: float 
    dz: float 
    R: float
    r: float
    linear_elastic_properties: LinearElasticProperties
    
    def addPile(self):
        outer_tag = gmsh.model.occ.addCylinder(self.x, self.y, self.z, self.dx, self.dy, self.dz, self.R, angle= 2*math.pi)
        inner_tag = gmsh.model.occ.addCylinder(self.x, self.y, self.z, self.dx, self.dy, self.dz, self.r, angle= 2*math.pi)
        return [outer_tag, inner_tag]
    
    # def create_CFGBLOCK(self) -> MFRONT_CONFIG_BLOCK:
    #     block = MFRONT_CONFIG_BLOCK(
    #         name=f'CYLINDER', 
    #         number_of_attributes=2, 
    #         attributes=[
    #         self.linear_elastic_properties.youngs_modulus,
    #         self.linear_elastic_properties.poisson_ratio,
    #     ])
    #     return block
    
class MeshsetInfo(BaseModel):
    meshset_id: int
    name: str

class PhysicalGroupType(Enum):
    MATERIAL = 1
    BOUNDARY_CONDITION = 2

class PhysicalGroup(BaseModel):
    dim: int
    tags: List[int]
    name: str
    meshnet_id: Optional[int] = None
    group_type: PhysicalGroupType
    props: Dict[PropertyTypeEnum, MaterialProperty] = {} 
    bc: Optional[BoundaryCondition] = None
    
    def verify(self):
        if meshnet_id == None:
            raise ValueError('Missing meshnet_id')  
    

if __name__ == "__main__":
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add(f"test")

    pile_manager = PileManager(x=0, y=0, z=10.5, dx=0, dy=0, dz=-20.5, R=1, r=0.975,
                              elastic_properties=LinearElasticProperties(youngs_modulus=200 * (10 ** 9), poisson_ratio=0.3)
                              )

    outer_cylinder_tag, inner_cylinder_tag = pile_manager.addPile()
    
    

    DIM=3
    cut_pile_tags, _ = gmsh.model.occ.cut(
        [[DIM, outer_cylinder_tag]],
        [[DIM, inner_cylinder_tag]],
        -1,
        removeObject=True,
        removeTool=True,
    )
    pile_boxes = [tag for dim, tag in cut_pile_tags][0]
    
    
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(dim=3, tags=[outer_cylinder_tag], name='pile')
    gmsh.model.addPhysicalGroup(dim=1, tags=[contact_arc], name='contact')

    gmsh.model.mesh.generate(3)
    tags = gmsh.model.mesh.getElementsByCoordinates(1,0,0,1,strict=False)
    print(tags)
    # contact_arc
    gmsh.write("/mofem_install/jupyter/thomas/mfront_example_test/test_data/123.med")
    gmsh.finalize()
    