import math
from enum import Enum
# from enum import IntEnum, StrEnum
from typing import Any, Dict, List, Union, Optional
from typing_extensions import Self
from pydantic import BaseModel, model_validator, ConfigDict, SerializeAsAny, root_validator
import gmsh
from enum import Enum

# one giant glob of parameters
class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value

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
    
class CurveTags(BaseModel):
    min_x_curves: list = []
    max_x_curves: list = []
    min_y_curves: list = []
    max_y_curves: list = []
    min_z_curves: list = []
    max_z_curves: list = []
    min_x: float = float('inf')
    max_x: float = float('-inf')
    min_y: float = float('inf')
    max_y: float = float('-inf')
    min_z: float = float('inf')
    max_z: float = float('-inf')

class NodeTags2D(BaseModel):
    min_x_min_y_node: int = None
    min_x_max_y_node: int = None
    max_x_min_y_node: int = None
    max_x_max_y_node: int = None
    min_x: float = float('inf')
    max_x: float = float('-inf')
    min_y: float = float('inf')
    max_y: float = float('-inf')

class GeometryTagManager(BaseModel):
    soil_volumes: list 
    pile_volumes: list 
    interface_volumes: list
    soil_surfaces: SurfaceTags
    pile_surfaces: SurfaceTags
    interface_surfaces: SurfaceTags
    FIX_ALL: list | None = None
    FIX_Y_0: list | None = None
    FIX_X_0: list | None = None
    FIX_Z_0: list | None = None
    FIX_X_1: list | None = None
    
    
class ManualGeometryTagManager(BaseModel):
    soil_volumes: list 
    pile_volumes: list 
    FIX_ALL: list
    FIX_Y_0: list
    FIX_X_0: list
    FIX_Z_0: list
    FIX_X_1: list

class TestTagManager(BaseModel):
    test_volume: list
    test_surfaces: SurfaceTags
    
class TestTagManager2D(BaseModel):
    test_surface: list
    test_curves: CurveTags
    test_nodes: NodeTags2D

class TimeHistory(BaseModel):
    history: dict[float, float] | None = None

    def write(self, filepath):
        attrs = ''
        print(self.history)
        for step, value in self.history.items():
            attrs += f"{step} {value}\n"

        with open(filepath, "w") as f:
            f.writelines(attrs)

class BoundaryCondition(BaseModel):
    @model_validator(mode="after")
    def check_single_dict_field(self):
        
        # Count the fields that are dictionaries
        dict_fields = [field for field in self.model_fields if isinstance(getattr(self, field), dict)]
        if len(dict_fields) > 1:
            raise ValueError("Only one of disp_ux, disp_uy, or disp_uz can be a dictionary.")
        return self
    
    def replace_dict_with_value(self, value: float = 1.0) -> TimeHistory:
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, dict):
                print("extracted the value")  # Print message
                extracted_dict = field_value  # Store the dict in the result
                setattr(self, field_name, value)  # Replace the field with 1.0
                return TimeHistory(history=extracted_dict)
        return None
    # model_config = ConfigDict(extra='allow')

    # def model_dump(self):
    #     return self.dict()


            

class NodeBoundaryCondition(BoundaryCondition):
    disp_ux: float | dict[float, float] | None = None
    disp_uy: float | dict[float, float] | None = None
    disp_uz: float | dict[float, float] | None = None

class EdgeBoundaryCondition(BoundaryCondition):
    disp_ux: float | dict[float, float] | None = None
    disp_uy: float | dict[float, float] | None = None
    disp_uz: float | dict[float, float] | None = None
    
class SurfaceBoundaryCondition(BoundaryCondition):
    disp_ux: float | None = None
    disp_uy: float | None = None
    disp_uz: float | None = None
    
class ForceBoundaryCondition(BoundaryCondition):
    fx: int = 0
    fy: int = 0
    fz: int = 0

# value should be the behvaiour name in mfront
class PropertyTypeEnum(str, Enum):
    elastic = "LinearElasticity" #good performance
    saint_venant_kirchhoff = "SaintVenantKirchhoff" #not using this
    von_mises = "VMSimo" #tested to be have the same results as mfront gallery implementation
    drucker_prager = "DruckerPragerSimple" 
    dp_hyperbolic = "DruckerPragerHyperboloidal"
    # drucker_prager = "DruckerPragerParaboloidal" # fails mfront integration
    cam_clay = "ModCamClay_semiExpl" # none of them is working so far

class MaterialProperty(BaseModel):
    
    @property
    def mi_param_0(self) -> float: return 0
    
    @property
    def mi_param_1(self) -> float: return 0
    
    @property
    def mi_param_2(self) -> float: return 0
    
    @property
    def mi_param_3(self) -> float: return 0
    
    @property
    def mi_param_4(self) -> float: return 0
    
    @property
    def mi_param_5(self) -> float: return 0
    
    # model_config = ConfigDict(extra='allow') 
    # def model_dump(self):
    #     return self.dict()
    
class ElasticProperties(MaterialProperty):
    youngs_modulus: float
    poisson_ratio: float
    
    @property
    def mi_param_0(self) -> float:
        return self.youngs_modulus

    @property
    def mi_param_1(self) -> float:
        return self.poisson_ratio

class VonMisesProperties(MaterialProperty):
    youngs_modulus: float
    poisson_ratio: float
    HardeningSlope: float
    YieldStress: float
    
    @property
    def mi_param_0(self) -> float:
        return self.YieldStress

    @property
    def mi_param_1(self) -> float:
        return self.HardeningSlope
    
    
    @property
    def mi_param_2(self) -> float:
        return self.youngs_modulus

    @property
    def mi_param_3(self) -> float:
        return self.poisson_ratio


class DruckerPragerProperties(MaterialProperty):
    youngs_modulus: float
    poisson_ratio: float
    phi: float
    c: float
    v: float
    
    @property
    def mi_param_0(self) -> float:
        return self.phi

    @property
    def mi_param_1(self) -> float:
        return self.c
    
    @property
    def mi_param_2(self) -> float:
        return self.youngs_modulus

    @property
    def mi_param_3(self) -> float:
        return self.poisson_ratio
    
    # @property
    # def mi_param_4(self) -> float:
    #     return self.v

class DruckerPragerHyperbolicProperties(MaterialProperty):
    youngs_modulus: float
    poisson_ratio: float
    phi: float
    c: float
    v: float
    proximity: float
    
    @property
    def mi_param_0(self) -> float:
        return self.phi

    @property
    def mi_param_1(self) -> float:
        return self.c
    
    @property
    def mi_param_2(self) -> float:
        return self.proximity
    
    @property
    def mi_param_3(self) -> float:
        return self.youngs_modulus

    @property
    def mi_param_4(self) -> float:
        return self.poisson_ratio
    
    # @property
    # def mi_param_4(self) -> float:
    #     return self.v


class CamClayProperties(MaterialProperty):
    nu: float = 0.3 #PoissonRatio
    M: float = 1.2 #CriticalStateLineSlope
    la: float = 7.7e-2 #ViringConsolidationLineSlope
    ka: float = 6.6e-3 #SwellingLineSlope    
    v0: float = 1.7857 #InitialVolumeRatio
    pc0: float = 400 #CharacteristicPreconsolidationPressure
    
    @property
    def mi_param_0(self) -> float: return self.nu

    @property
    def mi_param_1(self) -> float: return self.M

    @property
    def mi_param_2(self) -> float: return self.ka

    @property
    def mi_param_3(self) -> float: return self.la

    @property
    def mi_param_4(self) -> float: return self.pc0

    @property
    def mi_param_5(self) -> float: return self.v0


    

class SoilLayer(BaseModel):
    depth: float
    preferred_model: PropertyTypeEnum
    props: dict[PropertyTypeEnum, MaterialProperty]
    # linear_elastic_properties: LinearElasticProperties
    # mcc_properties: CamClayProperties = CamClayProperties()
    
    class Config:  
        use_enum_values = True  # <--

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
    
class TestAttr(BaseModel):
    preferred_model: PropertyTypeEnum = PropertyTypeEnum.elastic
    props: dict[PropertyTypeEnum, MaterialProperty| None] = {PropertyTypeEnum.elastic: None} 

class InterfaceManager(BaseModel):
    preferred_model: PropertyTypeEnum = PropertyTypeEnum.elastic
    props: dict[PropertyTypeEnum, MaterialProperty| None] = {PropertyTypeEnum.elastic: None} 

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
    layers: list[SoilLayer]
    
    
    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y

    @property
    def min_z(self) -> float:
        total_depth = sum(layer.depth for layer in self.layers)
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
    interface: bool
    preferred_model: PropertyTypeEnum = PropertyTypeEnum.elastic
    props: dict[PropertyTypeEnum, MaterialProperty| None] = {PropertyTypeEnum.elastic: None} 
    # linear_elastic_properties: LinearElasticProperties
    
    class Config:  
        use_enum_values = True  # <--
    
    def addPile(self):
        if self.interface:
            interface_tag = gmsh.model.occ.addCylinder(self.x, self.y, 0, self.dx, self.dy, self.dz+self.z, self.R+0.01, angle= 2*math.pi)
        outer_tag = gmsh.model.occ.addCylinder(self.x, self.y, self.z, self.dx, self.dy, self.dz, self.R, angle= 2*math.pi)
        inner_tag = gmsh.model.occ.addCylinder(self.x, self.y, self.z, self.dx, self.dy, self.dz, self.r, angle= 2*math.pi)
        if self.interface:
            return [interface_tag, outer_tag, inner_tag]
        else:
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
    preferred_model: PropertyTypeEnum = PropertyTypeEnum.elastic
    props: dict[PropertyTypeEnum, MaterialProperty] = {} 
    bc: Optional[BoundaryCondition] = None
    
    def verify(self):
        if meshnet_id == None:
            raise ValueError('Missing meshnet_id')  


if __name__ == "__main__":
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add(f"test")

    pile_manager = PileManager(x=0, y=0, z=10.5, dx=0, dy=0, dz=-20.5, R=1, r=0.975,
                              elastic_properties=ElasticProperties(youngs_modulus=200 * (10 ** 9), poisson_ratio=0.3)
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

