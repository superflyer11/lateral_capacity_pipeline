import gmsh
from pydantic import BaseModel
import os

os.chdir("/mofem_install/jupyter/thomas/mfront_example_test")
class Box(BaseModel):
    x: float = 0
    y: float = 0
    z: float = 0
    dx: float = 80
    dy: float = 80
    dz: float = 80

    def create(self):
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 99)
        
        pt1 = gmsh.model.occ.addPoint(self.x,self.y,self.z)
        pt2 = gmsh.model.occ.addPoint(self.x + self.dx ,self.y,self.z)
        pt3 = gmsh.model.occ.addPoint(self.x + self.dx ,self.y + self.dy,self.z)
        pt4 = gmsh.model.occ.addPoint(self.x,self.y + self.dy,self.z)

        l1 = gmsh.model.occ.addLine(pt1, pt2)
        l2 = gmsh.model.occ.addLine(pt2, pt3)
        l3 = gmsh.model.occ.addLine(pt3, pt4)
        l4 = gmsh.model.occ.addLine(pt4, pt1)
        cl1 = gmsh.model.occ.addCurveLoop([l1,l2,l3,l4])
        s1 = gmsh.model.occ.addPlaneSurface([cl1])
        
        VOLUME = gmsh.model.occ.extrude([(2, s1)], 0, 0, -80, [2,2,2,4,4], [2/80,1.4/80,7.1/80,29.5/80], recombine=True)
        
        gmsh.model.occ.synchronize()
        for curve in [l1,l2,l3,l4]:
            gmsh.model.mesh.setTransfiniteCurve(curve,10)
            
        for surf in gmsh.model.occ.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(surf[1])
        
        for vol in gmsh.model.occ.getEntities(3):
            gmsh.model.mesh.setTransfiniteVolume(vol[1])
        
        gmsh.option.setNumber('Mesh.RecombineAll', 1)
        gmsh.option.setNumber('Mesh.Recombine3DAll', 1)
        gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 2)
        gmsh.option.setNumber('Mesh.Recombine3DLevel', 0)
        # gmsh.option.setNumber('Mesh.ElementOrder', 2)
        # gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        # gmsh.option.setNumber('Mesh.MedFileMinorVersion', 0)
        # gmsh.option.setNumber('Mesh.SaveAll', 0)
        gmsh.option.setNumber('Mesh.SaveGroupsOfNodes', 1)
        gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", 10)
        # gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)


        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.recombine()
        gmsh.write("testing.med")
        gmsh.finalize()
    
    
Box().create()