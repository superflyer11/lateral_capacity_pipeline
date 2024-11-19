import gmsh

gmsh.initialize()
gmsh.model.add("Semi Circle Radial Triangles")
gmsh.option.setNumber('Mesh.Algorithm', 8)
gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 3)
gmsh.option.setNumber('Mesh.RecombineAll', 1)

# Define the half-circle geometry
R = 1.0  # Radius of the half-circle
lc = 0.1  # Characteristic length for mesh refinement

# Define points (center and boundary)
p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)  # Center
p1 = gmsh.model.occ.addPoint(R, 0, 0, lc)  # Right edge of the half-circle
p3 = gmsh.model.occ.addPoint(-R, 0, 0, lc) # Left edge of the half-circle

# Define half-circle arc
arc1 = gmsh.model.occ.addCircleArc(p3, p0, p1)  # Right half

# Define the straight line at the base of the half-circle
line1 = gmsh.model.occ.addLine(p3, p0)
line2 = gmsh.model.occ.addLine(p0, p1)

# Create a closed loop for the half-circle surface
curve_loop = gmsh.model.occ.addCurveLoop([arc1, line1, line2])
surface = gmsh.model.occ.addPlaneSurface([curve_loop])

# Synchronize the CAD model with the Gmsh model
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2,[surface],name="surface")
# Set transfinite meshing on the curves
# Divide each curve into a specific number of points
divisions = 20  # Number of divisions along the radius and arcs

gmsh.model.mesh.setTransfiniteCurve(arc1, 2*divisions)
gmsh.model.mesh.setTransfiniteCurve(line1, divisions)
gmsh.model.mesh.setTransfiniteCurve(line2, divisions)

# Define the surface as transfinite with a radial layout
gmsh.model.mesh.setTransfiniteSurface(surface, "Left", [p0, p1, p3])
gmsh.model.mesh.setRecombine(2, surface)  # Optional: to recombine into quadrangles if desired
# Generate the mesh
gmsh.model.mesh.generate(3)

# Optionally save the mesh to file
gmsh.write("semi_circle_radial_triangles.med")

# gmsh.fltk.run()

gmsh.finalize()
