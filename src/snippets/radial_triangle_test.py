import gmsh

gmsh.initialize()
gmsh.model.add("Semi Circle Radial Triangles")

# Define the half-circle geometry
R = 1.0  # Radius of the half-circle
lc = 0.1  # Characteristic length for mesh refinement

# Define points (center and boundary)
p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)  # Center
p1 = gmsh.model.occ.addPoint(R, 0, 0, lc)  # Right edge of the half-circle
p2 = gmsh.model.occ.addPoint(0, R, 0, lc)  # Top of the half-circle
p3 = gmsh.model.occ.addPoint(-R, 0, 0, lc) # Left edge of the half-circle

# Define half-circle arc
arc1 = gmsh.model.occ.addCircleArc(p1, p0, p2)  # Right half
arc2 = gmsh.model.occ.addCircleArc(p2, p0, p3)  # Left half

# Define the straight line at the base of the half-circle
line = gmsh.model.occ.addLine(p3, p1)

# Create a closed loop for the half-circle surface
curve_loop = gmsh.model.occ.addCurveLoop([arc1, arc2, line])
surface = gmsh.model.occ.addPlaneSurface([curve_loop])

# Synchronize the CAD model with the Gmsh model
gmsh.model.occ.synchronize()

# Set transfinite meshing on the curves
# Divide each curve into a specific number of points
divisions = 20  # Number of divisions along the radius and arcs

gmsh.model.mesh.setTransfiniteCurve(arc1, divisions)
gmsh.model.mesh.setTransfiniteCurve(arc2, divisions)
gmsh.model.mesh.setTransfiniteCurve(line, divisions)

# Define the surface as transfinite with a radial layout
gmsh.model.mesh.setTransfiniteSurface(surface, "Left", [p3, p0, p2])
# gmsh.model.mesh.setRecombine(2, surface)  # Optional: to recombine into quadrangles if desired

# Generate the mesh
gmsh.model.mesh.generate(2)

# Optionally save the mesh to file
gmsh.write("semi_circle_radial_triangles.msh")

gmsh.fltk.run()

gmsh.finalize()
