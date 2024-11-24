# trace generated using paraview version 5.12.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12
import sys
import subprocess

#### import the simple module from the paraview
from paraview.simple import *

class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value

def contour(params):
    paraview.simple._DisableFirstRenderCameraReset()
    renderView1 = GetActiveViewOrCreate('RenderView')
    vtk_files = subprocess.run(f"ls -c1 {params.vtk_dir}/*.vtk  | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        out_mi_0vtk = LegacyVTKReader(registrationName="final", FileNames=files)
        if out_mi_0vtk:
            print("Read Success")
        else:
            print("Read Failed")

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # show data in view
    out_mi_0vtkDisplay = Show(out_mi_0vtk, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    out_mi_0vtkDisplay.Representation = 'Surface'

    # show color bar/color legend
    out_mi_0vtkDisplay.SetScalarBarVisibility(renderView1, True)


    # update the view to ensure updated data information
    renderView1.Update()

    # get color transfer function/color map for 'GLOBAL_ID'
    gLOBAL_IDLUT = GetColorTransferFunction('GLOBAL_ID')

    # get opacity transfer function/opacity map for 'GLOBAL_ID'
    gLOBAL_IDPWF = GetOpacityTransferFunction('GLOBAL_ID')

    # get 2D transfer function for 'GLOBAL_ID'
    gLOBAL_IDTF2D = GetTransferFunction2D('GLOBAL_ID')

    # # hide data in view
    # Hide(programmableFilter1, renderView1)

    # create a new 'Programmable Filter'
    programmableFilter2 = ProgrammableFilter(registrationName='ProgrammableFilter2', Input=out_mi_0vtk)

    

    # Properties modified on programmableFilter2
    programmableFilter2.Script = """import numpy as np

def process_unstructured_dataset(input0):
    # Pick up input arrays
    e = input0.PointData["STRAIN"]  # Shape: (153819, 3, 3) or similar
    print(f"Shape of strain data (e): {np.array(e).shape}")

    # Validate the shape of `e`
    if len(e) == 0 or len(e[0]) != 3 or len(e[0][0]) != 3:
        raise ValueError("Unexpected shape for strain tensor array. Expected (n_points, 3, 3).")

    # Convert `e` to a NumPy array if needed
    strain = np.array(e)  # Shape: (n_points, 3, 3)

    # Validate the dimensions
    if strain.shape[1:] != (3, 3):
        raise ValueError("Strain tensor should be 3x3 for each point.")

    # Calculate principal strains
    # The `eigvalsh` function computes the eigenvalues of a symmetric matrix
    principal_strain = np.linalg.eigvalsh(strain)  # Shape: (n_points, 3)

    mean_strain = np.trace(strain, axis1=1, axis2=2) / 3
    identity_matrix = np.eye(3)
    deviatoric_strain = strain - mean_strain[:, None, None] * identity_matrix
        
    # Compute J2 and J3 (deviatoric strain invariants)
    J2 = 0.5 * np.sum(deviatoric_strain**2, axis=(1, 2))  # Second invariant of deviatoric strain
    J3 = np.linalg.det(deviatoric_strain)  # Third invariant (determinant of deviatoric strain tensor)

        
    deviatoric_strain_magnitude_percentage = np.sqrt(np.sum(deviatoric_strain**2, axis=(1, 2)) / 2) * 100

    # Calculate the Lode angle (in radians)
    with np.errstate(divide=\'ignore\', invalid=\'ignore\'):  # Handle numerical issues
        lode_angle = np.degrees((1 / 3) * np.arcsin((3 * np.sqrt(3) * J3) / (2 * J2**(3 / 2))))



    # Assign principal strains to the output dataset
    output.PointData.append(principal_strain[:, 0], "e_1")  # Minimum principal strain
    output.PointData.append(principal_strain[:, 1], "e_2")  # Middle principal strain
    output.PointData.append(principal_strain[:, 2], "e_3")  # Maximum principal strain
    output.PointData.append(deviatoric_strain_magnitude_percentage, "e_d_pct")
    output.PointData.append(lode_angle, "theta")  # Lode angle in radians

    print("Principal strains, deviatoric strain magnitude, and Lode angle calculated and added to the output.")

    # Input dataset
    input0 = inputs[0]

    if input0.IsA("vtkUnstructuredGrid"):
        process_unstructured_dataset(input0)
        print(\'Done processing unstructured dataset.\')
    else:
        print("Bad dataset type for this script.")"""
        
    # Properties modified on programmableFilter2
    programmableFilter2.Script = ''
    programmableFilter2.RequestInformationScript = ''
    programmableFilter2.RequestUpdateExtentScript = ''
    programmableFilter2.PythonPath = ''

    # show data in view
    programmableFilter2Display = Show(programmableFilter2, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    programmableFilter2Display.Representation = 'Surface'

    # hide data in view
    Hide(out_mi_0vtk, renderView1)



    # update the view to ensure updated data information
    renderView1.Update()

    # get layout
    layout1 = GetLayout()


    # layout/tab size in pixels
    layout1.SetSize(677, 478)

    # current camera placement for renderView1
    renderView1.CameraPosition = [-30.60866177116384, 48.97696136009138, 21.386791748873094]
    renderView1.CameraFocalPoint = [15.823970070432473, -25.32794930415114, -13.237735255931925]
    renderView1.CameraViewUp = [0.21890211818805158, -0.2964016170103922, 0.9296386093995931]
    renderView1.CameraParallelScale = 92.74563845718737

    # set active source
    SetActiveSource(programmableFilter2)
    
    # set scalar coloring
    ColorBy(programmableFilter2Display, ('POINTS', 'e_d_pct'))

    # change representation type
    programmableFilter2Display.SetRepresentationType('Surface With Edges')

    # Properties modified on programmableFilter2Display
    programmableFilter2Display.EdgeColor = [0.0, 0.0, 0.0]

    # Properties modified on programmableFilter2Display
    programmableFilter2Display.EdgeOpacity = 0.4

    # change representation type
    programmableFilter2Display.SetRepresentationType('Surface LIC')

    # change representation type
    programmableFilter2Display.SetRepresentationType('Surface With Edges')

    # rescale color and/or opacity maps used to include current data range
    programmableFilter2Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    programmableFilter2Display.SetScalarBarVisibility(renderView1, True)
    
    ImportPresets(filename=f'{params.preset_dir}/e_d_pct.json', location=16)
    ImportPresets(filename=f'{params.preset_dir}/theta.json', location=16)

    # get color transfer function/color map for 'e_d_pct'
    e_d_pctLUT = GetColorTransferFunction('e_d_pct')

    # get opacity transfer function/opacity map for 'e_d_pct'
    e_d_pctPWF = GetOpacityTransferFunction('e_d_pct')

    # get 2D transfer function for 'e_d_pct'
    e_d_pctTF2D = GetTransferFunction2D('e_d_pct')

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    e_d_pctLUT.ApplyPreset('e_d_pct', True)

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    e_d_pctPWF.ApplyPreset('e_d_pct', True)

    # get the material library
    materialLibrary1 = GetMaterialLibrary()


    # #change interaction mode for render view
    # renderView1.InteractionMode = '3D'

    

    # save screenshot
    SaveScreenshot(filename=f'{params.graph_dir}/e_d_le.png', viewOrLayout=renderView1, location=16, ImageResolution=[677, 478],
        TransparentBackground=1)

    # set scalar coloring
    ColorBy(programmableFilter2Display, ('POINTS', 'theta'))

    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(e_d_pctLUT, renderView1)

    # rescale color and/or opacity maps used to include current data range
    programmableFilter2Display.RescaleTransferFunctionToDataRange(True, False)

    # show color bar/color legend
    programmableFilter2Display.SetScalarBarVisibility(renderView1, True)

    # get color transfer function/color map for 'theta'
    thetaLUT = GetColorTransferFunction('theta')

    # get opacity transfer function/opacity map for 'theta'
    thetaPWF = GetOpacityTransferFunction('theta')

    # get 2D transfer function for 'theta'
    thetaTF2D = GetTransferFunction2D('theta')

    # get color legend/bar for thetaLUT in view renderView1
    thetaLUTColorBar = GetScalarBar(thetaLUT, renderView1)

    # Properties modified on thetaLUTColorBar
    thetaLUTColorBar.WindowLocation = 'Lower Right Corner'

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    thetaLUT.ApplyPreset('theta', True)

    # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
    thetaPWF.ApplyPreset('theta', True)

    # save screenshot
    SaveScreenshot(filename=f'{params.graph_dir}s\theta_le.png', viewOrLayout=renderView1, location=16, ImageResolution=[677, 478],
        TransparentBackground=1)

params = AttrDict()
params.vtk_dir = sys.argv[1] 
params.graph_dir = sys.argv[2]
params.preset_dir = sys.argv[3]
contour(params)