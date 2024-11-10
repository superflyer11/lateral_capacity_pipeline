import sys
import subprocess
from paraview.simple import *
import utils as ut

class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value

@ut.track_time("EXTRACTING DATA OVER ALL TIMESTEPS AT CERTAIN POINTS FROM .vtk")
def to_time(params):
    paraview.simple._DisableFirstRenderCameraReset()
    renderView1 = GetActiveViewOrCreate('RenderView')
    vtk_files = subprocess.run(f"ls -c1 {params.data_dir}/*.vtk  | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        out_mi_0vtk = LegacyVTKReader(registrationName="final", FileNames=files)
        if out_mi_0vtk:
            print("Read Success")
        else:
            print("Read Failed")
        
        sel1 = QuerySelect(QueryString='(pointIsNear([(5, 5, 5),], 1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=out_mi_0vtk)
        plotSelectionOverTime1 = PlotSelectionOverTime(Input=out_mi_0vtk, Selection=sel1)
        SaveData(params.csv_filepath, proxy=plotSelectionOverTime1, RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
        FieldAssociation='Row Data',
        AddTimeStep=1,
        AddTime=1)

@ut.track_time("CREATING ANIMATION OVER ALL TIMESTEPS FROM .vtk")
def animation(params):    
    paraview.simple._DisableFirstRenderCameraReset()
    
    vtk_files = subprocess.run(f"ls -c1 {params.data_dir}/*.vtk | sort -V", shell=True, text=True, capture_output=True)
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

        # reset view to fit data
        renderView1.ResetCamera(False, 0.9)

        # get the material library
        materialLibrary1 = GetMaterialLibrary()

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

        # create a new 'Warp By Vector'
        warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=out_mi_0vtk)

        # show data in view
        warpByVector1Display = Show(warpByVector1, renderView1, 'UnstructuredGridRepresentation')

        # trace defaults for the display properties.
        warpByVector1Display.Representation = 'Surface'

        # hide data in view
        Hide(out_mi_0vtk, renderView1)

        # show color bar/color legend
        warpByVector1Display.SetScalarBarVisibility(renderView1, True)

        # update the view to ensure updated data information
        renderView1.Update()

        # set scalar coloring
        ColorBy(warpByVector1Display, ('POINTS', 'STRAIN', 'Magnitude'))

        # Hide the scalar bar for this color map if no visible data is colored by it.
        HideScalarBarIfNotNeeded(gLOBAL_IDLUT, renderView1)

        # rescale color and/or opacity maps used to include current data range
        # warpByVector1Display.RescaleTransferFunctionToDataRange(True, False)

        # show color bar/color legend
        warpByVector1Display.SetScalarBarVisibility(renderView1, True)

        # get color transfer function/color map for 'STRAIN'
        sTRAINLUT = GetColorTransferFunction('STRAIN')

        # get opacity transfer function/opacity map for 'STRAIN'
        sTRAINPWF = GetOpacityTransferFunction('STRAIN')

        # get 2D transfer function for 'STRAIN'
        sTRAINTF2D = GetTransferFunction2D('STRAIN')

        # Rescale transfer function
        sTRAINLUT.RescaleTransferFunction(params.color_min, params.color_max)

        # Rescale transfer function
        sTRAINPWF.RescaleTransferFunction(params.color_min, params.color_max)

        # get layout
        layout1 = GetLayout()

        # layout/tab size in pixels
        layout1.SetSize(2252, 867)

        # current camera placement for renderView1
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [0.0, 0.0, 67.0]
        renderView1.CameraParallelScale = 14.142135623730951
        # renderView1.Update()
        print("Number of frames:", len(files))
        animScene = GetAnimationScene()
        animScene.PlayMode='Snap To TimeSteps'
        animScene.Play()
        # save animation
        SaveAnimation(filename=f"{params.animation_filepath}", viewOrLayout=renderView1, location=16, ImageResolution=[2252, 866],
            FrameWindow=[0, len(files)-1], 
            FrameRate=1,
        )
        print('Saved animation')

params = AttrDict()
params.data_dir = sys.argv[1] 
params.animation_filepath = sys.argv[2] 
params.color_min = float(sys.argv[3])
params.color_max = float(sys.argv[4])
params.csv_filepath = sys.argv[5]   
if len(sys.argv) >= 1:
    to_time(params)
    animation(params)
