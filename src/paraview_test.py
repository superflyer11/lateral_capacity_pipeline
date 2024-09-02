import sys

from paraview.simple import *

import utils as ut

class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value

@ut.track_time("EXTRACTING GROUP DATA FROM .vtk")
def to_time(params):    
    paraview.simple._DisableFirstRenderCameraReset()
    renderView1 = GetActiveViewOrCreate('RenderView') #idk where to put this
    
    vtk_files = subprocess.run(f"ls -c1 {params.data_dir}*.vtk", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        reader = LegacyVTKReader(registrationName=params.vtk_filepath, FileNames=[os.path.join(params.data_dir, vtk_file) for vtk_file in vtk_files])
        if reader:
            print("Success")
        else:
            print("Failed")
        #getting a single point at mudline compression
        QuerySelect(QueryString='(pointIsNear([(1, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=reader)
        # help(SaveData)
        # help(QuerySelect)
        SaveData(f'{params.data_dir}/dis_to_time_compression_mudline.csv', proxy=extractSelection1, PointDataArrays=['DISPLACEMENT', 'GLOBAL_ID', 'STRAIN', 'STRESS'])
    
@ut.track_time("EXTRACTING GROUP DATA FROM .vtk")
def to_depth(params):
    reader = LegacyVTKReader(registrationName=params.vtk_filepath, FileNames=[params.vtk_filepath])
    #data over on the compression
    plotOverLine2 = PlotOverLine(registrationName='PlotOverLine2', Input=reader)
    plotOverLine2.Point1 = [1.0, 0.0, 0.0]
    plotOverLine2.Point2 = [1.0, 0.0, -40.0]
    SaveData(f'{params.data_dir}/dis_to_depth_compression_x_1.csv', proxy=plotOverLine2, PointDataArrays=['DISPLACEMENT', 'GLOBAL_ID', 'NB_IN_THE_LOOP', 'PARALLEL_PARTITION', 'STRAIN', 'STRESS', 'arc_length'])

    #data over on the compression
    plotOverLine3 = PlotOverLine(registrationName='PlotOverLine3', Input=reader)
    plotOverLine3.Point1 = [1.1, 0.0, 0.0]
    plotOverLine3.Point2 = [1.1, 0.0, -40.0]
    SaveData(f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv', proxy=plotOverLine3, PointDataArrays=['DISPLACEMENT', 'GLOBAL_ID', 'NB_IN_THE_LOOP', 'PARALLEL_PARTITION', 'STRAIN', 'STRESS', 'arc_length'])
    
    
    #data over on the tension
    plotOverLine4 = PlotOverLine(registrationName='plotOverLine4', Input=reader)
    plotOverLine4.Point1 = [-1.0, 0.0, 0.0]
    plotOverLine4.Point2 = [-1.0, 0.0, -40.0]
    SaveData(f'{params.data_dir}/dis_to_depth_tension_x_-1.csv', proxy=plotOverLine4, PointDataArrays=['DISPLACEMENT', 'GLOBAL_ID', 'NB_IN_THE_LOOP', 'PARALLEL_PARTITION', 'STRAIN', 'STRESS', 'arc_length'])
        

params = AttrDict()
params.vtk_filepath = sys.argv[1] 
params.data_dir = sys.argv[2] 
if len(sys.argv) > 1:
    to_time(params)
    to_depth(params)

        
# # create a query selection
# QuerySelect(QueryString='(mag(STRESS) == max(mag(STRESS)))', FieldType='POINT', InsideOut=0)

# # create a query selection
# QuerySelect(QueryString='(mag(STRESS) == max(mag(STRESS)))', FieldType='POINT', InsideOut=0)

# # create a new 'Extract Selection'
# extractSelection3 = ExtractSelection(registrationName='ExtractSelection3', Input=fx_18000000_fy_0_fz_0_2024_08_16_16_20_44vtk)


# QuerySelect(QueryString='(mag(STRESS) == max(mag(STRESS)))&(pointIsNear([(1, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)


    

# # trace generated using paraview version 5.12.0
# #import paraview
# #paraview.compatibility.major = 5
# #paraview.compatibility.minor = 12

# #### import the simple module from the paraview
# from paraview.simple import *
# #### disable automatic camera reset on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()

# # create a new 'Legacy VTK Reader'
# fx_18000000_fy_0_fz_0_2024_08_16_16_20_44vtk = LegacyVTKReader(registrationName='fx_-18000000_fy_0_fz_0_2024_08_16_16_20_44.vtk', FileNames=['C:\\Users\\laito\\Downloads\\fx_-18000000_fy_0_fz_0_2024_08_16_16_20_44.vtk'])

# # create a query selection
# QuerySelect(QueryString='(pointIsNear([(1, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)

# # create a new 'Extract Selection'
# extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=fx_18000000_fy_0_fz_0_2024_08_16_16_20_44vtk)

# # save data
# SaveData('C:\\Users\\laito\\Downloads\\test2.csv', proxy=extractSelection1, PointDataArrays=['DISPLACEMENT', 'GLOBAL_ID', 'STRAIN', 'STRESS'])

# # get active view
# renderView1 = GetActiveViewOrCreate('RenderView')

# #================================================================
# # addendum: following script captures some of the application
# # state to faithfully reproduce the visualization during playback
# #================================================================

# # get layout
# layout1 = GetLayout()

# #--------------------------------
# # saving layout sizes for layouts

# # layout/tab size in pixels
# layout1.SetSize(922, 737)

# #-----------------------------------
# # saving camera placements for views

# # current camera placement for renderView1
# renderView1.CameraPosition = [-6.396272633682902, 33.5676966387846, 11.175445242047234]
# renderView1.CameraFocalPoint = [0.0, -39.999999999999986, -14.542238712310803]
# renderView1.CameraViewUp = [0.19613499446427154, -0.30834700637642126, 0.93083467254137]
# renderView1.CameraParallelScale = 92.99514831312958


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------

