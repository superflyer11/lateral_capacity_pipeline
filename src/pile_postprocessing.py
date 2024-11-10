import os
import sys
import subprocess
# import custom_models as cm
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
    renderView1 = GetActiveViewOrCreate('RenderView') #idk where to put this
    
    vtk_files = subprocess.run(f"ls -c1 {params.data_dir}/*.vtk | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        
        files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        reader = LegacyVTKReader(registrationName="final", FileNames=files)
        if reader:
            print("Read Success")
        else:
            print("Read Failed")
        sel1 = QuerySelect(QueryString='(pointIsNear([(1, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=reader)
        plotSelectionOverTime1 = PlotSelectionOverTime(Input=reader, Selection=sel1)
        SaveData(f'{params.data_dir}/dis_to_time_compression_mudline_1.0.csv', proxy=plotSelectionOverTime1, RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
        FieldAssociation='Row Data',
        AddTimeStep=1,
        AddTime=1)
        sel2 = QuerySelect(QueryString='(pointIsNear([(1.1, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection2 = ExtractSelection(registrationName='ExtractSelection2', Input=reader)
        plotSelectionOverTime2 = PlotSelectionOverTime(Input=reader, Selection=sel2)
        SaveData(f'{params.data_dir}/dis_to_time_compression_mudline_1.1.csv', proxy=plotSelectionOverTime2, RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
        FieldAssociation='Row Data',
        AddTimeStep=1,
        AddTime=1)
        sel3 = QuerySelect(QueryString='(pointIsNear([(1.2, 0, 0),], 0.1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection2 = ExtractSelection(registrationName='ExtractSelection2', Input=reader)
        plotSelectionOverTime2 = PlotSelectionOverTime(Input=reader, Selection=sel2)
        SaveData(f'{params.data_dir}/dis_to_time_compression_mudline_1.2.csv', proxy=plotSelectionOverTime2, RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
        FieldAssociation='Row Data',
        AddTimeStep=1,
        AddTime=1)
        

    
@ut.track_time("EXTRACTING DATA AT FINAL TIMESTEP (OR AVG OVER TIMESTEP) FROM .vtk")
def to_depth(params):
    vtk_files = subprocess.run(f"ls -c1 {params.data_dir}/*.vtk  | sort -Vr", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_files.stdout.splitlines()[0]]
        reader = LegacyVTKReader(registrationName="final", FileNames=files)
        if reader:
            print("Read Success")
        else:
            print("Read Failed")
        #data over on the compression
        plotOverLine2 = PlotOverLine(registrationName='PlotOverLine2', Input=reader)
        plotOverLine2.Point1 = [1.0, 0.0, 0.0]
        plotOverLine2.Point2 = [1.0, 0.0, -40.0]
        SaveData(f'{params.data_dir}/dis_to_depth_compression_x_1.csv', proxy=plotOverLine2, PointDataArrays=['DISPLACEMENT',  'PARALLEL_PARTITION', 'STRAIN', 'STRESS', ])

        #data over on the compression
        plotOverLine3 = PlotOverLine(registrationName='PlotOverLine3', Input=reader)
        plotOverLine3.Point1 = [1.1, 0.0, 0.0]
        plotOverLine3.Point2 = [1.1, 0.0, -40.0]
        SaveData(f'{params.data_dir}/dis_to_depth_compression_x_1.1.csv', proxy=plotOverLine3, PointDataArrays=['DISPLACEMENT',  'PARALLEL_PARTITION', 'STRAIN', 'STRESS', ])
        
        #data over on the compression
        plotOverLine3 = PlotOverLine(registrationName='PlotOverLine12', Input=reader)
        plotOverLine3.Point1 = [1.2, 0.0, 0.0]
        plotOverLine3.Point2 = [1.2, 0.0, -40.0]
        SaveData(f'{params.data_dir}/dis_to_depth_compression_x_1.2.csv', proxy=plotOverLine3, PointDataArrays=['DISPLACEMENT',  'PARALLEL_PARTITION', 'STRAIN', 'STRESS', ])
        
        
        #data over on the tension
        plotOverLine4 = PlotOverLine(registrationName='plotOverLine4', Input=reader)
        plotOverLine4.Point1 = [-1.0, 0.0, 0.0]
        plotOverLine4.Point2 = [-1.0, 0.0, -40.0]
        SaveData(f'{params.data_dir}/dis_to_depth_tension_x_-1.csv', proxy=plotOverLine4, PointDataArrays=['DISPLACEMENT',  'PARALLEL_PARTITION', 'STRAIN', 'STRESS', ])
        

params = AttrDict()
params.data_dir = sys.argv[1] 
if len(sys.argv) >= 1:
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

