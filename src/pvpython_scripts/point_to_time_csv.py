import sys
import subprocess
from paraview.simple import *

class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value

def to_time(params):
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
        
        sel1 = QuerySelect(QueryString=f'(pointIsNear([({params.point_x}, {params.point_y}, {params.point_z}),], 1, inputs))', FieldType='POINT', InsideOut=0)
        extractSelection1 = ExtractSelection(registrationName='ExtractSelection1', Input=out_mi_0vtk)
        plotSelectionOverTime1 = PlotSelectionOverTime(Input=out_mi_0vtk, Selection=sel1)
        SaveData(params.csv_filepath, proxy=plotSelectionOverTime1, RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
        FieldAssociation='Row Data',
        AddTimeStep=1,
        AddTime=1)
        
params = AttrDict()
params.vtk_dir = sys.argv[1] 
params.csv_filepath = sys.argv[2]
params.point_x = str(sys.argv[3])
params.point_y = str(sys.argv[4])
params.point_z = str(sys.argv[5])
to_time(params)
