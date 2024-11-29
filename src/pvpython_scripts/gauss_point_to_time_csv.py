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
        SaveData(params.csv_filepath, proxy=plotSelectionOverTime1, RowDataArrays=['N', 'Time', 'avg(ElasticStrain (0))', 'avg(ElasticStrain (1))', 'avg(ElasticStrain (2))', 'avg(ElasticStrain (3))', 'avg(ElasticStrain (4))', 'avg(ElasticStrain (5))', 'avg(ElasticStrain (6))', 'avg(ElasticStrain (7))', 'avg(ElasticStrain (8))', 'avg(ElasticStrain (Magnitude))', 'avg(EquivalentPlasticStrain)', 'avg(GLOBAL_ID)', 'avg(PlasticStrain (0))', 'avg(PlasticStrain (1))', 'avg(PlasticStrain (2))', 'avg(PlasticStrain (3))', 'avg(PlasticStrain (4))', 'avg(PlasticStrain (5))', 'avg(PlasticStrain (6))', 'avg(PlasticStrain (7))', 'avg(PlasticStrain (8))', 'avg(PlasticStrain (Magnitude))', 'avg(Strain (0))', 'avg(Strain (1))', 'avg(Strain (2))', 'avg(Strain (3))', 'avg(Strain (4))', 'avg(Strain (5))', 'avg(Strain (6))', 'avg(Strain (7))', 'avg(Strain (8))', 'avg(Strain (Magnitude))', 'avg(Stress (0))', 'avg(Stress (1))', 'avg(Stress (2))', 'avg(Stress (3))', 'avg(Stress (4))', 'avg(Stress (5))', 'avg(Stress (6))', 'avg(Stress (7))', 'avg(Stress (8))', 'avg(Stress (Magnitude))', 'avg(U (0))', 'avg(U (1))', 'avg(U (2))', 'avg(U (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)', 'avg(yielding)'],
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