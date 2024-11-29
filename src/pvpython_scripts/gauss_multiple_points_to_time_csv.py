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
        print(len(files))
        out_mi_0vtk = LegacyVTKReader(registrationName="final", FileNames=files)
        if out_mi_0vtk:
            print("Reading .vtks succeeded.")
        else:
            print("Reading .vtks failed.")
        d = {}
        for i in range(params.no_of_points):
            SetActiveSource(out_mi_0vtk)
            x = getattr(params, f'pt{i+1}_x', None)
            y = getattr(params, f'pt{i+1}_y', None)
            z = getattr(params, f'pt{i+1}_z', None)
            csv_filepath = getattr(params, f'pt{i+1}_csv_filepath', None)
            if x and y and z and csv_filepath:
                d[f"sel{i}"] = QuerySelect(QueryString=f'(pointIsNear([({x}, {y}, {z}),], 1, inputs))', FieldType='POINT', InsideOut=0)
                # d[f"extractSelection{i}"]  = ExtractSelection(registrationName=f'ExtractSelection{i}', Input=out_mi_0vtk)
                d[f"plotSelectionOverTime{i}"]  = PlotSelectionOverTime(Input=out_mi_0vtk, Selection=d[f"sel{i}"])
                SetActiveSource(d[f"plotSelectionOverTime{i}"])
                SaveData(csv_filepath, proxy=d[f"plotSelectionOverTime{i}"], RowDataArrays=['N', 'Time', 'avg(ElasticStrain (0))', 'avg(ElasticStrain (1))', 'avg(ElasticStrain (2))', 'avg(ElasticStrain (3))', 'avg(ElasticStrain (4))', 'avg(ElasticStrain (5))', 'avg(ElasticStrain (6))', 'avg(ElasticStrain (7))', 'avg(ElasticStrain (8))', 'avg(ElasticStrain (Magnitude))', 'avg(EquivalentPlasticStrain)', 'avg(GLOBAL_ID)', 'avg(PlasticStrain (0))', 'avg(PlasticStrain (1))', 'avg(PlasticStrain (2))', 'avg(PlasticStrain (3))', 'avg(PlasticStrain (4))', 'avg(PlasticStrain (5))', 'avg(PlasticStrain (6))', 'avg(PlasticStrain (7))', 'avg(PlasticStrain (8))', 'avg(PlasticStrain (Magnitude))', 'avg(Strain (0))', 'avg(Strain (1))', 'avg(Strain (2))', 'avg(Strain (3))', 'avg(Strain (4))', 'avg(Strain (5))', 'avg(Strain (6))', 'avg(Strain (7))', 'avg(Strain (8))', 'avg(Strain (Magnitude))', 'avg(Stress (0))', 'avg(Stress (1))', 'avg(Stress (2))', 'avg(Stress (3))', 'avg(Stress (4))', 'avg(Stress (5))', 'avg(Stress (6))', 'avg(Stress (7))', 'avg(Stress (8))', 'avg(Stress (Magnitude))', 'avg(U (0))', 'avg(U (1))', 'avg(U (2))', 'avg(U (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)', 'avg(yielding)'],
                FieldAssociation='Row Data',
                AddTimeStep=1,
                AddTime=1)
                # Delete(d[f"extractSelection{i}"])
                Delete(d[f"plotSelectionOverTime{i}"])
                print(f'Pulled {x} {y} {z}.')
            else:
                raise RuntimeError(f'One of x, y, z or csv_filepath was not written or read properly')
            
params = AttrDict()
params.vtk_dir = sys.argv[1] 
params.no_of_points = int(sys.argv[2])

# Dynamically pull as many points as needed based on params.no_of_points
for i in range(params.no_of_points):
    x = sys.argv[3 + i * 4]  # X coordinate
    y = sys.argv[4 + i * 4]  # Y coordinate
    z = sys.argv[5 + i * 4]  # Z coordinate
    setattr(params, f'pt{i+1}_csv_filepath', sys.argv[6 + i * 4])
    setattr(params, f'pt{i+1}_x', str(x))
    setattr(params, f'pt{i+1}_y', str(y))
    setattr(params, f'pt{i+1}_z', str(z))

to_time(params)
