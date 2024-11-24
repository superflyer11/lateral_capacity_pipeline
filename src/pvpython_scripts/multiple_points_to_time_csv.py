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
                SaveData(csv_filepath, proxy=d[f"plotSelectionOverTime{i}"], RowDataArrays=['N', 'Time', 'avg(DISPLACEMENT (0))', 'avg(DISPLACEMENT (1))', 'avg(DISPLACEMENT (2))', 'avg(DISPLACEMENT (Magnitude))', 'avg(GLOBAL_ID)', 'avg(STRAIN (0))', 'avg(STRAIN (1))', 'avg(STRAIN (2))', 'avg(STRAIN (3))', 'avg(STRAIN (4))', 'avg(STRAIN (5))', 'avg(STRAIN (6))', 'avg(STRAIN (7))', 'avg(STRAIN (8))', 'avg(STRAIN (Magnitude))', 'avg(STRESS (0))', 'avg(STRESS (1))', 'avg(STRESS (2))', 'avg(STRESS (3))', 'avg(STRESS (4))', 'avg(STRESS (5))', 'avg(STRESS (6))', 'avg(STRESS (7))', 'avg(STRESS (8))', 'avg(STRESS (Magnitude))', 'avg(X)', 'avg(Y)', 'avg(Z)', 'avg(vtkOriginalPointIds)',],
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
