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

def to_depth(params):
    paraview.simple._DisableFirstRenderCameraReset()
    renderView1 = GetActiveViewOrCreate('RenderView')
    vtk_files = subprocess.run(f"ls -c1 {params.vtk_dir}/*.vtk  | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_files.stdout.splitlines()[-1]] # only take the final timestep
        # files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        out_mi_0vtk = LegacyVTKReader(registrationName="final", FileNames=files)
        if out_mi_0vtk:
            print("Read Success")
        else:
            print("Read Failed")
        
        plotOverLine1 = PlotOverLine(registrationName='PlotOverLine2', Input=out_mi_0vtk)
        plotOverLine1.Point1 = [params.pt1_x, params.pt1_y, params.pt1_z]
        plotOverLine1.Point2 = [params.pt2_x, params.pt2_y, params.pt2_z]
        SaveData(params.csv_filepath, proxy=plotOverLine1, PointDataArrays=['DISPLACEMENT', 'PARALLEL_PARTITION', 'STRAIN', 'STRESS', ])
        
params = AttrDict()
params.vtk_dir = sys.argv[1] 
params.csv_filepath = sys.argv[2]
params.pt1_x = float(sys.argv[3])
params.pt1_y = float(sys.argv[4])
params.pt1_z = float(sys.argv[5])
params.pt2_x = float(sys.argv[6])
params.pt2_y = float(sys.argv[7])
params.pt2_z = float(sys.argv[8])
to_depth(params)
