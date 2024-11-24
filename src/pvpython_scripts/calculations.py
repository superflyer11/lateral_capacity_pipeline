import sys
import subprocess
from paraview.simple import *
import numpy as np

class AttrDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError(f"'AttrDict' object has no attribute '{attr}'")
    def __setattr__(self, key, value):
        self[key] = value


def calculate(params):
    paraview.simple._DisableFirstRenderCameraReset()
    renderView1 = GetActiveViewOrCreate('RenderView')
    vtk_files = subprocess.run(f"ls -c1 {params.vtk_dir}/*.vtk  | sort -V", shell=True, text=True, capture_output=True)
    if vtk_files.returncode == 0:
        files = [vtk_file for vtk_file in vtk_files.stdout.splitlines()]
        out_mi_0vtk = LegacyVTKReader(registrationName="final", FileNames=files)
        # create a new 'Point Data to Cell Data'
        pointDatatoCellData2 = PointDatatoCellData(registrationName='PointDatatoCellData2', Input=out_mi_0vtk)
        
        
    def process_composite_dataset(input0): 
        # Pick up input arrays
        xxar = input0.CellData["EPSXX"]
        xyar = input0.CellData["EPSXY"]
        zxar = input0.CellData["EPSZX"]
        yyar = input0.CellData["EPSYY"]
        yzar = input0.CellData["EPSYZ"]
        zzar = input0.CellData["EPSZZ"]
        
        outarray0 = xxar*0.5
        outarray1 = xxar*0.5
        outarray2 = xxar*0.5
        
        # Run a for loop over all blocks
        numsubarrays = len(xxar.Arrays)
        for ii in range(0, numsubarrays):

            # pick up input arrays for each block.
            xxarsub = xxar.Arrays[ii]
            xyarsub = xyar.Arrays[ii]
            zxarsub = zxar.Arrays[ii]
            yyarsub = yyar.Arrays[ii]
            yzarsub = yzar.Arrays[ii]
            zzarsub = zzar.Arrays[ii]

            #print `xxarsub`

            # Transpose and calculate the principle strain.
            strain = np.transpose( 
                np.array( 
                    [ [xxarsub, xyarsub, zxarsub],
                    [xyarsub, yyarsub, yzarsub],
                    [zxarsub, yzarsub, zzarsub] ] ),
                        (2,0,1))

            principal_strain = np.linalg.eigvalsh(strain)

            # Move principle strain to temp output arrays for this block
            outarray0.Arrays[ii] = principal_strain[:,0]
            outarray1.Arrays[ii] = principal_strain[:,1]
            outarray2.Arrays[ii] = principal_strain[:,2]

        #ps0 = principal_strain[:,0]
        #print "ps0 len: " + str(len(ps0))

        # Finally, move the temp arrays to output arrays
        output.CellData.append(outarray0, "principal_strain_0")
        output.CellData.append(outarray1, "principal_strain_1")
        output.CellData.append(outarray2, "principal_strain_2")


    def process_unstructured_dataset(input0):

        # Pick up input arrays
        xxar = input0.CellData["EPSXX"]
        xyar = input0.CellData["EPSXY"]
        zxar = input0.CellData["EPSZX"]
        yyar = input0.CellData["EPSYY"]
        yzar = input0.CellData["EPSYZ"]
        zzar = input0.CellData["EPSZZ"]

        #print `xxar`
        #print len(xxar.Arrays)

        # Transpose and calculate the principle strain.
        strain = np.transpose( 
            np.array( 
                [ [xxar, xyar, zxar],
                [xyar, yyar, yzar],
                [zxar, yzar, zzar] ] ),
                    (2,0,1))

        principal_strain = np.linalg.eigvalsh(strain)


        #ps0 = principal_strain[:,0]
        #print "ps0 len: " + str(len(ps0))

        # Finally, move the temp arrays to output arrays
        output.CellData.append(principal_strain[:,0],
            "principal_strain_0")
        output.CellData.append(principal_strain[:,1],
            "principal_strain_1")
        output.CellData.append(principal_strain[:,2],
            "principal_strain_2")

    input0 = inputs[0]

    if input0.IsA("vtkCompositeDataSet"):
        process_composite_dataset(input0)
    elif input0.IsA("vtkUnstructuredGrid"):
        process_unstructured_dataset(input0)
    else:
        print("Bad dataset type for this script")
        print("\n")
            
    
        
params = AttrDict()
params.vtk_dir = sys.argv[1] 
to_time(params)
