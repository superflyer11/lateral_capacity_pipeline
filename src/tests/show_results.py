import pandas as pd
import matplotlib.pyplot as plt

model_name = "ModCamClay_semiExpl"
# model_name = "ModCamClay_semiExpl"

# Load data from CSV file
file_path = r"/mofem_install/jupyter/thomas/mfront_interface/behaviours/ModCamClay_semiExpl.res"
# file_path = f"/mofem_install/jupyter/thomas/mfront_example_test/mtest/{model_name}.res"
if model_name == "ModCamClay_semiExpl":
    columns = ["time", "EXX", "EYY", "EZZ", "EXY", "EXZ", "EYZ", "SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ", 
            "ElasticStrain1", "ElasticStrain2", "ElasticStrain3", "ElasticStrain4", "ElasticStrain5", "ElasticStrain6", 
            "EquivalentPlasticStrain", "PreConsolidationPressure", "PlasticVolumetricStrain", "VolumeRatio", "StoredEnergy", "DissipatedEnergy"]
else:
    columns = ["time", "EXX", "EYY", "EZZ", "EXY", "EXZ", "EYZ", "SXX", "SYY", "SZZ", "SXY", "SXZ", "SYZ", 
            "ElasticStrain1", "ElasticStrain2", "ElasticStrain3", "ElasticStrain4", "ElasticStrain5", "ElasticStrain6", 
            "EquivalentPlasticStrain", "StoredEnergy", "DissipatedEnergy"]

save_path_deviatoric_strain = f"/mofem_install/jupyter/thomas/mfront_example_test/mtest/{model_name}_q_e.png"
save_path_hydrostatic_stress = f"/mofem_install/jupyter/thomas/mfront_example_test/mtest/{model_name}_q_p.png"

# Read the file into a DataFrame
data = pd.read_csv(file_path, sep='\s+', names=columns, comment='#')
# data.to_csv("/mofem_install/jupyter/thomas/mfront_example_test/mtest/test.csv")
# Compute deviatoric strain: ε_dev = ε_XX - (ε_XX + ε_YY + ε_ZZ) / 3
data['deviatoric_strain'] = data['EXX'] - (data['EXX'] + data['EYY'] + data['EZZ']) / 3

# Compute deviatoric stress: σ_dev = σ_XX - (σ_XX + σ_YY + σ_ZZ) / 3
data['deviatoric_stress'] = data['SXX'] - (data['SXX'] + data['SYY'] + data['SZZ']) / 3

# Compute hydrostatic stress: p = (σ_XX + σ_YY + σ_ZZ) / 3
data['hydrostatic_stress'] = (data['SXX'] + data['SYY'] + data['SZZ']) / 3
print( data['EXX'])
print( data['EYY'])
print(data['deviatoric_strain'])
print(data['deviatoric_stress'])
print(data['hydrostatic_stress'])
# Plot deviatoric stress vs deviatoric strain
plt.figure(figsize=(10, 6))
plt.plot(data['deviatoric_strain'], data['deviatoric_stress'], label="Deviatoric Stress vs Deviatoric Strain", marker='o')
plt.xlabel("Deviatoric Strain")
plt.ylabel("Deviatoric Stress (MPa)")
plt.title("Deviatoric Stress vs Deviatoric Strain")
plt.grid(True)
plt.legend()

plt.savefig(save_path_deviatoric_strain)

# Plot deviatoric stress vs hydrostatic stress
plt.figure(figsize=(10, 6))
plt.plot(data['hydrostatic_stress'], data['deviatoric_stress'], label="Deviatoric Stress vs Hydrostatic Stress", color='r', marker='o')
plt.xlabel("Hydrostatic Stress (MPa)")
plt.ylabel("Deviatoric Stress (MPa)")
plt.title("Deviatoric Stress vs Hydrostatic Stress")
plt.grid(True)
plt.legend()

# Display the plots
plt.savefig(save_path_hydrostatic_stress)
