import numpy as np

# Function to calculate principal stresses and directions
# def calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz):
#     sig_1 = []
#     sig_2 = []
#     sig_3 = []

#     for i in range(len(sig_xx)):
#         # Create stress tensor
#         stress_tensor = np.array([
#             [sig_xx[i], sig_xy[i], sig_xz[i]],
#             [sig_xy[i], sig_yy[i], sig_yz[i]],
#             [sig_xz[i], sig_yz[i], sig_zz[i]]
#         ])

#         # Calculate principal stresses (eigenvalues)
#         principal_stresses, _ = np.linalg.eigh(stress_tensor)
#         principal_stresses = np.sort(principal_stresses)[::-1]  # Sort in descending order

#         # Append principal stresses to respective lists
#         sig_1.append(principal_stresses[0])
#         sig_2.append(principal_stresses[1])
#         sig_3.append(principal_stresses[2])

#     # Convert lists to numpy arrays
#     sig_1 = np.array(sig_1)
#     sig_2 = np.array(sig_2)
#     sig_3 = np.array(sig_3)

#     return sig_1, sig_2, sig_3

# import numpy as np

def calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz):
    sig_1 = []
    sig_2 = []
    sig_3 = []

    for i in range(len(sig_xx)):
        # Create stress tensor
        stress_tensor = np.array([
            [sig_xx[i], sig_xy[i], sig_xz[i]],
            [sig_xy[i], sig_yy[i], sig_yz[i]],
            [sig_xz[i], sig_yz[i], sig_zz[i]]
        ])

        # Check for invalid values
        if not np.isfinite(stress_tensor).all():
            raise ValueError(f"Invalid values in stress tensor: {stress_tensor}")

        # Ensure tensor symmetry
        if not np.allclose(stress_tensor, stress_tensor.T, atol=1e-8):
            raise ValueError(f"Stress tensor is not symmetric: {stress_tensor}")

        try:
            # Calculate principal stresses (eigenvalues)
            principal_stresses, _ = np.linalg.eigh(stress_tensor)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Eigenvalue computation failed for tensor: {stress_tensor}") from e

        # Sort principal stresses in descending order
        principal_stresses = np.sort(principal_stresses)[::-1]

        # Append principal stresses to respective lists
        sig_1.append(principal_stresses[0])
        sig_2.append(principal_stresses[1])
        sig_3.append(principal_stresses[2])

    # Convert lists to numpy arrays
    sig_1 = np.array(sig_1)
    sig_2 = np.array(sig_2)
    sig_3 = np.array(sig_3)

    return sig_1, sig_2, sig_3


def calculate_p(sig_1, sig_2, sig_3):
    return (sig_1 + sig_2 + sig_3) / 3

def calculate_J2(sig_1, sig_2, sig_3):
    J2_list = (1/6) * ((sig_1 - sig_2) ** 2 + (sig_2 - sig_3) ** 2 + (sig_3 - sig_1) ** 2)
    
    return J2_list

# Function to calculate volumetric strain and deviatoric strain
def calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz):
    volumetric_strain_list = []
    deviatoric_strain_list = []

    for i in range(len(e_xx)):
        # Volumetric strain is the trace of the strain tensor
        volumetric_strain = e_xx[i] + e_yy[i] + e_zz[i]
        volumetric_strain_list.append(volumetric_strain)

        # Deviatoric strain components
        e_mean = volumetric_strain / 3
        e_dev_xx = e_xx[i] - e_mean
        e_dev_yy = e_yy[i] - e_mean
        e_dev_zz = e_zz[i] - e_mean
        e_dev_xy = e_xy[i]
        e_dev_xz = e_xz[i]
        e_dev_yz = e_yz[i]

        # Deviatoric strain magnitude
        deviatoric_strain = np.sqrt(2/3 * (e_dev_xx**2 + e_dev_yy**2 + e_dev_zz**2) + 2 * (e_dev_xy**2 + e_dev_xz**2 + e_dev_yz**2))
        deviatoric_strain_list.append(deviatoric_strain)

    volumetric_strain_list = np.array(volumetric_strain_list)
    deviatoric_strain_list = np.array(deviatoric_strain_list)
    return volumetric_strain_list, deviatoric_strain_list
