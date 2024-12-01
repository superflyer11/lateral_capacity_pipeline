def postprocessing(params):
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ""

    def run_command(command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')  # Print to console
        process.stdout.close()
        process.wait()
        return process.returncode
    
    @ut.track_time("PULLING A SELECTED POINT OVER TIME WITH pvpython")
    def point_to_csv(params, point: cm.Point):
        command = [
            params.paraview_path,
            "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/gauss_point_to_time_csv.py" if params.save_gauss == 1 else "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/point_to_time_csv.py",
            params.vtk_gauss_dir if params.save_gauss == 1 else params.vtk_dir,
            # params.vtk_dir,
            point.point_against_time_csv_filepath(params),
            *point.flat(),
        ]
        # Run the command using subprocess
        run_command(command)
        
    @ut.track_time("PULLING MULTIPLE SELECTED POINTS OVER TIME WITH pvpython")
    def multiple_points_to_csv(params, points_of_interest: list[cm.Point]):
        command = [
            params.paraview_path,
            "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/gauss_multiple_points_to_time_csv.py" if params.save_gauss == 1 else "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/multiple_points_to_time_csv.py",
            params.vtk_gauss_dir if params.save_gauss == 1 else params.vtk_dir,
            # params.vtk_dir,
            str(len(points_of_interest)),
        ]
        for point in points_of_interest:
            command.extend(point.flat())
            command.append(point.point_against_time_csv_filepath(params))
        # Run the command using subprocess
        run_command(command)
        
    @ut.track_time("PULLING A SELECTED LINE OVER DEPTH AT THE FINAL TIMESTEP WITH pvpython")
    def line_to_csv(params, line: cm.Line):
        command = [
            params.paraview_path,
            "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/gauss_line_to_depth_csv.py" if params.save_gauss == 1 else "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/line_to_depth_csv.py",
            params.vtk_gauss_dir if params.save_gauss == 1 else params.vtk_dir,
            # params.vtk_dir,
            line.line_against_depth_csv_filepath(params),
            *line.pt1.flat(),
            *line.pt2.flat(),
        ]
        # Run the command using subprocess
        run_command(command)


    @ut.track_time("PLOTTING COLOR CONTOUR MAP WITH pvpython")
    def contours(params):
        command = [
            params.paraview_path,
            "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/contours.py",
            params.vtk_gauss_dir if params.save_gauss == 1 else params.vtk_dir,
            params.vtk_dir,
            params.graph_dir,
            params.preset_dir,
        ]
        # Run the command using subprocess
        run_command(command)

    

    @ut.track_time("ANIMATING OVER TIME WITH pvpython")
    def animate(params, color_min, color_max):
        command = [
            params.paraview_path,
            "/mofem_install/jupyter/thomas/mfront_example_test/src/pvpython_scripts/pile_animate.py",
            params.vtk_dir,
            params.strain_animation_filepath_png,
            str(color_min),
            str(color_max),
        ]
        # Run the command using subprocess
        try:
            process = subprocess.run(command, check=True, capture_output=True)
            create_mp4_from_png_sequence(params.strain_animation_filepath_png_ffmpeg_regex, params.strain_animation_filepath_mp4, framerate=40)
            
        except subprocess.CalledProcessError as e:
            print("An error occurred:", e.stderr.decode())
        finally:
        # Restore the original PYTHONPATH
            os.environ["PYTHONPATH"] = original_pythonpath

    #Stitch together animations
    @ut.track_time("STITCHING .pngs TOGETHER with ffmpeg")
    def create_mp4_from_png_sequence(animation_filepath_png_ffmpeg_regex, animation_filepath_mp4, framerate=40):
        # Build the ffmpeg command
        ffmpeg_command = [
            params.ffmpeg_path,
            '-framerate', str(framerate),  # Set input framerate
            '-y',
            '-i', animation_filepath_png_ffmpeg_regex,  # Input image sequence (with regex pattern)
            '-c:v', 'libx264',  # Set video codec to libx264
            '-pix_fmt', 'yuv420p',  # Set pixel format for compatibility
            '-loglevel', 'warning',
            animation_filepath_mp4  # Output .mp4 file
        ]

        # Run the ffmpeg command as a subprocess
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True)
            print(f"MP4 video created successfully: {animation_filepath_mp4}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during ffmpeg execution: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    try:
        # for point in points_of_interest:
            # point_to_csv(params, point)
        multiple_points_to_csv(params, params.points_of_interest)
        # line_to_csv(params, params.line_of_interest)
        # contours(params)
        # df = pd.read_csv(params.points_of_interest[1].point_against_time_csv_filepath(params))
        # strain_magnitude = np.array(df['avg(STRAIN (Magnitude))'])
        # color_max = strain_magnitude.max()
        # color_min = strain_magnitude.min()
        # animate(params, color_min, color_max)
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e.stderr.decode())
    finally:
        # Restore the original PYTHONPATH
        os.environ["PYTHONPATH"] = original_pythonpath
        
    
   
def plot_all_and_auxiliary_saves(params):
    # if params.save_gauss == 1:
    #     pass
    # else:
    #     df = pd.read_csv(params.line_of_interest.line_against_depth_csv_filepath(params))
    #     df = df.dropna()
    #     x = np.array(df['Points:0'])
    #     y = np.array(df['Points:1'])
    #     z = np.array(df['Points:2'])
    #     disp_x = np.array(df['DISPLACEMENT:0'])
    #     disp_y = np.array(df['DISPLACEMENT:1'])
    #     disp_z = np.array(df['DISPLACEMENT:2'])
        
    #     e_xx = np.array(df['STRAIN:0'])
    #     e_xy = np.array(df['STRAIN:1'])
    #     e_xz = np.array(df['STRAIN:2'])
    #     e_yy = np.array(df['STRAIN:4'])
    #     e_yz = np.array(df['STRAIN:5'])
    #     e_zz = np.array(df['STRAIN:8'])
        
    #     sig_xx = np.array(df['STRESS:0'])
    #     sig_xy = np.array(df['STRESS:1'])
    #     sig_xz = np.array(df['STRESS:2'])
    #     sig_yy = np.array(df['STRESS:4'])
    #     sig_yz = np.array(df['STRESS:5'])
    #     sig_zz = np.array(df['STRESS:8'])
        
    #     sig_1, sig_2, sig_3 = calc.calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz)
    #     p = calc.calculate_p(sig_1, sig_2, sig_3)
    #     J_2 = calc.calculate_J2(sig_1, sig_2, sig_3)
    #     J  = np.sqrt(J_2)
    #     tau_oct = np.sqrt(2 * J_2)
    #     sig_eq = np.sqrt(3 * J_2)
    #     e_v, e_d = calc.calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)
    #     graph_dir = params.line_of_interest.graph_dir(params)
    #     if params.save_gauss == 1:
    #         pass
    #     else:
    #         plotting.plot_x_ys(disp_x, [z], labels=["FEA"], x_label='$\mu_x$ [m]', y_label='Depth [m]', title='Depth vs $\mu_x$', save_as = f"{graph_dir}/401_z_ux.png")

    # for point in params.points_of_interest:
    #     graph_dir = point.graph_dir(params)
    #     df = pd.read_csv(point.point_against_time_csv_filepath(params))
    #     if params.save_gauss == 1:
    #         disp_x = np.array(df['avg(U (0))'])
    #         disp_y = np.array(df['avg(U (1))'])
    #         disp_z = np.array(df['avg(U (2))'])
            
    #         e_xx = np.array(df['avg(Strain (0))'])
    #         e_xy = np.array(df['avg(Strain (1))'])
    #         e_xz = np.array(df['avg(Strain (2))'])
    #         e_yy = np.array(df['avg(Strain (4))'])
    #         e_yz = np.array(df['avg(Strain (5))'])
    #         e_zz = np.array(df['avg(Strain (8))'])
                
    #         sig_xx = np.array(df['avg(Stress (0))'])
    #         sig_xy = np.array(df['avg(Stress (1))'])
    #         sig_xz = np.array(df['avg(Stress (2))'])
    #         sig_yy = np.array(df['avg(Stress (4))'])
    #         sig_yz = np.array(df['avg(Stress (5))'])
    #         sig_zz = np.array(df['avg(Stress (8))'])
    #     else:
            
    #         disp_x = np.array(df['avg(DISPLACEMENT (0))'])
    #         disp_y = np.array(df['avg(DISPLACEMENT (1))'])
    #         disp_z = np.array(df['avg(DISPLACEMENT (2))'])
            
    #         e_xx = np.array(df['avg(STRAIN (0))'])
    #         e_xy = np.array(df['avg(STRAIN (1))'])
    #         e_xz = np.array(df['avg(STRAIN (2))'])
    #         e_yy = np.array(df['avg(STRAIN (4))'])
    #         e_yz = np.array(df['avg(STRAIN (5))'])
    #         e_zz = np.array(df['avg(STRAIN (8))'])
                
    #         sig_xx = np.array(df['avg(STRESS (0))'])
    #         sig_xy = np.array(df['avg(STRESS (1))'])
    #         sig_xz = np.array(df['avg(STRESS (2))'])
    #         sig_yy = np.array(df['avg(STRESS (4))'])
    #         sig_yz = np.array(df['avg(STRESS (5))'])
    #         sig_zz = np.array(df['avg(STRESS (8))'])

    #     time = np.linspace(0, 1, len(sig_xx))
        
    #     sig_1, sig_2, sig_3 = calc.calculate_principal_stresses(sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz)
    #     p = calc.calculate_p(sig_1, sig_2, sig_3)
    #     J_2 = calc.calculate_J2(sig_1, sig_2, sig_3)
    #     J  = np.sqrt(J_2)
    #     tau_oct = np.sqrt(2 * J_2)
    #     sig_eq = np.sqrt(3 * J_2)
    #     e_v, e_d = calc.calculate_volumetric_and_deviatoric_strain(e_xx, e_yy, e_zz, e_xy, e_xz, e_yz)

    #     plotting.plot_stress_field(sig_1, sig_2, sig_3, 
    #                     #   tau_oct, 
    #                     #   cone_radius, 
    #                     #   cone_tip_p = cone_tip_p, 
    #                     #   max_p = np.max(p),
    #                       elev=np.degrees(np.arccos(np.sqrt(2/3))),
    #                       azim=45,
    #                       roll=0,
    #                       save_as=f"{graph_dir}/100_stress_field.png",
    #                       show=False,
    #                       )
    #     plotting.plot_stress_field(sig_1, sig_2, sig_3, 
    #                     #   tau_oct, 
    #                     #   cone_radius, 
    #                     #   cone_tip_p = cone_tip_p, 
    #                     #   max_p = np.max(p),
    #                       elev=np.degrees(np.arccos(np.sqrt(2/3))),
    #                       azim=-45,
    #                       roll=90,
    #                       save_as=f"{graph_dir}/100_1_stress_field.png",
    #                       show=False,
    #                       )

    #     label = f"""{params.soil_model.value} (FEA) 
    #     at {point.string()}"""
    #     plotting.plot_x_ys(p, [sig_eq], labels=[label], x_label='Hydrostatic stress $p$', y_label='Equivalent stress $sig_{eq}$', title='Equivalent Stress vs Hydrostatic stress', save_as = f"{graph_dir}/111_sigeq_p.png")

    #     plotting.plot_x_ys(time, [sig_eq], labels=[label], x_label='Time $t$', y_label='Equivalent Stress $\sigma_{eq}$', title='Equivalent Stress $\sigma_{eq}$ vs Time $t$', save_as = f"{graph_dir}/201_sigeq_t.png")
    #     plotting.plot_x_ys(e_zz, [sig_zz], labels=["test1"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress $\sigma_{zz}$', title='Stress $\sigma_{zz}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{graph_dir}/202_sigeq_ezz.png")
    #     plotting.plot_x_ys(e_zz, [sig_yy], labels=["test1"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress $\sigma_{yy}$', title='Stress $\sigma_{yy}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{graph_dir}/203_sigeq_eyy.png")
    #     plotting.plot_x_ys(e_zz, [sig_xx], labels=["test1"], x_label='Axial Strain $\epsilon_{zz}$', y_label='Stress $\sigma_{xx}$', title='Stress $\sigma_{xx}$ vs Axial Strain $\epsilon_{zz}$', save_as = f"{graph_dir}/204_sigeq_exx.png")

    #     plotting.plot_x_ys(e_zz[1:], [sig_eq[1:]/p[1:]], labels=["test1"], x_label='Axial Strain $\epsilon_{yy}$', y_label='Stress ratio $q/p$', title='Stress ratio vs Axial Strain', save_as = f"{graph_dir}/211_sigeq_div_p_eyy.png")

    #     plotting.plot_x_ys(e_d, [e_v], labels=["volumetric - axial"], x_label='Axial strain $\epsilon_{zz}$', y_label='Volumetric strain $\epsilon^v$', title='Volumetric strain vs Axial strain', save_as = f"{graph_dir}/302_ev_ed.png")
    #     plotting.plot_x_ys(e_d, [e_d], labels=["Deviatoric - axial"], x_label='Axial strain $\epsilon_{zz}$', y_label='Deviatoric strain $\epsilon^v$', title='Deviatoric strain vs Axial strain', save_as = f"{graph_dir}/303_ev_ed.png")
    #     plotting.plot_x_ys(e_d, [e_v], labels=["FEA"], x_label='Deviatoric strain $\epsilon^d$', y_label='Volumetric strain $\epsilon^v$', title='Volumetric strain vs Deviatoric strain', save_as = f"{graph_dir}/304_ev_ed.png")
    #     if params.save_gauss == 1:
    #         pass
    #     else:
    #         plotting.plot_x_ys(disp_x, [sig_eq], labels=["FEA"], x_label='$\mu_x$ [m]', y_label='Equivalent stress $sig_{eq}$', title='Equivalent Stress vs $\mu_x$', save_as = f"{graph_dir}/411_ux_sigeq.png")
    #         plotting.plot_x_ys(disp_x, [sig_xx], labels=["FEA"], x_label='$\mu_x$ [m]', y_label='Stress $\sigma_{xx}$', title='Stress $\sigma_{xx}$ vs $\mu_x$', save_as = f"{graph_dir}/412_ux_sigxx.png")
        
    data_force=pd.read_csv(params.FIX_X_1_force_log_file,sep='\s+',header=None)
    pile_tip_lateral_load = - data_force[4].values * 2 * (10 ** 6) / 1000
    
    df_ground_level_passive = pd.read_csv("/mofem_install/jupyter/thomas/mfront_example_test/simulations/pile_day_104_sim_1_20241129_010700_vM/-1.0_0.0_0.0/point_data_71015.csv")
    ground_level_displacement = - np.array(df_ground_level_passive['DISPLACEMENT_0']) * 1000
    
    # if params.save_gauss == 1:
    #     ground_level_displacement = - np.array(df_ground_level_passive['avg(U (0))']) * 1000
    # else:
    #     ground_level_displacement = - np.array(df_ground_level_passive['avg(DISPLACEMENT (0))']) * 1000
    print(len(pile_tip_lateral_load))
    print(len(ground_level_displacement))
    
    # Create a DataFrame from the arrays
    df_final = pd.DataFrame({
        'ground_level_displacement': ground_level_displacement,
        'pile_tip_lateral_load': pile_tip_lateral_load
    })
    # Save the DataFrame to a CSV file
    df_final.to_csv(f"{params.data_dir}/ground_level_displacement_vs_pile_tip_lateral_load.csv", index=False)
    
    plotting.plot_x_ys(ground_level_displacement, [pile_tip_lateral_load], labels=["FEA"], x_label='Ground-level displacement$\mu_x$ [mm]', y_label='Lateral load $H$ [kN]', title='Lateral load $H$ vs $\mu_x$', save_as = f"{params.graph_dir}/412_H_uxpng", show=True)