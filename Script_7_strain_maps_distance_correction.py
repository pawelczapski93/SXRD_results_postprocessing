# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:26:41 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt
# import h5py 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate
import ast
import pyvista as pv


#the function that moves the chart to position 0,0 
def transform_motors_to_origin(X_motor, Y_motor):
    X_motor_transformed = X_motor - min(X_motor)
    Y_motor_transformed = Y_motor - min(Y_motor)
    return X_motor_transformed, Y_motor_transformed



def distance_correction(theta, distance):
    n = 1
    sample_detector_dist = 72
    theta_corrected = np.rad2deg(np.arctan(sample_detector_dist/(sample_detector_dist-distance)*np.tan(np.deg2rad(theta))))
    d_spacing = 0.065263 * n / (2 * np.sin(np.deg2rad(theta_corrected/2)))
    return d_spacing






#file names
# half_CC1_1_no_load
# half_CC1_1_one_third_buckling_load
# half_CC1_1_one_buckling_load


# #sample CC1 (CC1)
# CC1_no_load
# CC1_one_third_buckling_load
# CC1_one_buckling_load

# #sample CC2 (CC2)
# CC2_no_load
# CC2_one_third_buckling_load
# CC2_one_buckling_load

# #sample half_CC1 (1_2_CC1)
# half_CC1_2_no_load
# half_CC1_2_one_third_buckling_load
# half_CC1_2_one_buckling_load


# %%% reading the file

SL1 = ['CC1_no_load', 'CC1_one_third_buckling_load', 'CC1_one_buckling_load', 'SL1']
SL2 = ['CC2_no_load', 'CC2_one_third_buckling_load', 'CC2_one_buckling_load', 'SL2']

HS1 = ['half_CC1_1_no_load', 'half_CC1_1_one_third_buckling_load', 'half_CC1_1_one_buckling_load', 'HS1']
HS2 = ['half_CC1_2_no_load', 'half_CC1_2_one_third_buckling_load', 'half_CC1_2_one_buckling_load', 'HS2']

sample = HS2


name_0_percent_BL = sample[0]
name_33_percent_BL = sample[1]
name_100_percent_BL  = sample[2]


file_name_0_percent_BL = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\May_res/' +
             str(name_0_percent_BL) + "_results.xlsx")

file_name_33_percent_BL = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\May_res/' +
             str(name_33_percent_BL) + "_results.xlsx")

file_name_100_percent_BL = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\May_res/' +
             str(name_100_percent_BL) + "_results.xlsx")



data_0_percent_BL = pd.read_excel(file_name_0_percent_BL)
data_33_percent_BL = pd.read_excel(file_name_33_percent_BL)
data_100_percent_BL = pd.read_excel(file_name_100_percent_BL)


# %%% importing STL_file
SL1 = ['CC1_0_BL_RBC', 'CC1_33_BL_RBC', 'CC1_105_BL_RBC', 'SL1']
SL2 = ['CC2_0_BL_RBC', 'CC2_33_BL_RBC', 'CC2_105_BL_RBC', 'SL2']

HS1 = ['1_2_CC1_1_0_BL_RBC', '1_2_CC1_1_33_BL_RBC', '1_2_CC1_1_105_BL_RBC', 'HS1']
HS2 = ['1_2_CC1_1_0_BL_RBC', '1_2_CC1_2_33_BL_RBC', '1_2_CC1_2_105_BL_RBC', 'HS2']


selected_sample = HS2


file_1 = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\STL/' + \
        selected_sample[0] + '.stl'
 
file_2 = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\STL/' + \
        selected_sample[1] + '.stl'
        
file_3 = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\STL/' + \
        selected_sample[2] + '.stl'

mesh_1 = pv.read(file_1)
mesh_2 = pv.read(file_2)
mesh_3 = pv.read(file_3)


points_1 = np.array(mesh_1.points.tolist())
points_2 = np.array(mesh_2.points.tolist())
points_3 = np.array(mesh_3.points.tolist())


zi_1 = points_1[:,2]
zi_2 = points_2[:,2]
zi_3 = points_3[:,2]


#common grid
n_points_x = 240
n_points_y = 200


xi_1 = np.linspace(points_1[:,0].min(), points_1[:,0].max(), n_points_x)
yi_1 = np.linspace(points_1[:,1].min(), points_1[:,1].max(), n_points_y)
    
xi_1, yi_1 = np.meshgrid(xi_1, yi_1)
    
z_at_1 = [ points_1[np.argmax(points_1[:,0]),2]]



xi_2 = np.linspace(points_2[:,0].min(), points_2[:,0].max(), n_points_x)
yi_2 = np.linspace(points_2[:,1].min(), points_2[:,1].max(), n_points_y)
    
xi_2, yi_2 = np.meshgrid(xi_2, yi_2)



xi_3 = np.linspace(points_3[:,0].min(), points_3[:,0].max(), n_points_x)
yi_3 = np.linspace(points_3[:,1].min(), points_3[:,1].max(), n_points_y)
    
xi_3, yi_3 = np.meshgrid(xi_3, yi_3)




#interpolation of the data:
zi_1 = scipy.interpolate.griddata((points_1[:,0], points_1[:,1]), \
                            zi_1, (xi_1, yi_1), method='cubic',  rescale = False)
        
    
zi_2 = scipy.interpolate.griddata((points_2[:,0], points_2[:,1]), \
                            zi_2, (xi_2, yi_2), method='cubic',  rescale = False)

zi_3 = scipy.interpolate.griddata((points_3[:,0], points_3[:,1]), \
                            zi_3, (xi_3, yi_3), method='cubic',  rescale = False)        


zi_33 = zi_2 - zi_1

zi_100 = zi_3 - zi_2
    




# %%% reading two thetas


#d_spacings stored in the following order: d_002_plus_45, d_002_minus_45, d_100_plus_45, d_100_minus_45
#no load 

X_motor_0_percent_BL, Y_motor_0_percent_BL = transform_motors_to_origin(np.array(data_0_percent_BL['Xmotor']),
                                                                        np.array(data_0_percent_BL['Ymotor']))

two_theta_0_percent_BL = [np.array(data_0_percent_BL['2_theta_peak_1_1st_3rd_quarters_mean']),
                           np.array(data_0_percent_BL['2_theta_peak_1_2nd_4th_quarters_mean']),
                           np.array(data_0_percent_BL['2_theta_peak_2_1st_3rd_quarters_mean']),
                           np.array(data_0_percent_BL['2_theta_peak_2_2nd_4th_quarters_mean'])]


#one third buckling load
X_motor_33_percent_BL, Y_motor_33_percent_BL = transform_motors_to_origin(np.array(data_33_percent_BL['Xmotor']), 
                                                              np.array(data_33_percent_BL['Ymotor']))

two_theta_33_percent_BL = [np.array(data_33_percent_BL['2_theta_peak_1_1st_3rd_quarters_mean']),
                            np.array(data_33_percent_BL['2_theta_peak_1_2nd_4th_quarters_mean']),
                            np.array(data_33_percent_BL['2_theta_peak_2_1st_3rd_quarters_mean']),
                            np.array(data_33_percent_BL['2_theta_peak_2_2nd_4th_quarters_mean'])]


#one buckling load
X_motor_100_percent_BL, Y_motor_100_percent_BL = transform_motors_to_origin(np.array(data_100_percent_BL['Xmotor']), 
                                                                            np.array(data_100_percent_BL['Ymotor']))


two_theta_100_percent_BL = [np.array(data_100_percent_BL['2_theta_peak_1_1st_3rd_quarters_mean']),
                             np.array(data_100_percent_BL['2_theta_peak_1_2nd_4th_quarters_mean']),
                             np.array(data_100_percent_BL['2_theta_peak_2_1st_3rd_quarters_mean']),
                             np.array(data_100_percent_BL['2_theta_peak_2_2nd_4th_quarters_mean'])]


#common grid
xi = np.linspace(X_motor_0_percent_BL.min(), X_motor_0_percent_BL.max(), n_points_x)
yi = np.linspace(Y_motor_0_percent_BL.min(), Y_motor_0_percent_BL.max(), n_points_y)

xi, yi = np.meshgrid(xi, yi)





# %%% interpolation of the data:

# no load
two_theta_002_plus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    two_theta_0_percent_BL[0], (xi, yi), method='cubic',  rescale = False)
two_theta_002_minus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    two_theta_0_percent_BL[1], (xi, yi), method='cubic',  rescale = False)

    
two_theta_100_plus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    two_theta_0_percent_BL[2], (xi, yi), method='cubic',  rescale = False)
two_theta_100_minus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    two_theta_0_percent_BL[3], (xi, yi), method='cubic',  rescale = False)


    
# one third buckling load
two_theta_002_plus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    two_theta_33_percent_BL[0], (xi, yi), method='cubic',  rescale = False)
two_theta_002_minus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    two_theta_33_percent_BL[1], (xi, yi), method='cubic',  rescale = False)

    
two_theta_100_plus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    two_theta_33_percent_BL[2], (xi, yi), method='cubic',  rescale = False)
two_theta_100_minus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    two_theta_33_percent_BL[3], (xi, yi), method='cubic',  rescale = False)

    
# one  buckling load
two_theta_002_plus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    two_theta_100_percent_BL[0], (xi, yi), method='cubic',  rescale = False)
two_theta_002_minus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    two_theta_100_percent_BL[1], (xi, yi), method='cubic',  rescale = False)

    
two_theta_100_plus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    two_theta_100_percent_BL[2], (xi, yi), method='cubic',  rescale = False)
two_theta_100_minus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    two_theta_100_percent_BL[3], (xi, yi), method='cubic',  rescale = False)

    
    
# %%% applying sample detector correction factor, computation of d-spacings


correction = 1 

if correction == 0:
    correction_flag = "uncorrected"
    zi_33 = 0
    zi_100 = 0
elif correction == 1: 
    correction_flag = "corrected"
    zi_33 = zi_33
    zi_100 = zi_100





#d_spacing for 0 load - not corrected
d_002_plus_45_0_percent_BL_intpol = distance_correction(two_theta_002_plus_45_33_percent_BL_intpol, 0)
d_002_minus_45_0_percent_BL_intpol = distance_correction(two_theta_002_minus_45_33_percent_BL_intpol, 0)

d_100_plus_45_0_percent_BL_intpol = distance_correction(two_theta_100_plus_45_33_percent_BL_intpol, 0)
d_100_minus_45_0_percent_BL_intpol = distance_correction(two_theta_100_minus_45_33_percent_BL_intpol, 0)


#d_spacing for 33 load - corrected
d_002_plus_45_33_percent_BL_intpol = distance_correction(two_theta_002_plus_45_33_percent_BL_intpol, zi_33)
d_002_minus_45_33_percent_BL_intpol = distance_correction(two_theta_002_minus_45_33_percent_BL_intpol, zi_33)

d_100_plus_45_33_percent_BL_intpol = distance_correction(two_theta_100_plus_45_33_percent_BL_intpol, zi_33)
d_100_minus_45_33_percent_BL_intpol = distance_correction(two_theta_100_minus_45_33_percent_BL_intpol, zi_33)


#d_spacing for 100 load - corrected
d_002_plus_45_100_percent_BL_intpol = distance_correction(two_theta_002_plus_45_100_percent_BL_intpol, zi_100)
d_002_minus_45_100_percent_BL_intpol = distance_correction(two_theta_002_minus_45_100_percent_BL_intpol, zi_100)

d_100_plus_45_100_percent_BL_intpol = distance_correction(two_theta_100_plus_45_100_percent_BL_intpol, zi_100)
d_100_minus_45_100_percent_BL_intpol = distance_correction(two_theta_100_minus_45_100_percent_BL_intpol, zi_100)



# #strain: no to one third buckling load
d_002_plus_45_strain_0_33_load = 1 -  d_002_plus_45_33_percent_BL_intpol/d_002_plus_45_0_percent_BL_intpol
d_002_minus_45_strain_0_33_load = 1 - d_002_minus_45_33_percent_BL_intpol/d_002_minus_45_0_percent_BL_intpol

d_100_plus_45_strain_0_33_load = 1 - d_100_plus_45_33_percent_BL_intpol/d_100_plus_45_0_percent_BL_intpol
d_100_minus_45_strain_0_33_load = 1 - d_100_minus_45_33_percent_BL_intpol/d_100_minus_45_0_percent_BL_intpol

    

# #strain: no to one  buckling load
d_002_plus_45_strain_0_100_load = 1 -  d_002_plus_45_100_percent_BL_intpol/d_002_plus_45_0_percent_BL_intpol
d_002_minus_45_strain_0_100_load = 1 - d_002_minus_45_100_percent_BL_intpol/d_002_minus_45_0_percent_BL_intpol

d_100_plus_45_strain_0_100_load = 1 - d_100_plus_45_100_percent_BL_intpol/d_100_plus_45_0_percent_BL_intpol
d_100_minus_45_strain_0_100_load = 1 - d_100_minus_45_100_percent_BL_intpol/d_100_minus_45_0_percent_BL_intpol



# d_002_plus_45_strain_0_100_load = (d_002_plus_45_100_percent_BL_intpol-d_002_plus_45_33_percent_BL_intpol)/d_002_plus_45_33_percent_BL_intpol


# d_002_plus_45_strain_0_100_load = (d_002_plus_45_100_percent_BL_intpol-d_002_plus_45_0_percent_BL_intpol)/d_002_plus_45_0_percent_BL_intpol



maps = [d_002_plus_45_strain_0_33_load, d_002_minus_45_strain_0_33_load,
        d_100_plus_45_strain_0_33_load, d_100_minus_45_strain_0_33_load,
        d_002_plus_45_strain_0_100_load, d_002_minus_45_strain_0_100_load,
        d_100_plus_45_strain_0_100_load, d_100_minus_45_strain_0_100_load]

maps_descriptors = ['$d_{002}$ strain at +45\N{DEGREE SIGN}, 33 % buckling load', '$d_{002}$ strain at -45\N{DEGREE SIGN}, 33 % buckling load',
                    '$d_{100}$ strain at +45\N{DEGREE SIGN}, 33 % buckling load', '$d_{100}$ strain at -45\N{DEGREE SIGN}, 33 % buckling load',
                    
                    '$d_{002}$ strain at +45\N{DEGREE SIGN}, 105 % buckling load', '$d_{002}$ strain at -45\N{DEGREE SIGN}, 105 % buckling load',
                    '$d_{100}$ strain at +45\N{DEGREE SIGN}, 105 % buckling load', '$d_{100}$ strain at -45\N{DEGREE SIGN}, 105 % buckling load']


maps_files = ['d002_strain_at_+45_33_BL', 'd002_strain_at_-45_33_BL',
              'd100_strain_at_+45_33_BL', 'd100_strain_at_-45_33_BL',
              
              'd002_strain_at_+45_105_BL', 'd002_strain_at_-45_105_BL',
              'd100_strain_at_+45_105_BL', 'd100_strain_at_-45_105_BL']



deformations_folder = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Diamond light source scripts\Strains/'





n_points = 1000

# scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)


for k in range(len(maps)):
    plt.close()
    
    
    fig = plt.figure(1,figsize = (20,10), frameon = 'True')  
    
    title = f' Sample: {selected_sample[3]}, {maps_descriptors[k]}, {correction_flag}'
    
    plt.title(title, fontsize=24 , fontweight="bold", x=0.5, y=1.03)
    
    
    im = plt.imshow(maps[k], vmin=-0.01, vmax=0.01, origin='lower',
                extent=[-1 , 241, -1, 81], cmap='gist_rainbow')
    
    plt.xticks(np.linspace(0,240,num=7), fontsize=16)
    plt.xlim(-0.1,240.1)              
    plt.yticks(np.linspace(0,80,num=9), fontsize=16)
    plt.ylim(-0.1,81)          
    
    plt.xlabel('y position [mm]',fontsize=20, fontweight='bold')
    plt.ylabel('x position [mm]',fontsize=20, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    
    
    # plt.xticks(np.linspace(0,50,num=6), fontsize=8)
    # plt.xlim(0,50)                    
    # plt.yticks(fontsize=8)
    
    cbar = plt.colorbar(im, orientation = 'horizontal')
    # plt.colorbar(orientation = 'horizontal')
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.set_xlabel('Strain',fontsize=20, fontweight='bold')
    
    
    plt.show()
    
    file_name = deformations_folder + selected_sample[3] + '_' + maps_files[k] + '_' + correction_flag + '.png'
    
    # fig.savefig(file_name, format='png', dpi=1200,  facecolor='w', bbox_inches='tight')
    # plt.close()







# fig1 = plt.figure(1,figsize = (20,10), frameon = 'True')  
    
# im = plt.imshow(zi_33, 
#                 extent=[-5 , 245, -5, 85], cmap='gist_rainbow')
    
# plt.clim(-1.5, 1.5)

# # Move title upwards
# title = f'Web deflection, sample {selected_sample[3]} at 33 % buckling load'


# plt.title(title, fontsize=30, fontweight='bold', x=0.5, y=1.03)

# plt.xlabel('x position [mm]',fontsize=24, fontweight='bold')
# plt.ylabel('y position [mm]',fontsize=24, fontweight='bold')
# plt.xticks(fontsize=16, fontweight='bold')
# plt.yticks(fontsize=16, fontweight='bold')
    
    
# plt.xticks(np.linspace(0,240,num=7), fontsize=16)
# plt.xlim(-0.1,240.1)              
# plt.yticks(np.linspace(0,80,num=9), fontsize=16)
# plt.ylim(-0.1,81)            
          
#     # # plt.yticks(fontsize=8)
    
# cbar = plt.colorbar(im, orientation = 'horizontal')
#     # plt.colorbar(orientation = 'horizontal')
# cbar.ax.tick_params(labelsize=20)
# cbar.ax.set_xlabel('Web deflection [mm]',fontsize=24, fontweight='bold')
# plt.show()    

# figure_name_1 = deformations_folder + f'web_deflection_33_BL_sample_{selected_sample[3]}.png'

# # fig1.savefig(figure_name_1, format='png', dpi=1200,  facecolor='w', bbox_inches='tight')
# plt.close()


