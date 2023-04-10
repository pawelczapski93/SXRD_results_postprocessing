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

SL1 = ['CC1_no_load', 'CC1_one_third_buckling_load', 'CC1_one_buckling_load']
SL2 = ['CC2_no_load', 'CC2_one_third_buckling_load', 'CC2_one_buckling_load']

HS1 = ['half_CC1_1_no_load', 'half_CC1_1_one_third_buckling_load', 'half_CC1_1_one_buckling_load']
HS2 = ['half_CC1_2_no_load', 'half_CC1_2_one_third_buckling_load', 'half_CC1_2_one_buckling_load']

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


# %%% strain calculation: compare no load to one third load (four strains are to be compared:
# d_002_45, 
# d_002_-45, 
# d100_45, 
# d100_-45

#the function that moves the chart to position 0,0 
def transform_motors_to_origin(X_motor, Y_motor):
    X_motor_transformed = X_motor - min(X_motor)
    Y_motor_transformed = Y_motor - min(Y_motor)
    return X_motor_transformed, Y_motor_transformed



#residual strains






#d_spacings stored in the following order: d_002_plus_45, d_002_minus_45, d_100_plus_45, d_100_minus_45
#no load 

X_motor_0_percent_BL, Y_motor_0_percent_BL = transform_motors_to_origin(np.array(data_0_percent_BL['Xmotor']),
                                                                        np.array(data_0_percent_BL['Ymotor']))

d_spacings_0_percent_BL = [np.array(data_0_percent_BL['d002_peak_1_1st_3rd_quarters_mean']),
                           np.array(data_0_percent_BL['d002_peak_1_2nd_4th_quarters_mean']),
                           np.array(data_0_percent_BL['d100_peak_2_1st_3rd_quarters_mean']),
                           np.array(data_0_percent_BL['d100_peak_2_2nd_4th_quarters_mean'])]


#one third buckling load
X_motor_33_percent_BL, Y_motor_33_percent_BL = transform_motors_to_origin(np.array(data_33_percent_BL['Xmotor']), 
                                                              np.array(data_33_percent_BL['Ymotor']))

d_spacings_33_percent_BL = [np.array(data_33_percent_BL['d002_peak_1_1st_3rd_quarters_mean']),
                            np.array(data_33_percent_BL['d002_peak_1_2nd_4th_quarters_mean']),
                            np.array(data_33_percent_BL['d100_peak_2_1st_3rd_quarters_mean']),
                            np.array(data_33_percent_BL['d100_peak_2_2nd_4th_quarters_mean'])]


#one buckling load
X_motor_100_percent_BL, Y_motor_100_percent_BL = transform_motors_to_origin(np.array(data_100_percent_BL['Xmotor']), 
                                                                            np.array(data_100_percent_BL['Ymotor']))


d_spacings_100_percent_BL = [np.array(data_100_percent_BL['d002_peak_1_1st_3rd_quarters_mean']),
                             np.array(data_100_percent_BL['d002_peak_1_2nd_4th_quarters_mean']),
                             np.array(data_100_percent_BL['d100_peak_2_1st_3rd_quarters_mean']),
                             np.array(data_100_percent_BL['d100_peak_2_2nd_4th_quarters_mean'])]


#common grid
xi = np.linspace(X_motor_0_percent_BL.min(), X_motor_0_percent_BL.max(), 200)
yi = np.linspace(Y_motor_0_percent_BL.min(), Y_motor_0_percent_BL.max(), 200)

xi, yi = np.meshgrid(xi, yi)


#interpolation of the data:

# no load
d_002_plus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    d_spacings_0_percent_BL[0], (xi, yi), method='cubic',  rescale = False)
d_002_minus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    d_spacings_0_percent_BL[1], (xi, yi), method='cubic',  rescale = False)

    
d_100_plus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    d_spacings_0_percent_BL[2], (xi, yi), method='cubic',  rescale = False)
d_100_minus_45_0_percent_BL_intpol = scipy.interpolate.griddata((X_motor_0_percent_BL, Y_motor_0_percent_BL), \
                                    d_spacings_0_percent_BL[3], (xi, yi), method='cubic',  rescale = False)


    
# one third buckling load
d_002_plus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    d_spacings_33_percent_BL[0], (xi, yi), method='cubic',  rescale = False)
d_002_minus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    d_spacings_33_percent_BL[1], (xi, yi), method='cubic',  rescale = False)

    
d_100_plus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    d_spacings_33_percent_BL[2], (xi, yi), method='linear',  rescale = False)
d_100_minus_45_33_percent_BL_intpol = scipy.interpolate.griddata((X_motor_33_percent_BL, Y_motor_33_percent_BL), \
                                    d_spacings_33_percent_BL[3], (xi, yi), method='linear',  rescale = False)


    
# one  buckling load
d_002_plus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    d_spacings_100_percent_BL[0], (xi, yi), method='linear',  rescale = False)
d_002_minus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    d_spacings_100_percent_BL[1], (xi, yi), method='linear',  rescale = False)

    
d_100_plus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    d_spacings_100_percent_BL[2], (xi, yi), method='linear',  rescale = False)
d_100_minus_45_100_percent_BL_intpol = scipy.interpolate.griddata((X_motor_100_percent_BL, Y_motor_100_percent_BL), \
                                    d_spacings_100_percent_BL[3], (xi, yi), method='linear',  rescale = False)




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





n_points = 1000

# scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)



fig = plt.figure(1,figsize = (20,10), frameon = 'True')  



title = ' sample: ' + 'HS1' +  ' -45 d100 strain: ' + r'(105 load) to (0 load)'


# title = ' sample: ' + str(name_0_percent_BL[0:8]) +  ', 2nd & 4th quarter, d100 ratio: ' + r'$\dfrac{buckling load}{no load}$'



plt.title(title, fontsize=24 , fontweight="bold" )


im = plt.imshow(d_100_minus_45_strain_0_100_load, vmin=-0.02, vmax=0.02, origin='lower',
            extent=[-1 , 241, -1, 81], cmap='gist_rainbow')

# im = plt.imshow(d_100_plus_45_100_percent_BL_intpol, 
#             extent=[-5 , 245, -5, 85], cmap='gist_rainbow')

# plt.plot(X_motor_no_load, Y_motor_no_load,  'yx')
# plt.clim(1.00, 1.04)

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
cbar.ax.set_xlabel('d200 [nm]',fontsize=20, fontweight='bold')


plt.show()


# fig.savefig('HS1_d100_-45_105_BL.png', format='png', dpi=1200,  facecolor='w', bbox_inches='tight')
# plt.close()

