# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:29:25 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate
import ast


def theta_to_dspacing(list_theta, n):    
    d_spacings = []        
    for i in range(len(list_theta)):    
        d = 0.065263 * n / (2 * np.sin(np.deg2rad(list_theta[i]/2)))
        d_spacings.append(d)
    return d_spacings






#file names
# half_CC1_1_0_BL_azi_int 
# half_CC1_1_33_BL_azi_int 
# half_CC1_1_105_BL_azi_int 

# #sample CC1 (CC1)
# CC1_0_BL_azi_int
# CC1_33_BL_azi_int 
# CC1_105_BL_azi_int 

# #sample CC2 (CC2)
# CC2_0_BL_azi_int
# CC2_33_BL_azi_int 
# CC2_105_BL_azi_int 

# #sample half_CC1 (1_2_CC1)
# half_CC1_2_0_BL_azi_int 
# half_CC1_2_33_BL_azi_int 
# half_CC1_2_105_BL_azi_int 

# %%% reading the file

name = 'half_CC1_2_105_BL_azi_int'

file_name = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\Azimuthal_int_samples/' + 
            str(name) + ".xlsx")


data = pd.read_excel(file_name)



# %%% prepare new dataframe

feature_lists = ['Total_scan', 'Xmotor', 'Ymotor', 
                    '2_theta_peak_1_1st_3rd_quarters_mean', '2_theta_peak_1_2nd_4th_quarters_mean', 
                    '2_theta_peak_1_1st_3rd_quarters_std', '2_theta_peak_1_2nd_4th_quarters_std',  
                    
                    'd002_peak_1_1st_3rd_quarters_mean', 'd002_peak_1_2nd_4th_quarters_mean', 
                    'd002_peak_1_1st_3rd_quarters_std', 'd002_peak_1_2nd_4th_quarters_std', 
                    
                    
                    
                    '2_theta_peak_2_1st_3rd_quarters_mean', '2_theta_peak_2_2nd_4th_quarters_mean', 
                    '2_theta_peak_2_1st_3rd_quarters_std', '2_theta_peak_2_2nd_4th_quarters_std',  
                    
                    'd100_peak_2_1st_3rd_quarters_mean', 'd100_peak_2_2nd_4th_quarters_mean', 
                    'd100_peak_2_1st_3rd_quarters_std', 'd100_peak_2_2nd_4th_quarters_std', 
                    
                    
                    
                    '2_theta_peak_3_1st_3rd_quarters_mean', '2_theta_peak_3_2nd_4th_quarters_mean', 
                    '2_theta_peak_3_1st_3rd_quarters_std', '2_theta_peak_3_2nd_4th_quarters_std',  
                    
                    'd110_peak_3_1st_3rd_quarters_mean', 'd110_peak_3_2nd_4th_quarters_mean', 
                    'd110_peak_3_1st_3rd_quarters_std', 'd110_peak_3_2nd_4th_quarters_std' ]
    
    
    
    
postprocessed_data = pd.DataFrame(np.zeros((len(data), len(feature_lists))), columns = feature_lists, dtype=(object))
    
    


# %%% prepare lists for appending





for i in range(len(data)):

# for i in range(3,5):

    
    #read cakes list from dataframe

    postprocessed_data.at[i, 'Total_scan'] = i
    
    plt.show()
    
    
    
    postprocessed_data.at[i, 'Xmotor'] = data.at[i, 'Xmotor']
        
    postprocessed_data.at[i, 'Ymotor'] = data.at[i, 'Ymotor']

    
    cakes = data.at[i, 'Cakes']

    #prepare the list for calculating the average and appending 
    
    peak_1_first_third_quarter_cakes = []

    peak_1_second_fourth_quarter_cakes = []
    
    peak_2_first_third_quarter_cakes = []
    
    peak_2_second_fourth_quarter_cakes = []
    
    peak_3_first_third_quarter_cakes = []
    
    peak_3_second_fourth_quarter_cakes = []
    

#lists are as written in excel as strings - they must be reconverted into lists before evaluation,
#the function ast.literal_eval(cakes) does the job - converts list as string to list
#to execute the function we are checking whether 'cakes' are string, 0 is integer and we don't want to touch it
    

    if type(cakes) == str : 
        cakes = ast.literal_eval(cakes)
        
        peak_1_values = ast.literal_eval(data.at[i, 'P1_x0'])
        
        peak_2_values = ast.literal_eval(data.at[i, 'P2_x0'])

        peak_3_values = ast.literal_eval(data.at[i, 'P3_x0'])        

        for j in range(len(cakes)):
            
            #limit to cakes in first and third quarter and eliminate stupid results 
            if cakes[j] in range(0,23) or cakes[j] in range(67,90) \
                and  10.5 <= peak_1_values[j] <= 11.5 \
                and  18.0 <= peak_2_values[j] <= 19.5 \
                and  32.0 <= peak_3_values[j] <= 33.5 :
                    
                    
                    peak_1_first_third_quarter_cakes.append(peak_1_values[j])

                    peak_2_first_third_quarter_cakes.append(peak_2_values[j])
                    
                    peak_3_first_third_quarter_cakes.append(peak_3_values[j])
                    
            
            #limit to cakes in second and fourth quarter and eliminate stupid results 
            elif cakes[j] in range(24,45) or cakes[j] in range(46,66) \
                and  10.5 <= peak_1_values[j] <= 11.5 \
                and  18.0 <= peak_2_values[j] <= 19.5 \
                and  32.0 <= peak_3_values[j] <= 33.5 :                    
                    
                    peak_1_second_fourth_quarter_cakes.append(peak_1_values[j])

                    peak_2_second_fourth_quarter_cakes.append(peak_2_values[j])
                    
                    peak_3_second_fourth_quarter_cakes.append(peak_3_values[j])
                    
        

        #after for loop calculate the mean and standard deviation
                 
        #peak 1
        peak_1_1st_3rd_quarters_mean = (np.array(peak_1_first_third_quarter_cakes)).mean()
        peak_1_2nd_4th_quarters_mean = (np.array(peak_1_second_fourth_quarter_cakes)).mean()
        
        
        postprocessed_data.at[i, '2_theta_peak_1_1st_3rd_quarters_mean'] = peak_1_1st_3rd_quarters_mean
        postprocessed_data.at[i, '2_theta_peak_1_2nd_4th_quarters_mean'] = peak_1_2nd_4th_quarters_mean
        
        
        peak_1_1st_3rd_quarters_std =  (np.array(peak_1_first_third_quarter_cakes)).std()
        peak_1_2nd_4th_quarters_std = (np.array(peak_1_second_fourth_quarter_cakes)).std()
        
        
        postprocessed_data.at[i, '2_theta_peak_1_1st_3rd_quarters_std'] = peak_1_1st_3rd_quarters_std
        postprocessed_data.at[i, '2_theta_peak_1_2nd_4th_quarters_std'] = peak_1_2nd_4th_quarters_std


        #convert list of thetas to d-spacing: (it must be done separately, in order to receive reasonable results)         
        d002_peak_1_1st_3rd_quarters_list = theta_to_dspacing(peak_1_first_third_quarter_cakes, 1)
        d002_peak_1_2nd_4th_quarters_list = theta_to_dspacing(peak_1_second_fourth_quarter_cakes, 1)
        
        
        postprocessed_data.at[i, 'd002_peak_1_1st_3rd_quarters_mean'] = (np.array(d002_peak_1_1st_3rd_quarters_list)).mean()
        postprocessed_data.at[i, 'd002_peak_1_2nd_4th_quarters_mean'] = (np.array(d002_peak_1_2nd_4th_quarters_list)).mean()
        
        postprocessed_data.at[i, 'd002_peak_1_1st_3rd_quarters_std'] = (np.array(d002_peak_1_1st_3rd_quarters_list)).std()
        postprocessed_data.at[i, 'd002_peak_1_2nd_4th_quarters_std'] = (np.array(d002_peak_1_2nd_4th_quarters_list)).std()



        #peak 2
        peak_2_1st_3rd_quarters_mean = (np.array(peak_2_first_third_quarter_cakes)).mean()
        peak_2_2nd_4th_quarters_mean = (np.array(peak_2_second_fourth_quarter_cakes)).mean()
        

        postprocessed_data.at[i, '2_theta_peak_2_1st_3rd_quarters_mean'] = peak_2_1st_3rd_quarters_mean
        postprocessed_data.at[i, '2_theta_peak_2_2nd_4th_quarters_mean'] = peak_2_2nd_4th_quarters_mean        

        
        peak_2_1st_3rd_quarters_std =  (np.array(peak_2_first_third_quarter_cakes)).std()
        peak_2_2nd_4th_quarters_std = (np.array(peak_2_second_fourth_quarter_cakes)).std()
        
        
        postprocessed_data.at[i, '2_theta_peak_2_1st_3rd_quarters_std'] = peak_2_1st_3rd_quarters_std
        postprocessed_data.at[i, '2_theta_peak_2_2nd_4th_quarters_std'] = peak_2_2nd_4th_quarters_std
        
        
        #convert list of thetas to d-spacing: (it must be done separately, in order to receive reasonable results)         
        d100_peak_2_1st_3rd_quarters_list = theta_to_dspacing(peak_2_first_third_quarter_cakes, 1)
        d100_peak_2_2nd_4th_quarters_list = theta_to_dspacing(peak_2_second_fourth_quarter_cakes, 1)
        
        
        postprocessed_data.at[i, 'd100_peak_2_1st_3rd_quarters_mean'] = (np.array(d100_peak_2_1st_3rd_quarters_list)).mean()
        postprocessed_data.at[i, 'd100_peak_2_2nd_4th_quarters_mean'] = (np.array(d100_peak_2_2nd_4th_quarters_list)).mean()
        
        postprocessed_data.at[i, 'd100_peak_2_1st_3rd_quarters_std'] = (np.array(d100_peak_2_1st_3rd_quarters_list)).std()
        postprocessed_data.at[i, 'd100_peak_2_2nd_4th_quarters_std'] = (np.array(d100_peak_2_2nd_4th_quarters_list)).std()


        #peak 3
        peak_3_1st_3rd_quarters_mean = (np.array(peak_3_first_third_quarter_cakes)).mean()
        peak_3_2nd_4th_quarters_mean = (np.array(peak_3_second_fourth_quarter_cakes)).mean()
        

        postprocessed_data.at[i, '2_theta_peak_3_1st_3rd_quarters_mean'] = peak_3_1st_3rd_quarters_mean
        postprocessed_data.at[i, '2_theta_peak_3_2nd_4th_quarters_mean'] = peak_3_2nd_4th_quarters_mean        

        
        peak_3_1st_3rd_quarters_std =  (np.array(peak_2_first_third_quarter_cakes)).std()
        peak_3_2nd_4th_quarters_std = (np.array(peak_2_second_fourth_quarter_cakes)).std()
        
        
        postprocessed_data.at[i, '2_theta_peak_3_1st_3rd_quarters_std'] = peak_2_1st_3rd_quarters_std
        postprocessed_data.at[i, '2_theta_peak_3_2nd_4th_quarters_std'] = peak_2_2nd_4th_quarters_std
        
        
        #convert list of thetas to d-spacing: (it must be done separately, in order to receive reasonable results)         
        d110_peak_3_1st_3rd_quarters_list = theta_to_dspacing(peak_3_first_third_quarter_cakes, 1)
        d110_peak_3_2nd_4th_quarters_list = theta_to_dspacing(peak_3_second_fourth_quarter_cakes, 1)
        
        
        postprocessed_data.at[i, 'd110_peak_3_1st_3rd_quarters_mean'] = (np.array(d110_peak_3_1st_3rd_quarters_list)).mean()
        postprocessed_data.at[i, 'd110_peak_3_2nd_4th_quarters_mean'] = (np.array(d110_peak_3_2nd_4th_quarters_list)).mean()
        
        postprocessed_data.at[i, 'd110_peak_3_1st_3rd_quarters_std'] = (np.array(d110_peak_3_1st_3rd_quarters_list)).std()
        postprocessed_data.at[i, 'd110_peak_3_2nd_4th_quarters_std'] = (np.array(d110_peak_3_2nd_4th_quarters_list)).std()



        
    else:
        # print('None')
        None 
        
    
    
    

postprocessed_data.fillna(0)

postprocessed_data.fillna(value=0, inplace=True) # This fills all the null values in the columns with 0.


output_file = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\Azimuthal_int_d_spacing/' +
              str(name) + "_dspacing.xlsx")

postprocessed_data.to_excel(output_file)

