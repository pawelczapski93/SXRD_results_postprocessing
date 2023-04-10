# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:59:16 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import inspect


def data_merging(to_merge, merged):
    for i in range(len(to_merge)):
        for j in range(len(to_merge[i])):
            merged.append(to_merge[i][j])


# def retrieve_name(var):
#         """
#         Gets the name of var. Does it from the out most frame inner-wards.
#         :param var: variable to get name from.
#         :return: string
#         """
#         for fi in reversed(inspect.stack()):
#             names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
#             if len(names) > 0:
#                 return names[0]


Case_dataset = []
Xmotor_dataset = []
Ymotor_dataset = []

P1_amp_dataset = []
P1_x0_dataset = []
P1_SD_dataset = []
P1_diff_dataset = []
P1_amp_err_dataset = []
P1_x0_err_dataset = []
P1_SD_err_dataset = []

P2_amp_dataset = []
P2_x0_dataset = []
P2_SD_dataset = []
P2_diff_dataset = []
P2_amp_err_dataset = []
P2_x0_err_dataset = []
P2_SD_err_dataset = []

P3_amp_dataset = []
P3_x0_dataset = []
P3_SD_dataset = []
P3_diff_dataset = []
P3_amp_err_dataset = []
P3_x0_err_dataset = []
P3_SD_err_dataset = []



#sample half_CC1 (1_2_CC1)
half_CC1_1_0_BL = [371901, 371902, 371903]
half_CC1_1_33_BL = [371908, 371909, 371910]
half_CC1_1_105_BL = [371912, 371913, 371914]

#sample CC1 (CC1)
CC1_0_BL = [371919, 371920, 371921]
CC1_33_BL = [371923, 371924, 371925]
CC1_105_BL = [371927, 371928, 371929]

#sample CC2 (CC2)
CC2_0_BL = [371933, 371934, 371935]
CC2_33_BL = [371937, 371938, 371939]
CC2_105_BL = [371941, 371942, 371943]

#sample half_CC1 (1_2_CC1)
half_CC1_2_0_BL = [371948, 371949, 371950]
half_CC1_2_33_BL = [371952, 371953, 371954]
half_CC1_2_105_BL = [371956, 371957, 371958]


sample_case = half_CC1_2_105_BL

name = "CC1_105_BL"

#1st_dataset


    
path_first_dataset = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\Azimuthal_int_peaks/'


path_name_1 = path_first_dataset + 'ipp_' + str(sample_case[0]) + '_azimuthal_integration' + '.xlsx'
df1 = pd.read_excel(path_name_1)


path_name_2 = path_first_dataset + 'ipp_' + str(sample_case[1]) + '_azimuthal_integration' + '.xlsx'
df2 = pd.read_excel(path_name_2)


path_name_3 = path_first_dataset + 'ipp_' + str(sample_case[2]) + '_azimuthal_integration' + '.xlsx'
df3 = pd.read_excel(path_name_3)





frames = [df1, df2, df3]

result = pd.concat(frames)






file_name = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\Azimuthal_int_samples/' + 
            str(name) +"_azi_int.xlsx")

result.to_excel(file_name)


