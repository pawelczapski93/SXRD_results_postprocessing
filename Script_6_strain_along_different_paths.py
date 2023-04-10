# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:26:41 2022

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
from scipy import interpolate
import ast


def interpolation(x1, y1, x2, y2, x):
    f1 = interpolate.interp1d(x1, y1)
    f2 = interpolate.interp1d(x1, y1)
    
    y1_interpolated = f1(x)
    y2_interpolated = f2(x)
    
    return y1_interpolated, y2_interpolated


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


samples_old_names = ['half_CC1_1_no_load', 'half_CC1_1_one_third_buckling_load', 'half_CC1_1_one_buckling_load',
                     'half_CC1_2_no_load', 'half_CC1_2_one_third_buckling_load', 'half_CC1_2_one_buckling_load',
                     'CC1_no_load', 'CC1_one_third_buckling_load', 'CC1_one_buckling_load',
                     'CC2_no_load', 'CC2_one_third_buckling_load', 'CC2_one_buckling_load']


samples_new_names = ['HS1_0_BL', 'HS1_33_BL', 'HS1_105_BL',
                     'HS2_0_BL', 'HS2_33_BL', 'HS2_105_BL',
                     'SL1_0_BL', 'SL1_33_BL', 'SL1_105_BL',
                     'SL2_0_BL', 'SL2_33_BL', 'SL2_105_BL']


#dictionary for all results dataframe
d = {}
#dictionary for particular results dataframe

web = {}
length = {}

for i in range(len(samples_new_names)):

    #create dataframe with a new name of the sample 
    d[samples_new_names[i]] = pd.DataFrame() #create dataframe with the name of the sample
    
    web[samples_new_names[i]] = pd.DataFrame() #create dataframe with the name of the sample
    length[samples_new_names[i]] = pd.DataFrame() #create dataframe with the name of the sample

    
    #open file using old name
    
    file_name = (r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\May_res/' + 
                  str(samples_old_names[i]) + "_results.xlsx")
    
    opened_data = pd.read_excel(file_name)

    d[samples_new_names[i]]['X_pos'] =  opened_data['Xmotor']
    d[samples_new_names[i]]['Y_pos'] =  opened_data['Ymotor']
    
    d[samples_new_names[i]]['d002_plus_45'] =  opened_data['d002_peak_1_1st_3rd_quarters_mean']
    d[samples_new_names[i]]['d002_minus_45'] =  opened_data['d002_peak_1_2nd_4th_quarters_mean']
    
    d[samples_new_names[i]]['d100_plus_45'] =  opened_data['d100_peak_2_1st_3rd_quarters_mean']
    d[samples_new_names[i]]['d100_minus_45'] =  opened_data['d100_peak_2_2nd_4th_quarters_mean']

    
    #sum along the web and flange
    #web positions     
    x = d[samples_new_names[i]]['Y_pos'].drop_duplicates(keep="first").dropna(how='all').values.tolist() # take all scanning points at the flange and remove duplicates    
    d002_plus_45_web_list, d002_minus_45_web_list, d100_plus_45_web_list, d100_minus_45_web_list  = [],[],[],[]


    for j in range(len(x)):
        
        d002_plus_45_web_averaged =  d[samples_new_names[i]]['d002_plus_45'][d[samples_new_names[i]]['Y_pos'] == x[j]] \
            [0.38 > d[samples_new_names[i]]['d002_plus_45']][d[samples_new_names[i]]['d002_plus_45'] > 0.32].mean() 
        d002_plus_45_web_list.append(d002_plus_45_web_averaged)


        d002_minus_45_web_averaged =  d[samples_new_names[i]]['d002_minus_45'][d[samples_new_names[i]]['Y_pos'] == x[j]]\
            [0.38 > d[samples_new_names[i]]['d002_minus_45']][d[samples_new_names[i]]['d002_minus_45'] > 0.32].mean() 
        d002_minus_45_web_list.append(d002_minus_45_web_averaged)


        d100_plus_45_web_averaged =  d[samples_new_names[i]]['d100_plus_45'][d[samples_new_names[i]]['Y_pos'] == x[j]]\
            [0.24 > d[samples_new_names[i]]['d100_plus_45']][d[samples_new_names[i]]['d100_plus_45'] > 0.14].mean() 
        d100_plus_45_web_list.append(d100_plus_45_web_averaged)


        d100_minus_45_web_averaged =  d[samples_new_names[i]]['d100_minus_45'][d[samples_new_names[i]]['Y_pos'] == x[j]]\
            [0.24 > d[samples_new_names[i]]['d100_minus_45']][d[samples_new_names[i]]['d100_minus_45'] > 0.14].mean() 
        d100_minus_45_web_list.append(d100_minus_45_web_averaged)


    web[samples_new_names[i]]['Y_pos_web'] = x 
    web[samples_new_names[i]]['d002_plus_45_web'] = d002_plus_45_web_list
    web[samples_new_names[i]]['d002_minus_45_web'] = d002_minus_45_web_list
    web[samples_new_names[i]]['d100_plus_45_web'] = d100_plus_45_web_list
    web[samples_new_names[i]]['d100_minus_45_web'] = d100_minus_45_web_list
    

    #flanges positions
    y = d[samples_new_names[i]]['X_pos'].drop_duplicates(keep="first").dropna(how='all').values.tolist() # take all scanning points at the flange and remove duplicates    
    d002_plus_45_length_list, d002_minus_45_length_list, d100_plus_45_length_list, d100_minus_45_length_list  = [],[],[],[]


    for k in range(len(y)):

        d002_plus_45_length_averaged =  d[samples_new_names[i]]['d002_plus_45'][d[samples_new_names[i]]['X_pos'] == y[k]]\
            [0.38 > d[samples_new_names[i]]['d002_plus_45']][d[samples_new_names[i]]['d002_plus_45'] > 0.32].mean() 
        d002_plus_45_length_list.append(d002_plus_45_length_averaged)


        d002_minus_45_length_averaged =  d[samples_new_names[i]]['d002_minus_45'][d[samples_new_names[i]]['X_pos'] == y[k]]\
            [0.38 > d[samples_new_names[i]]['d002_minus_45']][d[samples_new_names[i]]['d002_minus_45'] > 0.32].mean()             
        d002_minus_45_length_list.append(d002_minus_45_length_averaged)


        d100_plus_45_length_averaged =  d[samples_new_names[i]]['d100_plus_45'][d[samples_new_names[i]]['X_pos'] == y[k]]\
            [0.24 > d[samples_new_names[i]]['d100_plus_45']][d[samples_new_names[i]]['d100_plus_45'] > 0.14].mean() 
        d100_plus_45_length_list.append(d100_plus_45_length_averaged)

        d100_minus_45_length_averaged =  d[samples_new_names[i]]['d100_minus_45'][d[samples_new_names[i]]['X_pos'] == y[k]]\
            [0.24 > d[samples_new_names[i]]['d100_minus_45']][d[samples_new_names[i]]['d100_minus_45'] > 0.14].mean() 
        d100_minus_45_length_list.append(d100_minus_45_length_averaged)



    length[samples_new_names[i]]['X_pos_length'] = y 
    length[samples_new_names[i]]['d002_plus_45_web'] = d002_plus_45_length_list
    length[samples_new_names[i]]['d002_minus_45_web'] = d002_minus_45_length_list
    length[samples_new_names[i]]['d100_plus_45_web'] = d100_plus_45_length_list
    length[samples_new_names[i]]['d100_minus_45_web'] = d100_minus_45_length_list


    

# %%% Plotting d-spacing along the web
fig1, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2, sharex=False, sharey=False, figsize = (15,10))
fig1.tight_layout()
plt.subplots_adjust(
                    wspace=0.3,
                    hspace=0.33)


linestyles = ['solid', 'solid', 'solid','solid']

colors = ['red', 'blue', 'green','orange']


samples = ['SL1_0_BL', 'SL2_0_BL','HS1_0_BL', 'HS2_0_BL' ]

titles = ['d200 +45', 'd200 -45','d100 +45', 'd100 -45']

axis = [ax1, ax2, ax3, ax4]

for ax_i in range(len(axis)):
    
    axis[ax_i].set_xlim(-1,81)
    axis[ax_i].set_xlabel('Vertical position [mm]',size=28, fontweight='bold')
    # axis.set_xticks(labelsize=20, fontweight='bold')
    # axis.set_yticks(labelsize=20, fontweight='bold')
    
    axis[ax_i].xaxis.set_tick_params(labelsize = 24)
    axis[ax_i].yaxis.set_tick_params(labelsize = 24)
    axis[ax_i].set_title(titles[ax_i])
    axis[ax_i].set_title(titles[ax_i], fontdict={'fontsize': 30, 'fontweight': 'bold'})



for u in range(len(samples)):
        
    ax1.plot(web[samples[u]]['Y_pos_web'], web[samples[u]]['d002_plus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax2.plot(web[samples[u]]['Y_pos_web'], web[samples[u]]['d002_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    ax1.set_ylim(0.340, 0.355)
    ax2.set_ylim(0.340, 0.355)


    ax3.plot(web[samples[u]]['Y_pos_web'], web[samples[u]]['d100_plus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax4.plot(web[samples[u]]['Y_pos_web'], web[samples[u]]['d100_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )     


    ax3.set_ylim(0.19, 0.2)
    ax4.set_ylim(0.19, 0.2)

    
#     # ax1.set_ylim(39,51)


    ax1.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    ax2.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    
    ax3.set_ylabel('d100 [nm]',size=28, fontweight='bold')
    ax4.set_ylabel('d100 [nm]',size=28, fontweight='bold')
   


    ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.40), fontsize = 30, 
          ncol=4, fancybox=True, shadow=True)
    
plt.show()
# fig.savefig('d_spacings_after_manufacturing_web.png', format='png', dpi=1200,  facecolor='w', bbox_inches='tight')

fig1.clear()


# %%% Plotting d-spacing along the length
fig1, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2, sharex=False, sharey=False, figsize = (17,9))
fig1.tight_layout()
plt.subplots_adjust(wspace=0.3,
                    hspace=0.37)


linestyles = ['solid', 'solid', 'solid','solid']

colors = ['red', 'blue', 'green','orange']


samples = ['SL1_0_BL', 'SL2_0_BL','HS1_0_BL', 'HS2_0_BL' ]

titles = ['d200 +45', 'd200 -45','d100 +45', 'd100 -45']

axis = [ax1, ax2, ax3, ax4]

for ax_i in range(len(axis)):
    
    # axis[ax_i].set_xlim(-1,241)
    axis[ax_i].set_xlabel('Horizontal position [mm]',size=28, fontweight='bold')
    # axis.set_xticks(labelsize=20, fontweight='bold')
    # axis.set_yticks(labelsize=20, fontweight='bold')
    
    axis[ax_i].xaxis.set_tick_params(labelsize = 24)
    axis[ax_i].yaxis.set_tick_params(labelsize = 24)
    axis[ax_i].set_title(titles[ax_i])
    axis[ax_i].set_title(titles[ax_i], fontdict={'fontsize': 30, 'fontweight': 'bold'})

    axis[ax_i].set_xlim(-1, 241)
    axis[ax_i].set_xlim(-1, 241)

    axis[ax_i].set_xticks([0,40,80,120,160,200,240])

for u in range(len(samples)):
        
    x = -1 * np.array(length[samples[u]]['X_pos_length'])
    
    ax1.plot(x, length[samples[u]]['d002_plus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax2.plot(x, length[samples[u]]['d002_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    ax1.set_ylim(0.340, 0.355)
    ax2.set_ylim(0.340, 0.355)




    ax3.plot(x, length[samples[u]]['d100_plus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax4.plot(x, length[samples[u]]['d100_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )     


    ax3.set_ylim(0.19, 0.2)
    ax4.set_ylim(0.19, 0.2)

    
#     # ax1.set_ylim(39,51)


    ax1.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    ax2.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    
    ax3.set_ylabel('d100 [nm]',size=28, fontweight='bold')
    ax4.set_ylabel('d100 [nm]',size=28, fontweight='bold')
   


    ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.40), fontsize = 30, 
          ncol=4, fancybox=True, shadow=True)
    
plt.show()
# fig1.savefig('d_spacings_after_manufacturing_length.png', format='png', dpi=1200,  facecolor='w', bbox_inches='tight')




# %%% Plotting strains

fig2, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2, 2, sharex=False, sharey=False, figsize = (17,9))
fig2.tight_layout()
plt.subplots_adjust(wspace=0.3,
                    hspace=0.37)


linestyles = ['solid', 'solid', 'solid','solid']

colors = ['red', 'blue', 'green','orange']


samples_0_BL = ['SL1_0_BL', 'SL2_0_BL','HS1_0_BL', 'HS2_0_BL' ]
samples_33_BL = ['SL1_33_BL', 'SL2_33_BL','HS1_33_BL', 'HS2_33_BL' ]
samples_105_BL = ['SL1_105_BL', 'SL2_105_BL','HS1_105_BL', 'HS2_105_BL' ]




titles = ['d200 +45', 'd200 -45','d100 +45', 'd100 -45']

axis = [ax1, ax2, ax3, ax4]

for ax_i in range(len(axis)):
    
    # axis[ax_i].set_xlim(-1,241)
    axis[ax_i].set_xlabel('Horizontal position [mm]',size=28, fontweight='bold')
    # axis.set_xticks(labelsize=20, fontweight='bold')
    # axis.set_yticks(labelsize=20, fontweight='bold')
    
    axis[ax_i].xaxis.set_tick_params(labelsize = 24)
    axis[ax_i].yaxis.set_tick_params(labelsize = 24)
    axis[ax_i].set_title(titles[ax_i])
    axis[ax_i].set_title(titles[ax_i], fontdict={'fontsize': 30, 'fontweight': 'bold'})

    axis[ax_i].set_xlim(-1, 241)
    axis[ax_i].set_xlim(-1, 241)

    axis[ax_i].set_xticks([0,40,80,120,160,200,240])



def interpolation(x1, y1, x2, y2, x):
    
    f1 = interpolate.interp1d(x1, y1)
    f2 = interpolate.interp1d(x2, y2)
    
    y1_interpolated = f1(x)
    y2_interpolated = f2(x)
    
    return y1_interpolated, y2_interpolated



for u in range(len(samples_0_BL)):
    
    
#
    

    x1 = -1 * np.array(length[samples_0_BL[u]]['X_pos_length'])
    x2 = -1 * np.array(length[samples_33_BL[u]]['X_pos_length'])


    y1 = length[samples_0_BL[u]]['d002_plus_45_web'] 
    y2 = length[samples_33_BL[u]]['d002_plus_45_web'] 

    x = x1
    
    # y1, y2 = interpolation(x1, y1, x2, y2, x)




    d002_strain_33_BL = (length[samples_33_BL[u]]['d002_plus_45_web'] - length[samples_0_BL[u]]['d002_plus_45_web'])/ \
        length[samples_0_BL[u]]['d002_plus_45_web']


    ax1.plot(x, d002_strain_33_BL, 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax2.plot(x, length[samples[u]]['d002_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    ax1.set_ylim(0.340, 0.355)
    ax2.set_ylim(0.340, 0.355)




    ax3.plot(x, length[samples[u]]['d100_plus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )
    
    
    ax4.plot(x, length[samples[u]]['d100_minus_45_web'], 
             label = samples[u], linestyle = linestyles[u], color = colors[u] )     


    ax3.set_ylim(0.19, 0.2)
    ax4.set_ylim(0.19, 0.2)

    
#     # ax1.set_ylim(39,51)


    ax1.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    ax2.set_ylabel('d200 [nm]',size=28, fontweight='bold')
    
    ax3.set_ylabel('d100 [nm]',size=28, fontweight='bold')
    ax4.set_ylabel('d100 [nm]',size=28, fontweight='bold')
   


    ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.40), fontsize = 30, 
          ncol=4, fancybox=True, shadow=True)
    
plt.show()
# fig2.savefig('d_spacings_after_manufacturing_length.png', format='png', dpi=1200,  facecolor='w', bbox_inches='tight')

