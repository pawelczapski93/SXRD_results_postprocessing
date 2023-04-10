# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:35:52 2022

@author: pawel
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py 
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# %%
# 1st order Gaussian fit

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))

# 2nd order Gaussian fit

def _2gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
        return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
                amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))
    
def densify (np_array, m):
    new_array = np.zeros(len(np_array)+m*(len(np_array)-1))
    counter = 0    
    for k in np.ndindex(len(np_array)-1):
        new_array[counter] = np_array[k[0]] 
        counter = counter + 1 
    
    for l in range(m): 
        # print(l)
        new_array[counter] = np_array[k[0]] + (np_array[[k[0]+1]] - np_array[k[0]] )*(l+1)/(m+1) 
        counter = counter + 1 
        new_array[-1] = np_array[-1]
    return (new_array)

def pixel_to_2theta(Pixeldist):
    pixelSize = 31E-3 #mm (31micron)
    SDdist = 76.06 #mm (Sample-to-detector distance)
    twoTheta = np.tan(Pixeldist*pixelSize/SDdist) #(Conversion equation)
    twoTheta = twoTheta * 180 / np.pi      #(Convert radian to degree)
    return twoTheta


# function peak search returns peak position based on the smoothened plot, please provide:
# x_array, smoothened function, search interval and beginning and marker to mark on the plot 'bx'
def peak_search(x_array, y_array_smooth, search_start, search_end, marker):
    peak_pos, _ = find_peaks(y_array_smooth[search_start : search_end])
    
    if len(peak_pos) > 0:
        peak_pos = search_start + min(peak_pos)
        # plt.plot(x_array[peak_pos], y_array_smooth[peak_pos], marker, markersize = 10, markeredgewidth=3)
        return peak_pos
    else:
        return search_start



#evaluates whether the chart is okay for carbon diffraction pattern 
def chart_evaluation(x_array, y_array):
    y_array_total_smooth = gaussian_filter1d(y_array, 3)
    y_array_total_smooth_1_derivative = (np.gradient(y_array_total_smooth))
    y_array_total_smooth_2_derivative = np.gradient(np.gradient(y_array_total_smooth))
    
    max_derivative_1 = round(max(np.abs((y_array_total_smooth_1_derivative))),3)
    max_derivative_2 = round(max(np.abs((y_array_total_smooth_2_derivative))),3)
    
    
    max_index_1st_peak = y_array_total_smooth.argmax()
    
    max_2nd_peak = max(y_array_total_smooth[1000:1200])
    
    max_after_2nd_peak = max(y_array_total_smooth[1800 : 2000])
    
    first_peak_pos = peak_search(x_array, y_array_total_smooth, 550, 650, 'rx')
            
    second_peak_pos = peak_search(x_array, y_array_total_smooth, 1000, 1200, 'bx')
    
    third_peak_pos = peak_search(x_array, y_array_total_smooth, 1900, 2100, 'gx')

    third_peak_values = y_array_total_smooth[1900 : 2100]
    
    
    if max (y_array_total_smooth) > 5000 and max_index_1st_peak < 650 and 550 < max_index_1st_peak  and \
        max_derivative_2 < 50 and max_derivative_1 < 200 and max_2nd_peak < 3000 and max_after_2nd_peak > 1500 \
        and min(third_peak_values) > 1500:    
            
        return True 
    else:
        return False 






def trim_peak(x_array_total, y_array_total, peak_position_guess, interval_left, interval_right):

    trim_peak = [peak_position_guess - interval_left , peak_position_guess + interval_right]    
    
    x_array = np.array(x_array_total [trim_peak[0] : trim_peak[1]])
    y_array = np.array(y_array_total [trim_peak[0] : trim_peak[1]])

    return x_array, y_array




def peak_fitting(x_array, y_array, imageNo, cake, peak_string):
    
    plt.close()
    bounds = (0, 0, 0) , (np.inf, np.inf, np.inf) #Tuple

    # figure = plt.figure()
    
    # plt.plot(x_array, y_array, 'violet') 


    y_array_smooth = gaussian_filter1d(y_array, 5)
    
    # plt.plot(x_array, y_array_smooth)
    # plt.show()
    

    peak_pos = find_peaks(y_array_smooth)
    peak_pos = peak_pos[0][0]
    # plt.show()
    
    # plt.plot(x_array[peak_pos] , y_array_smooth[peak_pos], 'yx', markersize = 10, markeredgewidth=3)
    
    
    peak_centre = x_array[peak_pos]
    peak_amplitude = max(y_array)
    peak_standard_deviation = float(x_array.std())
    
    
    popt_gauss = np.array([0.000, 0.000, 0.000]) 
    
    pcov_gauss = np.array([[0.000, 0.000, 0.000],
                          [0.000, 0.000, 0.000],
                          [0.000, 0.000, 0.000]]) 
            
    try:
        initial_vals = [peak_amplitude, peak_centre, peak_standard_deviation]
        popt_gauss, pcov_gauss = curve_fit(_1gaussian, x_array, y_array, p0 = initial_vals, bounds=bounds)
            
        perr_gauss = np.sqrt(np.diag(pcov_gauss))
        y_fit_peak = _1gaussian(x_array, *popt_gauss)
        # plt.close()
        
        
        print('Image ' + str(imageNo) + ', cake ' + str(cake) + ', ' + peak_string + ' found')
        
        
    except:
        print('Image ' + str(imageNo) + ', cake ' + str(cake) + ', Peak not found')

        y_fit_peak = 0.75 *peak_amplitude * np.ones((len(x_array)))
        
        pass    



    # plt.plot(x_array, y_array, label= "original curve", linestyle='dashed', linewidth = 0.8, 
    #          marker='o', markerfacecolor='black', markersize=3, color = 'black')
    
    
    # plt.plot(x_array, y_fit_peak, label= "fitted curve",  linewidth = 1, color = 'black')
        

    # plt.plot(popt_gauss[1], np.interp(popt_gauss[1], x_array, y_fit_peak),'rx', label= "fitted peak", 
    #          markersize = 10, markeredgewidth=4, color = 'black') 

    # plt.xlabel('Angle 2\u03B8  [\N{DEGREE SIGN}]',fontsize=12, fontweight='bold')
    # plt.ylabel('Intensity [a.u.]',fontsize=12, fontweight='bold')
    
    
    
    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)
    
    # imageNo = 20
    
    # peak = '1'
    
    # cake = '30'
    
    # plt.title('Azimuthal integration, Image ' + str(imageNo) + ', Cake ' + str(cake) + ', Peak ' + str(peak) , 
    #               fontsize=14, fontweight='bold')
    # plt.legend()
    # plt.show()    

    return popt_gauss, pcov_gauss, y_fit_peak



def single_plot(x_array, y_array, popt_gauss):
    y_fit_peak = _1gaussian(x_array, *popt_gauss)
    plt.plot(x_array, y_array, label= "original curve", linestyle='dashed', linewidth = 0.8, 
             marker='o', markerfacecolor='black', markersize=3, color = 'black')

    plt.plot(popt_gauss[1], np.interp(popt_gauss[1], x_array, y_fit_peak),'rx', label= "fitted peak", 
             markersize = 10, markeredgewidth=4, color = 'black') 
    
    plt.legend()
    plt.show()
    
    
    
def combined_plot(scanNo, imageNo, cake, 
                  x_array_total, y_array_total, 
                  x_array_1st, y_array_1st, y_fit_peak_1st, popt_gauss_1st,
                  x_array_2nd, y_array_2nd, y_fit_peak_2nd, popt_gauss_2nd,
                  x_array_3rd, y_array_3rd, y_fit_peak_3rd, popt_gauss_3rd):
    
    plt.close()
    

    #title
    title = 'Azimuthal integration, scan ' + str(scanNo) + ', Image ' + str(imageNo) + ',\n Cake ' + str(cake) + \
        ' (' + str(cake*4) + ' - ' + str((cake + 1)*4) + ' [\N{DEGREE SIGN}])'
        

    # Main plot     
    fig = plt.figure(figsize=(14,8))
    
    fig.suptitle(title, fontsize = 24, fontweight= 'bold')
    fig.subplots_adjust(top=0.85)
    


    
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    

    ax1.plot(x_array_total, y_array_total, label= "experimental curve", color = 'b')

    ax1.plot(x_array_1st, y_array_1st, label= "1st peak curve fitted", linewidth = 3, color = 'red')
    
    ax1.plot(popt_gauss_1st[1], np.interp(popt_gauss_1st[1], x_array_1st, y_fit_peak_1st),'rx', label= "1st peak fitted", 
             markersize = 12, markeredgewidth=2, color = 'red') 


    ax1.plot(x_array_2nd, y_array_2nd, label= "2nd peak curve fitted", linewidth = 3, color = 'green')
    
    ax1.plot(popt_gauss_2nd[1], np.interp(popt_gauss_2nd[1], x_array_2nd, y_fit_peak_2nd),'rx', label= "2nd peak fitted", 
             markersize = 12, markeredgewidth=2, color = 'green')     


    ax1.plot(x_array_3rd, y_array_3rd, label= "3rd peak curve fitted", linewidth = 3, color = 'violet')
    
    ax1.plot(popt_gauss_3rd[1], np.interp(popt_gauss_3rd[1], x_array_3rd, y_fit_peak_3rd),'rx', label= "3rd peak fitted", 
             markersize = 12, markeredgewidth=2, color = 'violet')             



    
    ax1.set_xlabel('Angle 2\u03B8  [\N{DEGREE SIGN}]',fontsize=20, fontweight='bold')
    ax1.set_ylabel('Intensity [a.u.]',fontsize=20, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    
    
    ax1.set_xlim(0, 50)
    ax1.xaxis.set_tick_params(labelsize = 20)
    ax1.yaxis.set_tick_params(labelsize = 20)

    ax1.set_aspect( 1/450, anchor = 'C' )


    # 1st peak     
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=1)
    

    ax2.plot(x_array_1st, y_array_1st, label= "experimental curve", linestyle='dashed', linewidth = 0.8, 
             marker='o', markerfacecolor='blue', markersize=3, color = 'blue')
    
    
    ax2.plot(x_array_1st, y_fit_peak_1st, label= "fitted curve",  linewidth = 1, color = 'red')
        

    ax2.plot(popt_gauss_1st[1], np.interp(popt_gauss_1st[1], x_array_1st, y_fit_peak_1st),'rx', label= "fitted peak", 
             markersize = 16, markeredgewidth=3, color = 'red') 

    peak_1_title = 'Peak 1 (002), 2\u03B8 = {:.2f}'.format(popt_gauss_1st[1])  
    
    ax2.set_title(peak_1_title, fontsize=16, fontweight='bold' )
    ax2.set_xlabel('Angle 2\u03B8  [\N{DEGREE SIGN}]',fontsize=13, fontweight='bold')
    ax2.set_ylabel('Intensity [a.u.]',fontsize=13, fontweight='bold')
    
    
    ax2.legend(loc='lower center', fontsize=14, shadow = False)


    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)



    # 2nd peak 
    ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=1)
    

    ax3.plot(x_array_2nd, y_array_2nd, label= "experimental curve", linestyle='dashed', linewidth = 0.8, 
             marker='o', markerfacecolor='blue', markersize=3, color = 'blue')
    
    
    ax3.plot(x_array_2nd, y_fit_peak_2nd, label= "fitted curve",  linewidth = 1, color = 'green')
        

    ax3.plot(popt_gauss_2nd[1], np.interp(popt_gauss_2nd[1], x_array_2nd, y_fit_peak_2nd),'rx', label= "fitted peak", 
             markersize = 16, markeredgewidth=3, color = 'green') 

    peak_2_title = 'Peak 2 (100), 2\u03B8 = {:.2f}'.format(popt_gauss_2nd[1])  


    ax3.set_title(peak_2_title, fontsize=16, fontweight='bold' )
    ax3.set_xlabel('Angle 2\u03B8  [\N{DEGREE SIGN}]',fontsize=13, fontweight='bold')
    ax3.set_ylabel('Intensity [a.u.]',fontsize=13, fontweight='bold')
    
    
    ax3.legend(loc='lower center', fontsize=14, shadow = False)


    ax3.xaxis.set_tick_params(labelsize=12)
    ax3.yaxis.set_tick_params(labelsize=12)
    
    
    
    # 3rd peak 
    ax4 = plt.subplot2grid((2, 3), (1, 2), colspan=1)
    

    ax4.plot(x_array_3rd, y_array_3rd, label= "experimental curve", linestyle='dashed', linewidth = 0.8, 
             marker='o', markerfacecolor='blue', markersize=3, color = 'blue')
    
    
    ax4.plot(x_array_3rd, y_fit_peak_3rd, label= "fitted curve",  linewidth = 1, color = 'violet')
        

    ax4.plot(popt_gauss_3rd[1], np.interp(popt_gauss_3rd[1], x_array_3rd, y_fit_peak_3rd),'rx', label= "fitted peak", 
             markersize = 16, markeredgewidth=3, color = 'violet') 

    peak_3_title = 'Peak 3 (110), 2\u03B8 = {:.2f}'.format(popt_gauss_3rd[1])  


    ax4.set_title(peak_3_title, fontsize=16, fontweight='bold' )
    ax4.set_xlabel('Angle 2\u03B8  [\N{DEGREE SIGN}]',fontsize=13, fontweight='bold')
    ax4.set_ylabel('Intensity [a.u.]',fontsize=13, fontweight='bold')
    
    
    ax4.legend(loc='lower center', fontsize=14, shadow = False)


    ax4.xaxis.set_tick_params(labelsize=12)
    ax4.yaxis.set_tick_params(labelsize=12)    

    fig.tight_layout()
    plt.show()
        
    return 0






# %% Read and import data from nexsus file 
plt.close('all')

parent_dir = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs\Raw_processed_nxs/'



half_CC1_1_no_load = [371901, 371902, 371903]
half_CC1_1_one_third_buckling_load = [371908, 371909, 371910]
half_CC1_1_one_buckling_load = [371912, 371913, 371914]

#sample CC1 (CC1)
CC1_no_load = [371919, 371920, 371921]
CC1_one_third_buckling_load = [371923, 371924, 371925]
CC1_one_buckling_load = [371927, 371928, 371929]

#sample CC2 (CC2)
CC2_no_load = [371933, 371934, 371935]
CC2_one_third_buckling_load = [371937, 371938, 371939]
CC2_one_buckling_load = [371941, 371942, 371943]

#sample half_CC1 (1_2_CC1)
half_CC1_2_no_load = [371948, 371949, 371950]
half_CC1_2_one_third_buckling_load = [371952, 371953, 371954]
half_CC1_2_one_buckling_load = [371956, 371957, 371958]


all_measurements = [371901, 371902, 371903,
                    371908, 371909, 371910,
                    371912, 371913, 371914,
                    
                    371919, 371920, 371921,
                    371923, 371924, 371925,                
                    371927, 371928, 371929, 
                    
                    371933, 371934, 371935,
                    371937, 371938, 371939,
                    371941, 371942, 371943,
                    
                    371948, 371949, 371950,
                    371952, 371953, 371954,
                    371956, 371957, 371958]



scans = all_measurements

for scanNo in scans:
    
    # scanNo =  371948 #371901
    
    
    file_name = parent_dir  + 'ipp_' + str(scanNo) + '_processed.nxs'
    
    file = h5py.File(file_name , 'a')
    
    
    # Read data from NEXSUS file for Azimuthal integration
    
    pixel = file['processed/result/pixel']
    All_azimuthal_intensity = file['processed/result/data']
    
    pixel = np.array(pixel)
    pixel = pixel_to_2theta(pixel)
    
    
    totalImage = All_azimuthal_intensity.shape[0]
    
    
    # %%
    # reading motors positions 
    motorpath = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Motors_processed/'
    
    motors = pd.read_csv(motorpath + 'ipp_' + str(scanNo) + '.csv')
    Xmotor = motors['X motor']
    Ymotor = motors['Y motor']
    
    # plt.plot(Xmotor,Ymotor, 'rx')
    
    no_images = len(All_azimuthal_intensity)
    
    
    #%% preparing the output list
    
    feature_lists = ['Image', 'Xmotor', 'Ymotor', 'Cakes',
                     'P1_amp', 'P1_x0', 'P1_SD', 'P1_amp_err', 'P1_x0_err', 'P1_SD_err',
                     'P2_amp', 'P2_x0', 'P2_SD', 'P2_amp_err', 'P2_x0_err', 'P2_SD_err',
                     'P3_amp', 'P3_x0', 'P3_SD', 'P3_amp_err', 'P3_x0_err', 'P3_SD_err']
    
    
    
    
    df = pd.DataFrame(np.zeros((no_images, len(feature_lists))), columns = feature_lists, dtype=(object))
    
    
    
    
    
    
    
    
    
    
    # %% loop beginning
    
    for imageNo in range(len(All_azimuthal_intensity)):
    # for imageNo in range(143,144):
        
        
        df.at[imageNo, 'Image'] = imageNo
        
        df.at[imageNo, 'Xmotor'] = Xmotor[imageNo]
        
        df.at[imageNo, 'Ymotor'] = Ymotor[imageNo]
        
    
        
        x_array_total = np.array(pixel) 
    
    
        # create lists to be filled with cakes and gauss parameters            
        cakes = [] 
    
            
        gauss_P1_amp, gauss_P1_x0, gauss_P1_SD = [], [], []
            
        gauss_P1_amp_err, gauss_P1_x0_err, gauss_P1_SD_err = [], [], []
            
    
    
        gauss_P2_amp, gauss_P2_x0, gauss_P2_SD = [], [], []
            
        gauss_P2_amp_err, gauss_P2_x0_err, gauss_P2_SD_err = [], [], []
            
            
            
        gauss_P3_amp, gauss_P3_x0, gauss_P3_SD = [], [], []
            
        gauss_P3_amp_err, gauss_P3_x0_err, gauss_P3_SD_err = [], [], []      
    
    #loop across the cakes
        for cake in range(len(All_azimuthal_intensity[imageNo])):
        # for cake in range(74,77): 
            
            x_array_total = np.array(pixel) 
            
            y_array_total = np.array(All_azimuthal_intensity[imageNo][cake])
            
            # plt.plot(x_array_total , y_array_total, linewidth = 2)    
            
            # plt.title('Image ' + str(imageNo) + ', Cake = ' + str(cake + 1) + ', Correctness = ' + str(chart_evaluation(x_array_total, y_array_total)) )
            
            print('Image ' + str(imageNo) + ', Cake = ' + str(cake + 1) + ', Correctness = ' + str(chart_evaluation(x_array_total, y_array_total)) )
            
            # plt.show()
            # plt.close()
            
            
            
            if chart_evaluation(pixel, y_array_total) == True :
                
                y_array_total_smooth = gaussian_filter1d(y_array_total, 15)
                
                # plt.plot(x_array_total, y_array_total, 'g', linewidth = 1)
                
                
                #finding all peaks positions
                first_peak_pos = peak_search(x_array_total, y_array_total_smooth, 550, 650, 'rx')
                
                second_peak_pos = peak_search(x_array_total, y_array_total_smooth, 1000, 1200, 'bx')
        
                third_peak_pos = peak_search(x_array_total, y_array_total_smooth, 1900, 2100, 'gx')
                
                
                
                #finding 1st peak
                        
                x_array, y_array = trim_peak(x_array_total, y_array_total, first_peak_pos, 50, 45)
        
                x_array_1st, y_array_1st = x_array, y_array
        
        
                gauss_parameters_1st = peak_fitting(x_array, y_array, imageNo, cake, 'Peak 1')   
        
                popt_gauss_1st = gauss_parameters_1st[0]
                
                pcov_gauss_1st = gauss_parameters_1st[1]
                
                y_fit_peak_1st = gauss_parameters_1st[2]      
                
                peak_1_fitted_amplitude = np.interp(popt_gauss_1st[1], x_array_1st, y_fit_peak_1st)
                
                perr_gauss_1st = np.sqrt(np.diag(pcov_gauss_1st))
                    
        
                #finding 2nd peak
                        
                x_array, y_array = trim_peak(x_array_total, y_array_total, second_peak_pos, 45, 45)
        
                x_array_2nd, y_array_2nd = x_array, y_array 
        
        
                
                gauss_parameters_2nd = peak_fitting(x_array, y_array, imageNo, cake, 'Peak 2')   
        
                popt_gauss_2nd = gauss_parameters_2nd[0]
                
                pcov_gauss_2nd = gauss_parameters_2nd[1]
                
                y_fit_peak_2nd = gauss_parameters_2nd[2]
                
                peak_2_fitted_amplitude = np.interp(popt_gauss_2nd[1], x_array_2nd, y_fit_peak_2nd)            
                
                perr_gauss_2nd = np.sqrt(np.diag(pcov_gauss_2nd))
                    
                
        
                #finding 3rd peak
                        
                x_array, y_array = trim_peak(x_array_total, y_array_total, third_peak_pos, 55, 55)
        
                x_array_3rd, y_array_3rd = x_array, y_array 
        
        
                gauss_parameters_3rd = peak_fitting(x_array, y_array, imageNo, cake, 'Peak 3')   
        
                popt_gauss_3rd = gauss_parameters_3rd[0]
                
                pcov_gauss_3rd = gauss_parameters_3rd[1]
                
                y_fit_peak_3rd = gauss_parameters_3rd[2]
                
                peak_3_fitted_amplitude = np.interp(popt_gauss_3rd[1], x_array_3rd, y_fit_peak_3rd)            
        
                perr_gauss_3rd = np.sqrt(np.diag(pcov_gauss_3rd))
                    
                
                #make nice plot (better to hide for the postprocessing in order to speed up)
                # combined_plot(scanNo, imageNo, cake, 
                #           x_array_total, y_array_total, 
                #           x_array_1st, y_array_1st, y_fit_peak_1st, popt_gauss_1st,
                #           x_array_2nd, y_array_2nd, y_fit_peak_2nd, popt_gauss_2nd,
                #           x_array_3rd, y_array_3rd, y_fit_peak_3rd, popt_gauss_3rd)
        
        #gauss parameters to be saved
                cakes.append(cake)         
                
                gauss_P1_amp.append(peak_1_fitted_amplitude)        
                gauss_P1_x0.append(gauss_parameters_1st[0][1])
                gauss_P1_SD.append(gauss_parameters_1st[0][2])
                
                gauss_P1_amp_err.append(perr_gauss_1st[0])
                gauss_P1_x0_err.append(perr_gauss_1st[1])
                gauss_P1_SD_err.append(perr_gauss_1st[2])
                
    
                
                gauss_P2_amp.append(peak_2_fitted_amplitude)        
                gauss_P2_x0.append(gauss_parameters_2nd[0][1])
                gauss_P2_SD.append(gauss_parameters_2nd[0][2])
                
                gauss_P2_amp_err.append(perr_gauss_2nd[0])
                gauss_P2_x0_err.append(perr_gauss_2nd[1])
                gauss_P2_SD_err.append(perr_gauss_2nd[2])
                
                
                
                gauss_P3_amp.append(peak_3_fitted_amplitude)        
                gauss_P3_x0.append(gauss_parameters_3rd[0][1])
                gauss_P3_SD.append(gauss_parameters_3rd[0][2])
                
                gauss_P3_amp_err.append(perr_gauss_3rd[0])
                gauss_P3_x0_err.append(perr_gauss_3rd[1])
                gauss_P3_SD_err.append(perr_gauss_3rd[2])            
                
            if len(cakes) > 0: 
                df.at[imageNo, 'Cakes'] = cakes
                
                
                df.at[imageNo, 'P1_amp'] = gauss_P1_amp
                df.at[imageNo, 'P1_x0'] = gauss_P1_x0
                df.at[imageNo, 'P1_SD'] = gauss_P1_SD
        
                df.at[imageNo, 'P1_amp_err'] = gauss_P1_amp_err
                df.at[imageNo, 'P1_x0_err'] = gauss_P1_x0_err
                df.at[imageNo, 'P1_SD_err'] = gauss_P1_SD_err          
        
        
        
                df.at[imageNo, 'P2_amp'] = gauss_P2_amp
                df.at[imageNo, 'P2_x0'] = gauss_P2_x0
                df.at[imageNo, 'P2_SD'] = gauss_P1_SD
        
                df.at[imageNo, 'P2_amp_err'] = gauss_P2_amp_err
                df.at[imageNo, 'P2_x0_err'] = gauss_P2_x0_err
                df.at[imageNo, 'P2_SD_err'] = gauss_P2_SD_err        
        
        
        
                df.at[imageNo, 'P3_amp'] = gauss_P3_amp
                df.at[imageNo, 'P3_x0'] = gauss_P3_x0
                df.at[imageNo, 'P3_SD'] = gauss_P3_SD
        
                df.at[imageNo, 'P3_amp_err'] = gauss_P3_amp_err
                df.at[imageNo, 'P3_x0_err'] = gauss_P3_x0_err
                df.at[imageNo, 'P3_SD_err'] = gauss_P3_SD_err       
                

    
    # excel = pd.DataFrame(output_data,columns=['Image','Xmotor', 'Ymotor', 'Cakes',
    #                     'P1_amp', 'P1_x0', 'P1_SD', 'P1_diff','P1_amp_err','P1_x0_err', 'P1_SD_err',
    #                     'P2_amp', 'P2_x0', 'P2_SD', 'P2_diff','P2_amp_err','P2_x0_err', 'P2_SD_err',
    #                     'P3_amp', 'P3_x0', 'P3_SD', 'P3_diff','P3_amp_err','P3_x0_err', 'P3_SD_err'])
    
    output_folder = r'C:\Users\pawel\OneDrive\Desktop\Politechnika\Moje artykuly\[Progressing]_Synchrotron_experiment\Diamond_Light_source_experiment\Nov_22_Python_scripts_final\Partial_integration_cakes_processed_nxs/'  
        
    output_file = output_folder +   "ipp_" + str(scanNo) +"_azimuthal_integration.xlsx"       
    
    df.to_excel(output_file)
    file.close()