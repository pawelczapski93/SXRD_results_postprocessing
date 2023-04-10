# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:35:52 2022

@author: pawel
"""

import pathlib



file = [371901, 371902, 371903, 
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
        371956, 371957, 371958] #State scan number

for scanNo in file:
    dirNexOUT = 'E:/dls/dawn_output_folder/'
    dirNexIN = 'E:/dls/mm28395-1_JiraphantSrisuriyachot/ippimages/' + 'ipp_' + str(scanNo) 
    
    
    initial_count = 0
    for path in pathlib.Path(dirNexIN).iterdir():
        if path.is_file():
            initial_count +=1
    print(initial_count)
    
    
    nex = open(dirNexOUT + 'ipp_' + str(scanNo) +'.dawn', 'w')
    nex.write('# DIR_NAME: ' + dirNexIN + '\n')
    nex.write('# DATASET_NAME: image-01' + '\n')
    nex.write('# FILE_NAME' + '\n')
    for i in range(1, initial_count + 1):
        nex.write('ipp_' + str(scanNo) + '_' + str(i) + '.TIF' + '\n')
    nex.close()