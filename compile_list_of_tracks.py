import pandas as pd 
import numpy as np
from pathlib import Path

## define dictionaries to unify naming scheme;
## also specify any conditions to remove

condition_mapping = {}
conditions_to_remove = {}

condition_mapping[30] = {
    'G7B8G2_GSK_500uMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nmdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_100uM_IAA_added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_100uM_IAAadded2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '15A_A6_G9_GSK': '15A-A6G9_GSK',
    '15A-A6G9_GSK_IAA(100uM)added2hrBefore': '15A-A6G9_GSK_RAD21depletion',
    'S+V-A6B8_GSK_IAA(100uM)added2hrBefore': 'S+V-A6B8_GSK_RAD21depletion',
    '15B-18G9_GSK_IAA(100uM)added2hrBefore': '15B-18G9_GSK_RAD21depletion',
    'S-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCreG5H7': 'SCre-G5-H7_GSK',
    '15B-18G9': '15B-18G9_GSK',
    '15B18G9_GSK': '15B-18G9_GSK',
    'VCreC2b_GSK': 'VCre-C2b_GSK',
    'S+V-A6B8': 'S+V-A6B8_GSK',
    'G7B8G2_GSK_IAA(100uM)added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '14A-A11E6_GSK': '14A-A11E6_GSK',
    '14A-A11_GSK': '14A-A11_GSK',
    '14B5-C9': '14B5-C9_GSK',
    '14B5-F10': '14B5-F10_GSK',
    '14B5-F8': '14B5-F8_GSK',
    '14B5-F12': '14B5-F12_GSK',
    'E11G8_GSK': 'E11G8_GSK',
    '15C-A2_GSK':'15C-A2_GSK',
    'Vika-H11_GSK': 'Vika-H11_GSK',
    'G7B8G2_GSK': 'G7B8G2_GSK',
    'G7B8G2_GSK_dTAG13(500uM)andIAA(100uM)added2hrBefore': 'G7B8G2_GSK_CTCF_RAD21depletion',
    '15C-A2_GSK_IAA(100uM)added2hrBefore': '15C-A2_GSK_RAD21depletion',
    '14B5-F10_GSK_IAA(100uM)added2hrBefore': '14B5-F10_GSK_RAD21depletion',
}

conditions_to_remove[30] = ['14B5-F12_GSK','14B5-F8_GSK','14B5-C9_GSK']

condition_mapping[5] = {
    'G7B8G2_GSK_500uMdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_500nmdTAG13added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added2hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added3hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added4hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added5hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added6hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_GSK_dTAG13(500uM)added7hrBefore': 'G7B8G2_GSK_CTCFdepletion',
    'G7B8G2_100uM_IAA_added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added5hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added4hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added6hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'G7B8G2_GSK_IAA(100uM)added3hrBefore': 'G7B8G2_GSK_RAD21depletion',
    '15A_A6_G9_GSK': '15A-A6G9_GSK',
    '15A-A6G9': '15A-A6G9_GSK',
    'S-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCre-G5H7_GSK': 'SCre-G5-H7_GSK',
    'SCreG5H7': 'SCre-G5-H7_GSK',
    '15B-18G9': '15B-18G9_GSK',
    '15B18G9_GSK': '15B-18G9_GSK',
    '15B18B9': '15B-18G9_GSK',
    '15B18G9': '15B-18G9_GSK',
    'VCreC2b_GSK': 'VCre-C2b_GSK',
    'S+V-A6B8': 'S+V-A6B8_GSK',
    'G7B8G2_GSK_IAA(100uM)added2hrBefore': 'G7B8G2_GSK_RAD21depletion',
    'VCre-C2b_GSK': 'VCre-C2b_GSK',
    'E11G8_GSK': 'E11G8_GSK',
    'Vika-H11_GSK': 'Vika-H11_GSK',
    'G7B8G2_GSK': 'G7B8G2_GSK',
    '15C-A2_GSK':'15C-A2_GSK',
    '15A-A6G9_GSK':'15A-A6G9_GSK',
}

conditions_to_remove[5] = []

folders_path_dict = {5: '/mnt/md0/jjusuf/bild/final_tracks_20250310/export_qc_filtered_5sTracks',
                    30: '/mnt/md0/jjusuf/bild/final_tracks_20250310/export_qc_filtered_30sTracks'}


## now generate a table of all tracks

all_tracks = []

for delta_t in [5, 30]:

    folders_path = folders_path_dict[delta_t]
    folders_path_obj = Path(folders_path)
    for folder in folders_path_obj.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        date = folder_name[:8]
        condition_name_raw = folder_name[9:].split('_30ms')[0]

        if condition_name_raw in condition_mapping[delta_t].values():
            condition_name = condition_name_raw
        else:
            condition_name = condition_mapping[delta_t][condition_name_raw]
        
        if condition_name in conditions_to_remove[delta_t]:
            continue
        
        for file_path in (folders_path_obj / folder_name).iterdir():
            file_path_str = str(file_path)
            file_path_list = file_path_str.split('/')
            name = f'{file_path_list[-2]}_{file_path_list[-1][:-4]}'
            track_len = len(pd.read_csv(file_path_str))
            all_tracks.append([date, delta_t, condition_name, file_path_str, name, track_len])

all_tracks = pd.DataFrame(all_tracks, columns=['date','delta_t','condition','path','name','track_len'])

all_tracks.to_csv('all_tracks.csv')

