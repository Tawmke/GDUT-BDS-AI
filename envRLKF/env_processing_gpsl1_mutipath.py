import gym
from gym import spaces
import random
import numpy as np
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import simdkalman
from env.env_funcs_gpsl1 import LOSPRRprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#导入数据
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'
# 10.23.18.26
data_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'

path_data = Path(data_path)
path_truth_data = (path_data/'train').rglob('ground_truth.csv')

data_truth_dic={}
gnss_dic={}
losfeature={}
try:
    with open(dir_path+'env/raw_gnss_gpsl1_mutipath.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()
    with open(dir_path+'env/raw_baseline_gpsl1_mutipath.pkl', "rb") as file:
        data_truth_dic = pickle.load(file)
    file.close()
except:
    filenonexist=1
tripIDlist=[]
# Ground truth: LatitudeDegrees_truth
# KF: LatitudeDegrees
# Baseline(WLS): LatitudeDegrees_bl
# Robust WLS: LatitudeDegrees_wls
train_columns = ['UnixTimeMillis','LatitudeDegrees','LongitudeDegrees','AltitudeMeters',
                 'ecefX','ecefY','ecefZ']
merge_columns = ['UnixTimeMillis']
# KF nan skip list
tripIDskiplist=['2020-08-03-US-MTV-2/GooglePixel5']
nullaltitude2022list=['2021-01-05-US-MTV-1', '2021-01-05-US-MTV-2']
zeroobs2022trajs=['2021-03-16-US-MTV-3/XiaomiMi8', '2021-04-02-US-SJC-1/GooglePixel4', '2021-04-02-US-SJC-1/XiaomiMi8',
                   '2021-04-26-US-SVL-2/SamsungGalaxyS20Ultra', '2021-04-26-US-SVL-2/XiaomiMi8', '2021-07-14-US-MTV-1/GooglePixel5',
                   '2021-07-14-US-MTV-1/SamsungGalaxyS20Ultra', '2021-07-14-US-MTV-1/XiaomiMi8', '2021-07-27-US-MTV-1/GooglePixel5',
                   '2021-12-07-US-LAX-1/GooglePixel5', '2021-12-07-US-LAX-2/GooglePixel5', '2021-12-09-US-LAX-2/GooglePixel5',
                   '2021-07-19-US-MTV-1/XiaomiMi8', '2021-08-24-US-SVL-1/SamsungGalaxyS20Ultra']
"""
# check minmax value
data_truth_list =[]
for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    if (not tripID in tripIDskiplist):
        # Read data
        truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
        baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
        pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
        data_truth_list.append(pd_train)
data_train_truth = pd.concat(data_truth_list, ignore_index=True)
ground_truth_colums = ['LatitudeDegrees_truth','LongitudeDegrees_truth']
minmax_=[]
for col in ground_truth_colums:
    max_ = np.max(data_train_truth[col])
    min_ = np.min(data_train_truth[col])
    minmax_.append([max_,min_])
"""
# Loop for each trip
truth_min_X=[]
truth_min_Y=[]
truth_min_Z=[]
truth_max_X=[]
truth_max_Y=[]
truth_max_Z=[]
kf_min_X=[]
kf_min_Y=[]
kf_min_Z=[]
kf_max_X=[]
kf_max_Y=[]
kf_max_Z=[]
biastrig=1

id_call=True
for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    if '2021' in drive:
        # ignoring trajs without altitude
        if drive not in nullaltitude2022list and tripID not in zeroobs2022trajs:
            if (not tripID in tripIDskiplist):
                print(tripID)
                # Read data
                truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
                baseline_df=pd.read_csv(f'{dirname}/baseline_ecef_igst.csv')
                gnss_df_raw = pd.read_csv(f'{dirname}/device_gnss.csv')
                LOSPRR=LOSPRRprocess(gnss_df_raw,truth_df,tripID,dir_path,baseline_df)
                try:
                    gnss_df=gnss_dic[tripID]
                    pd_train=data_truth_dic[tripID]
                except:
                    gnss_df=LOSPRR.LOSPRRprocesses()
                    gnss_dic[tripID]=gnss_df
                    pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
                    data_truth_dic[tripID]=pd_train
                featureall, sat_summary_gpsl1=LOSPRR.getitemECEF_mutipath(1,biastrig,gnss_df,id_call)
                try:
                    sat_summary_gpsl1_all.loc[:, tripID] = pd.DataFrame({tripID:sat_summary_gpsl1['Nums']})
                    # sat_summary_gpsl1_all = pd.concat([sat_summary_gpsl1_all, pd.DataFrame({tripID:sat_summary_gpsl1['Nums']})], sort=False)
                except:
                    sat_summary_gpsl1_all=sat_summary_gpsl1.rename(columns={'Nums':tripID})
                losfeature[tripID]=featureall
                tripIDlist.append(tripID)
                # truth XYZ min max baselin XYZ min max
                truth_min_X.append(np.min(pd_train['ecefX'].to_numpy()))
                truth_min_Y.append(np.min(pd_train['ecefY'].to_numpy()))
                truth_min_Z.append(np.min(pd_train['ecefZ'].to_numpy()))
                truth_max_X.append(np.max(pd_train['ecefX'].to_numpy()))
                truth_max_Y.append(np.max(pd_train['ecefY'].to_numpy()))
                truth_max_Z.append(np.max(pd_train['ecefZ'].to_numpy()))

                kf_min_X.append(np.min(baseline_df['XEcefMeters_kf_igst'].to_numpy()))
                kf_min_Y.append(np.min(baseline_df['YEcefMeters_kf_igst'].to_numpy()))
                kf_min_Z.append(np.min(baseline_df['ZEcefMeters_kf_igst'].to_numpy()))
                kf_max_X.append(np.max(baseline_df['XEcefMeters_kf_igst'].to_numpy()))
                kf_max_Y.append(np.max(baseline_df['YEcefMeters_kf_igst'].to_numpy()))
                kf_max_Z.append(np.max(baseline_df['ZEcefMeters_kf_igst'].to_numpy()))

#%% raw data saving

# with open(dir_path + 'env/raw_baseline_gpsl1.pkl', 'wb') as value_file:
#     pickle.dump(data_truth_dic, value_file, True)
# value_file.close()
# with open(dir_path + 'env/raw_gnss_gpsl1_mutipath.pkl', 'wb') as value_file:
#     pickle.dump(gnss_dic, value_file, True)
# value_file.close()
with open(dir_path + 'envmutipath/processed_features_gpsl1_ecef_id_mutipath.pkl', 'wb') as value_file:
    pickle.dump(losfeature, value_file, True)
value_file.close()
# with open(dir_path + 'env/raw_tripID_gpsl1.pkl', 'wb') as value_file:
#     pickle.dump(tripIDlist, value_file, True)
# value_file.close()
# tripID_df=pd.DataFrame(tripIDlist, columns=['tripID'])
# tripID_df['ecefX_min']=truth_min_X
# tripID_df['ecefY_min']=truth_min_Y
# tripID_df['ecefZ_min']=truth_min_Z
# tripID_df['ecefX_max']=truth_max_X
# tripID_df['ecefY_max']=truth_max_Y
# tripID_df['ecefZ_max']=truth_max_Z
#
# tripID_df['ecefX_min_kf']=kf_min_X
# tripID_df['ecefY_min_kf']=kf_min_Y
# tripID_df['ecefZ_min_kf']=kf_min_Z
# tripID_df['ecefX_max_kf']=kf_max_X
# tripID_df['ecefY_max_kf']=kf_max_Y
# tripID_df['ecefZ_max_kf']=kf_max_Z
# tripID_df.to_csv(dir_path + 'env/raw_tripID_gpsl1.csv', index=True)
#
# sat_summary_gpsl1_all.to_csv(dir_path + 'env/raw_satnum_gpsl1_50.csv', index=True)



# for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
#     drive, phone = dirname.split('/')[-3:-1]
#     tripID = f'{drive}/{phone}'
#     if (not tripID in tripIDskiplist):
#         # Read data
#         truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
#         baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
#         pd_train = truth_df[train_columns].merge(data_train, on=merge_columns, suffixes=("_truth", ""))
