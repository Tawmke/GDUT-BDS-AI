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
import sys
sys.path.append("..")
from envRLKF.env_funcs_multic_AKF import LOSPRRprocess
import src.gnss_lib.coordinates as coord

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#导入数据
# 10.23.18.26
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'
data_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/'
#%%
file_train = os.path.join(dir_path, "baseline_locations_train_2022.csv")
#%%
data_train = pd.read_csv(file_train)
#%%
path_data = Path(data_path)
path_truth_data = (path_data/'train').rglob('ground_truth.csv')
#%%
# data_truth_list =[]
# for file_name in tqdm(path_truth_data):
#     data_file = pd.read_csv(file_name)
#     data_truth_list.append(data_file)
# data_train_truth = pd.concat(data_truth_list, ignore_index=True)
# train_columns = ['UnixTimeMillis','LatitudeDegrees','LongitudeDegrees']
# merge_columns = ['UnixTimeMillis']
# pd_train_all = data_train_truth[train_columns].merge(data_train,
#                        on=merge_columns,
#                       suffixes=("_truth",""))

# import src.gnss_lib.coordinates as coord
# lla1=np.array([[23.01749199,113.0431822204,14.415],[23.145287380,113.361804403,15.3665]])
# ecef1 = coord.geodetic2ecef(lla1)

data_truth_dic={}
gnss_dic={}
losfeature={}
Complementaryfeature = {}
# record velocity and position
cov_xv_dic={}
data_truth_velocity_dic={}
data_kf_igst_realtime_dic={}
try:
    # with open(dir_path+'env/raw_gnss_multic.pkl', "rb") as file:
    #     gnss_dic = pickle.load(file)
    # file.close()
    with open(dir_path+'env/raw_baseline_multic_re.pkl', "rb") as file:
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
allzeroobs2022trajs=['2021-04-02-US-SJC-1/GooglePixel4', '2021-07-14-US-MTV-1/GooglePixel5',
                     '2021-07-27-US-MTV-1/GooglePixel5', '2021-12-07-US-LAX-1/GooglePixel5',
                     '2021-12-07-US-LAX-2/GooglePixel5', '2021-12-09-US-LAX-2/GooglePixel5', ]
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
truth_min_lat=[]
truth_min_lon=[]
truth_min_alt=[]
truth_max_lat=[]
truth_max_lon=[]
truth_max_alt=[]
kf_min_lat=[]
kf_min_lon=[]
kf_min_alt=[]
kf_max_lat=[]
kf_max_lon=[]
kf_max_alt=[]
biastrig=1
# Record AKF
covx_diff_list = []
covv_diff_list = []
covx_avg_list = []
covv_avg_list = []
covx_list = []
covv_list = []

record_data = True
id_call=True
count = -1
trip_wls_igst_null = ['2021-04-29-US-MTV-1/SamsungGalaxyS20Ultra','2021-07-01-US-MTV-1/XiaomiMi8',
                      '2021-12-07-US-LAX-1/XiaomiMi8','2021-12-07-US-LAX-2/XiaomiMi8','2021-12-08-US-LAX-1/GooglePixel6Pro',
                      '2021-12-09-US-LAX-2/XiaomiMi8']# ['2021-07-01-US-MTV-1/XiaomiMi8','2021-12-07-US-LAX-1/XiaomiMi8','2021-12-07-US-LAX-2/XiaomiMi8','2021-12-08-US-LAX-1/GooglePixel6Pro','2021-12-09-US-LAX-2/XiaomiMi8']

def calculate_cov_avgNdiff(cov_v,cov_x):
    conv_diff = np.zeros((len(cov_v)-1,3,3))
    conx_diff = np.zeros((len(cov_v)-1, 3, 3))
    conv_avg = np.zeros((len(cov_v)-1,3,3))
    conx_avg = np.zeros((len(cov_v)-1, 3, 3))
    for i in range(len(cov_v)):
        if i == 0:
            continue
        cov_v_diff_temp = (cov_v[i] - cov_v[i - 1])
        cov_v_avg_temp = (cov_v[i] + cov_v[i - 1]) * 0.5
        conv_diff[i - 1] = cov_v_diff_temp
        conv_avg[i - 1] = cov_v_avg_temp

    for i in range(len(cov_x)):
        if i == 0:
            continue
        cov_x_diff_temp = (cov_x[i] - cov_x[i - 1])
        cov_x_avg_temp = (cov_x[i] + cov_x[i - 1]) * 0.5
        conx_diff[i - 1] = cov_x_diff_temp
        conx_avg[i - 1] = cov_x_avg_temp

    return conx_diff,conv_diff,conx_avg,conv_avg

for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    # if tripID not in trip_wls_igst_null:
    #     continue
    if '2021' in drive:
        # ignoring trajs without altitude
        if drive not in nullaltitude2022list and tripID not in zeroobs2022trajs:
            if (not tripID in tripIDskiplist):
                # Read data
                count += 1
                print(f'{count}:{tripID}')
                # if count==75:
                #     print(tripID)
                #     continue

                truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
                baseline_df=pd.read_csv(f'{dirname}/baseline_ecef_igst.csv')
                gnss_df_raw = pd.read_csv(f'{dirname}/device_gnss.csv')
                Velocity_ecef_igst = pd.read_csv(f'{dirname}/Velocity_ecef_igst.csv')
                KF_igst_realtime = pd.read_csv(f'{dirname}/KF_ecef_igst_realtime.csv')
                with open(dirname + '/processed_covariance_velocity.pkl', "rb") as file:
                    cov_v = pickle.load(file)
                file.close()
                with open(dirname + '/processed_covariance_position.pkl', "rb") as file:
                    cov_x = pickle.load(file)
                file.close()

                LOSPRR=LOSPRRprocess(gnss_df_raw,truth_df,tripID,dir_path,baseline_df)
                try:
                    gnss_df=gnss_dic[tripID]
                    pd_train=data_truth_dic[tripID]=0
                except:
                    gnss_df=LOSPRR.LOSPRRprocesses()
                    # gnss_dic[tripID]=gnss_df
                    pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
                    ecefxyz_kf_igst = pd_train[
                        ['XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']].to_numpy()
                    lla_kf_igst = coord.ecef2geodetic(ecefxyz_kf_igst)
                    pd_train.loc[:, 'AltitudeMeters_kf_igst'] = lla_kf_igst[:,2]
                    data_truth_dic[tripID]=pd_train
                    # record cov of p and v
                    data_truth_velocity_dic[tripID]=Velocity_ecef_igst
                    data_kf_igst_realtime_dic[tripID]=KF_igst_realtime
                    cov_xv_dic[tripID]={'covx':cov_x,'covv':cov_v}
                featureall, sat_summary_multicCN0, Complementaryfeatureall=LOSPRR.getitemECEFCN0AA_RLKF(1,biastrig,gnss_df,id_call)
                try:
                    sat_summary_multicCN0AA_all.loc[:, tripID] = pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})
                    # sat_summary_multicCN0AA_all = pd.concat([sat_summary_multicCN0AA_all, pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})], sort=False)
                except:
                    sat_summary_multicCN0AA_all=sat_summary_multicCN0.rename(columns={'Nums':tripID})
                losfeature[tripID]=featureall
                Complementaryfeature[tripID]=Complementaryfeatureall

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

                truth_min_lat.append(np.min(pd_train['LatitudeDegrees'].to_numpy()))
                truth_min_lon.append(np.min(pd_train['LongitudeDegrees'].to_numpy()))
                truth_min_alt.append(np.min(pd_train['AltitudeMeters'].to_numpy()))
                truth_max_lat.append(np.max(pd_train['LatitudeDegrees'].to_numpy()))
                truth_max_lon.append(np.max(pd_train['LongitudeDegrees'].to_numpy()))
                truth_max_alt.append(np.max(pd_train['AltitudeMeters'].to_numpy()))

                kf_min_lat.append(np.min(pd_train['LatitudeDegrees_kf_igst'].to_numpy()))
                kf_min_lon.append(np.min(pd_train['LongitudeDegrees_kf_igst'].to_numpy()))
                kf_min_alt.append(np.min(pd_train['AltitudeMeters_kf_igst'].to_numpy()))
                kf_max_lat.append(np.max(pd_train['LatitudeDegrees_kf_igst'].to_numpy()))
                kf_max_lon.append(np.max(pd_train['LongitudeDegrees_kf_igst'].to_numpy()))
                kf_max_alt.append(np.max(pd_train['AltitudeMeters_kf_igst'].to_numpy()))

                # # record noise conv max and min
                # conx_diff,conv_diff,conx_avg,conv_avg = calculate_cov_avgNdiff(cov_v, cov_x)
                # covx_diff_list.append(conx_diff)
                # covv_diff_list.append(conv_diff)
                # covx_avg_list.append(conx_avg)
                # covv_avg_list.append(conv_avg)
                # covx_list.append(cov_x)
                # covv_list.append(cov_v)

# cov_x_all = np.concatenate(covx_list)
# cov_v_all = np.concatenate(covv_list)
# print(f'covx_mean={np.mean(cov_x_all)}+{np.std(cov_x_all)}')
# print(f'covv_mean={np.mean(cov_v_all)}+{np.std(cov_v_all)}')
# convv_diff_all = np.concatenate(covv_diff_list)
# convx_diff_all = np.concatenate(covx_diff_list)
# convx_avg_all = np.concatenate(covx_avg_list)
# convv_avg_all = np.concatenate(covv_avg_list)
# print(f'convv_diff_mean={np.mean(convv_diff_all)}')
# print(f'convv_diff_std={np.std(convv_diff_all)}')
# print(f'convx_diff_mean={np.mean(convx_diff_all)}')
# print(f'convx_diff_std={np.std(convx_diff_all)}')
#
# print(f'convx_avg_mean={np.mean(convx_avg_all)}')
# print(f'convx_avg_std={np.std(convx_avg_all)}')
# print(f'convv_avg_mean={np.mean(convv_avg_all)}')
# print(f'convv_avg_std={np.std(convv_avg_all)}')
#%% raw data saving

if record_data:
    with open(dir_path + 'env/raw_baseline_multic_re.pkl', 'wb') as value_file:
        pickle.dump(data_truth_dic, value_file, True)
    value_file.close()
    with open(dir_path + 'env/raw_gnss_multic_lla.pkl', 'wb') as value_file:
        pickle.dump(gnss_dic, value_file, True)
    value_file.close()
    with open(dir_path + 'envRLKF/processed_features_multic_RLAKF.pkl', 'wb') as value_file:
        pickle.dump(losfeature, value_file, True)
    value_file.close()

    with open(dir_path + 'envRLKF/raw_tripID_multicCN0AA_lla.pkl', 'wb') as value_file:
        pickle.dump(tripIDlist, value_file, True)
    value_file.close()
    ### record for AKF-RL
    # record conv velocity and position
    with open(dir_path + 'envRLKF/processed_ecef_covariance_wls.pkl', 'wb') as value_file:
        pickle.dump(cov_xv_dic, value_file, True)
    value_file.close()
    # record velocity of wls
    with open(dir_path + 'envRLKF/raw_baseline_velocity.pkl', 'wb') as value_file:
        pickle.dump(data_truth_velocity_dic, value_file, True)
    value_file.close()
    with open(dir_path + 'envRLKF/raw_kf_igst_realtime.pkl', 'wb') as value_file:
        pickle.dump(data_kf_igst_realtime_dic, value_file, True)
    value_file.close()

    with open(dir_path + 'envRLKF/Complementaryfeature_multic_RLAKF.pkl', 'wb') as value_file:
        pickle.dump(Complementaryfeature, value_file, True) # 补充观测量
    value_file.close()

    tripID_df=pd.DataFrame(tripIDlist, columns=['tripID'])
    tripID_df['ecefX_min']=truth_min_X
    tripID_df['ecefY_min']=truth_min_Y
    tripID_df['ecefZ_min']=truth_min_Z
    tripID_df['ecefX_max']=truth_max_X
    tripID_df['ecefY_max']=truth_max_Y
    tripID_df['ecefZ_max']=truth_max_Z

    tripID_df['ecefX_min_kf']=kf_min_X
    tripID_df['ecefY_min_kf']=kf_min_Y
    tripID_df['ecefZ_min_kf']=kf_min_Z
    tripID_df['ecefX_max_kf']=kf_max_X
    tripID_df['ecefY_max_kf']=kf_max_Y
    tripID_df['ecefZ_max_kf']=kf_max_Z

    tripID_df['lat_min']=truth_min_lat
    tripID_df['lon_min']=truth_min_lon
    tripID_df['alt_min']=truth_min_alt
    tripID_df['lat_max']=truth_max_lat
    tripID_df['lon_max']=truth_max_lon
    tripID_df['alt_max']=truth_max_alt

    tripID_df['lat_min_kf']=kf_min_lat
    tripID_df['lon_min_kf']=kf_min_lon
    tripID_df['alt_min_kf']=kf_min_alt
    tripID_df['lat_max_kf']=kf_max_lat
    tripID_df['lon_max_kf']=kf_max_lon
    tripID_df['alt_max_kf']=kf_max_alt

    # tripID_df.to_csv(dir_path + 'env/raw_tripID_multicCN0AA_lla.csv', index=True)
    #
    # sat_summary_multicCN0AA_all.to_csv(dir_path + 'envRLKF/raw_satnum_multicCN0AA_lla_50.csv', index=True)

# for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
#     drive, phone = dirname.split('/')[-3:-1]
#     tripID = f'{drive}/{phone}'
#     if (not tripID in tripIDskiplist):
#         # Read data
#         truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
#         baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
#         pd_train = truth_df[train_columns].merge(data_train, on=merge_columns, suffixes=("_truth", ""))
print('processing finish')