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
from envRLKF.env_funcs_multic_AKF import LOSPRRprocess, LOSPRRprocess_afterRLAKF
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

data_truth_dic={}
gnss_dic={}
losfeature={}

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

record_data = True
id_call=True
count = -1
# define path of the testing results of RL-AKF
RLdata_path = dir_path + f'records_values/RL4KF_halftrain/source=urban_1_losposcovR_onlyRNallcorrect_conv_corr_2/' \
                         f'continuous_lstmATF1_wls_igst_lr=0.0002_pos=10_QS=1e-10_RS=1e-05_allcorr=True_unit=64/RecurrentPPO_1'
triptype = 'urban'
try:
    # with open(dir_path+'env/raw_gnss_multic.pkl', "rb") as file:
    #     gnss_dic = pickle.load(file)
    # file.close()
    with open(RLdata_path+'/raw_baseline_multic_afterRLAKF_re.pkl', "rb") as file:
        data_truth_dic = pickle.load(file)
    file.close()
except:
    filenonexist=1

for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    tripIDtraj = f'{drive}_{phone}'
    if '2021' in drive:
        # ignoring trajs without altitude
        if drive not in nullaltitude2022list and tripID not in zeroobs2022trajs:
            if (not tripID in tripIDskiplist):
                # Read data
                count += 1
                print(f'{count}:{tripID}')
                truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
                filetraj = f'{RLdata_path}/testmore_{triptype}_rl_traj_{tripIDtraj}.csv'
                try:
                    baseline_df=pd.read_csv(filetraj)
                    baseline_df=baseline_df.rename(columns={'X_RLpredict': 'X_RLAKF', 'Y_RLpredict': 'Y_RLAKF','Z_RLpredict': 'Z_RLAKF'})
                    pos_num = len(truth_df) - len(baseline_df)
                    truth_df = truth_df.drop(truth_df.index[:pos_num]).reset_index(drop=True)
                except:
                    continue

                gnss_df_raw = pd.read_csv(f'{dirname}/device_gnss.csv')
                KF_igst_realtime = pd.read_csv(f'{dirname}/KF_ecef_igst_realtime.csv')
                LOSPRR=LOSPRRprocess_afterRLAKF(gnss_df_raw,truth_df,tripID,dir_path,baseline_df)
                try:
                    gnss_df=gnss_dic[tripID]
                    pd_train=data_truth_dic[tripID]
                except:
                    gnss_df=LOSPRR.LOSPRRprocesses()
                    # gnss_dic[tripID]=gnss_df
                    pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
                    ecefxyz_kf_igst = pd_train[['X_RLAKF', 'Y_RLAKF', 'Z_RLAKF']].to_numpy()
                    lla_kf_igst = coord.ecef2geodetic(ecefxyz_kf_igst)
                    pd_train.loc[:, 'AltitudeMeters_kf_igst'] = lla_kf_igst[:,2]
                    data_truth_dic[tripID]=pd_train

                featureall, sat_summary_multicCN0=LOSPRR.getitemECEFCN0AA_RLKF(1,biastrig,gnss_df,id_call)
                try:
                    sat_summary_multicCN0AA_all.loc[:, tripID] = pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})
                    # sat_summary_multicCN0AA_all = pd.concat([sat_summary_multicCN0AA_all, pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})], sort=False)
                except:
                    sat_summary_multicCN0AA_all=sat_summary_multicCN0.rename(columns={'Nums':tripID})
                losfeature[tripID]=featureall

                tripIDlist.append(tripID)
                # truth XYZ min max baselin XYZ min max
                truth_min_X.append(np.min(pd_train['ecefX'].to_numpy()))
                truth_min_Y.append(np.min(pd_train['ecefY'].to_numpy()))
                truth_min_Z.append(np.min(pd_train['ecefZ'].to_numpy()))
                truth_max_X.append(np.max(pd_train['ecefX'].to_numpy()))
                truth_max_Y.append(np.max(pd_train['ecefY'].to_numpy()))
                truth_max_Z.append(np.max(pd_train['ecefZ'].to_numpy()))

                kf_min_X.append(np.min(baseline_df['X_RLAKF'].to_numpy()))
                kf_min_Y.append(np.min(baseline_df['Y_RLAKF'].to_numpy()))
                kf_min_Z.append(np.min(baseline_df['Z_RLAKF'].to_numpy()))
                kf_max_X.append(np.max(baseline_df['X_RLAKF'].to_numpy()))
                kf_max_Y.append(np.max(baseline_df['Y_RLAKF'].to_numpy()))
                kf_max_Z.append(np.max(baseline_df['Z_RLAKF'].to_numpy()))

                truth_min_lat.append(np.min(pd_train['LatitudeDegrees'].to_numpy()))
                truth_min_lon.append(np.min(pd_train['LongitudeDegrees'].to_numpy()))
                truth_min_alt.append(np.min(pd_train['AltitudeMeters'].to_numpy()))
                truth_max_lat.append(np.max(pd_train['LatitudeDegrees'].to_numpy()))
                truth_max_lon.append(np.max(pd_train['LongitudeDegrees'].to_numpy()))
                truth_max_alt.append(np.max(pd_train['AltitudeMeters'].to_numpy()))

#%% raw data saving

if record_data:
    with open(RLdata_path + f'/raw_baseline_multic_{triptype}_afterRLAKF.pkl', 'wb') as value_file:
        pickle.dump(data_truth_dic, value_file, True)
    value_file.close()
    with open(RLdata_path + '/raw_gnss_multic_afterRLAKF.pkl', 'wb') as value_file:
        pickle.dump(gnss_dic, value_file, True)
    value_file.close()
    with open(RLdata_path + f'/processed_features_multic_{triptype}_afterRLAKF.pkl', 'wb') as value_file:
        pickle.dump(losfeature, value_file, True)
    value_file.close()

    with open(RLdata_path + '/raw_tripID_multicCN0AA_lla_afterRLAKF.pkl', 'wb') as value_file:
        pickle.dump(tripIDlist, value_file, True)
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

    tripID_df.to_csv(RLdata_path + '/raw_tripID_multicCN0AA_lla.csv', index=True)
    sat_summary_multicCN0AA_all.to_csv(RLdata_path + '/raw_satnum_multicCN0AA_lla_50_afterRLAKF.csv', index=True)

# for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
#     drive, phone = dirname.split('/')[-3:-1]
#     tripID = f'{drive}/{phone}'
#     if (not tripID in tripIDskiplist):
#         # Read data
#         truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
#         baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
#         pd_train = truth_df[train_columns].merge(data_train, on=merge_columns, suffixes=("_truth", ""))
