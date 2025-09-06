#强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
from env.env_param import *
from scipy.spatial import distance
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error
import simdkalman
step_print=False
mutipath=True
#导入数据
dir_path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022/' # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
# load raw data
with open(dir_path+'env/raw_baseline_multic_re.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
with open(dir_path + 'envRLKF/raw_kf_igst_realtime.pkl', "rb") as file:
    data_kf_realtime = pickle.load(file)  # load kf realtime
file.close()

gnss_trig=True
if gnss_trig:
    with open(dir_path+'env/raw_gnss_gpsl1.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()
with open(dir_path + 'env/raw_tripID_gpsl1.pkl', "rb") as file:
    tripIDlist_full = pickle.load(file)
file.close()

with open(dir_path + 'env/processed_features_gpsl1_ecef_id.pkl', "rb") as file:
    losfeature_all = pickle.load(file)
file.close()
# load raw velocity data
# with open(dir_path+'envRLKF/raw_baseline_velocity.pkl', "rb") as file:
#     data_truth_velocity_dic = pickle.load(file)
# file.close()
# # load 初始协方差数据 data
# with open(dir_path+'envRLKF/processed_ecef_covariance_wls.pkl', "rb") as file:
#     cov_xv_dic = pickle.load(file)
# file.close()

random.seed(0)

# satnum_df = pd.read_csv(f'{dir_path}/env/raw_satnum_gpsl1.csv')
satnum_df = pd.read_csv(f'{dir_path}/env/raw_satnum_gpsl1.csv')
traj_sum_df = pd.read_csv(f'{dir_path}/env/traj_summary_2.csv')
higwayID_traj = (traj_sum_df['Type']=='highway') & (traj_sum_df['AKF_wlsobs_Rcorr']=='improve')
traj_highway= traj_sum_df.loc[higwayID_traj]['tripId'].values.tolist()
traj_highway = traj_highway[::-1]
traj_urban=traj_sum_df.loc[traj_sum_df['Type']=='urban']['tripId'].values.tolist()
traj_full = traj_sum_df.loc[(traj_sum_df['Type']=='urban') | (traj_sum_df['Type']=='highway')]['tripId'].values.tolist()
# del traj list ID
# del_traj = ['2021-01-04-US-SFO-1/GooglePixel4XL','2021-01-04-US-SFO-2/GooglePixel4','2021-01-04-US-SFO-2/GooglePixel4XL',
#             '2021-07-14-US-MTV-1/GooglePixel4','2021-07-27-US-MTV-1/GooglePixel4','2021-12-09-US-LAX-2/SamsungGalaxyS20Ultra','2021-12-09-US-LAX-2/XiaomiMi8']
# for value in del_traj:
#     if value in traj_full:
#         traj_full.remove(value)
#         traj_highway.remove(value)
KF_colums = ['XEcefMeters_kf','YEcefMeters_kf','ZEcefMeters_kf']
KF_colums_igst = ['XEcefMeters_kf_igst','YEcefMeters_kf_igst','ZEcefMeters_kf_igst']
ground_truth_colums = ['ecefX','ecefY','ecefZ']

CN0PRUAAEA_num = 8 # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角+方位角
CN0PRUEA_num = 7 # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角
CN0EA_num = 6 # 伪距残差+LOS（3D）+CN0+高度角

class GPSPosition_continuous_lospos(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos, self).__init__()
        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * 4)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})

        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'full':
            self.tripIDlist = traj_full

        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float)#shape=(2, 1)
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.kf_realtime = data_kf_realtime[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature=losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.datatime=self.baseline['UnixTimeMillis']
        self.timeend=self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values)-1, 'UnixTimeMillis']
        #normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss=gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat=satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls_igst':
            obs = np.append(obs,[[self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_KF_realtime']],
                                 [self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_KF_realtime']],
                                 [self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_KF_realtime']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'), 'gnss': obs_feature.reshape(1, 4 * self.max_visible_sat, order='C')}
        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UnixTimeMillis']
        # action for new prediction
        action=np.reshape(action,[1,3])
        predict_x = action[0,0]*self.continuous_action_scale
        predict_y = action[0,1]*self.continuous_action_scale
        predict_z = action[0,2]*self.continuous_action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']
        elif self.baseline_mod == 'wls_igst': # 用实时卡尔曼滤波
            obs_x = self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_KF_realtime']
            obs_y = self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_KF_realtime']
            obs_z = self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_KF_realtime'] # modified in 0412 17.45
        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        correct_label = np.array([gro_x - obs_x, gro_y - obs_y, gro_z - obs_z, gro_x - rl_x, gro_y - rl_y, gro_z - rl_z, action[0,0],action[0,1], action[0,2]])
        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0

        error = np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))
        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline,'error':error,'label':correct_label} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class GPSPosition_continuous_lospos_re(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos_re, self).__init__()
        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss':spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * 4)),
                                              'pos':spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float)})

        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'semiurban':
            self.tripIDlist = traj_semiurban
        elif triptype == 'urban':
            self.tripIDlist = traj_urban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3), dtype=np.float) # modified for RLKF
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.kf_realtime = data_kf_realtime[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.baseline_speed = data_truth_velocity_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.covxv = cov_xv_dic[self.tripIDlist[self.tripIDnum]].copy()
        # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
        # so the variance is increased
        VX = self.baseline_speed['VXEcefMeters_wls_igst'].values
        VY = self.baseline_speed['VYEcefMeters_wls_igst'].values
        VZ = self.baseline_speed['VZEcefMeters_wls_igst'].values
        if 'XiaomiMi8' in self.tripIDlist[self.tripIDnum]:
            VX = np.append((VX[:-1] + VX[1:]) / 2, 0)
            VY = np.append((VY[:-1] + VY[1:]) / 2, 0)
            VZ = np.append((VZ[:-1] + VZ[1:]) / 2, 0)
            self.covxv['covv'] = 1000.0 ** 2 * self.covxv['covv']
        self.baseline_speed['VXEcefMeters_wls_igst'] = np.insert((VX[:-1] + VX[1:]) / 2, 0,0)
        self.baseline_speed['VYEcefMeters_wls_igst'] = np.insert((VY[:-1] + VY[1:]) / 2, 0,0)
        self.baseline_speed['VZEcefMeters_wls_igst'] = np.insert((VZ[:-1] + VZ[1:]) / 2, 0,0)
        self.P = 5.0 ** 2 * np.eye(3)  # initial State covariance

        self.datatime=self.baseline['UnixTimeMillis']
        self.timeend=self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values)-1, 'UnixTimeMillis']
        #normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss=gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat = satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self,gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:,1]=(gnss[:,1]) / max(res_max, np.abs(res_min))
        gnss[:,2]=(gnss[:,2]) / max(losx_max, np.abs(losx_min))
        gnss[:,3]=(gnss[:,3]) / max(losy_max, np.abs(losy_min))
        gnss[:,4]=(gnss[:,4]) / max(losz_max, np.abs(losz_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num-2), 'Z_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls_igst']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:,0]:
                obs_feature[i,:]=feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'), 'gnss': obs_feature.reshape(1, 4 * self.max_visible_sat, order='C')}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action): # modified in 3.3
        # judge if end #
        done=(self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values)*self.traj_type[-1] - (self.pos_num) - outlayer_in_end_ecef)
        satnum = len(self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features'])
        timestep=self.baseline.loc[self.current_step + (self.pos_num-1), 'UnixTimeMillis']
        # action for new prediction
        action=np.reshape(action,[1,3])
        predict_x = action[0,0] * self.continuous_action_scale
        predict_y = action[0,1] * self.continuous_action_scale
        predict_z = action[0,2] * self.continuous_action_scale

        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_wls_igst'] # WLS结果作为观测
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_wls_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_wls_igst']
            kf_x = self.kf_realtime.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_KF_realtime'] # 实时KF作为baselime
            kf_y = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_KF_realtime']
            kf_z = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_KF_realtime']
            v_x = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls_igst']
            v_y = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls_igst']
            v_z = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num-1), 'ecefZ']
        # modified for RLKF
        # rl_x = obs_x + predict_x
        # rl_y = obs_y + predict_y
        # rl_z = obs_z + predict_z
        x_wls = np.array([obs_x, obs_y, obs_z])
        v_wls = np.array([v_x, v_y, v_z])
        kf_x, kf_y, kf_z = self.RL4KFGSDC(x_wls,v_wls)
        rl_x = kf_x + predict_x
        rl_y = kf_y + predict_y
        rl_z = kf_z + predict_z

        self.baseline.loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['satnum']] = satnum
        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4KFGSDC(self, zs, us):  # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.kf_realtime.loc[self.current_step + (self.pos_num - 2), 'XEcefMeters_KF_realtime'],
                      self.kf_realtime.loc[self.current_step + (self.pos_num - 2), 'YEcefMeters_KF_realtime'],
                      self.kf_realtime.loc[self.current_step + (self.pos_num - 2), 'ZEcefMeters_KF_realtime']])  # State: 使用上一个时刻RL预测的位置
        I = np.eye(dim_x)

        # KF: Prediction step
        ## Estimated WLS velocity covariance
        Q = self.covxv['covv'][int(self.current_step + (self.pos_num - 1))]

        x = F @ x + us.T
        self.P = (F @ self.P) @ F.T + Q
        d = distance.mahalanobis(zs, H @ x, np.linalg.inv(self.P))
        # KF: Update step
        if d < sigma_mahalanobis:
            R = self.covxv['covx'][int(self.current_step + (self.pos_num - 1))]
            y = zs.T - H @ x
            S = (H @ self.P) @ H.T + R
            K = (self.P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            self.P = (I - (K @ H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * Q

        return x[0], x[1], x[2]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class GPSPosition_discrete_pos(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len):
        super(GPSPosition_discrete_pos, self).__init__()
        self.max_visible_sat=13
        self.traj_len = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)

        self.observation_space = spaces.Box(low=0, high=1, shape=(3, self.traj_len), dtype=np.float)

        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'full':
            self.tripIDlist = traj_full

        self.traj_type = traj_type

        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range
        # discrete action
        self.discrete_actionspace = discrete_actionspace
        self.action_space = spaces.Discrete(discrete_actionspace**3)
        self.action_scale = action_scale
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort=='randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum=random.randint(self.trajdata_range[0],self.trajdata_range[1])
        elif self.trajdata_sort=='sorted':
            self.tripIDnum = self.tripIDnum+1
            if self.tripIDnum>self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline=data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature=losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.datatime=self.baseline['UnixTimeMillis']
        self.kf_realtime = data_kf_realtime[self.tripIDlist[self.tripIDnum]].copy()
        self.timeend=self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values)-1, 'UnixTimeMillis']
        #normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']
        elif self.baseline_mod == 'wls':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_wls_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_wls_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss=gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat=satnum_df.loc[satnum_df.loc[:,self.tripIDlist[self.tripIDnum]]>0,'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs=self._next_observation()
        # must return in observation scale
        return obs#self.tripIDnum#, obs#, {}

    def _normalize_pos(self,state):
        state[0]=(state[0]-xecef_min) / (xecef_max - xecef_min)
        state[1]=(state[1]-yecef_min) / (yecef_max - yecef_min)
        state[2]=(state[2]-zecef_min) / (zecef_max - zecef_min)
        return state

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.traj_len-2), 'Z_RLpredict'].values
        ])
        if self.baseline_mod == 'bl':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_bl']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_bl']]],axis=1)
        elif self.baseline_mod == 'wls_igst':
            obs = np.append(obs,[[self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_KF_realtime']],
                             [self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'YEcefMeters_KF_realtime']],
                             [self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'ZEcefMeters_KF_realtime']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_wls_igst']],
                             [self.baseline.loc[self.current_step + (self.traj_len - 1), 'YEcefMeters_wls_igst']],
                             [self.baseline.loc[self.current_step + (self.traj_len - 1), 'ZEcefMeters_wls_igst']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf']]],axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs,[[self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf_igst']],
                                 [self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf_igst']]],axis=1)

        obs=self._normalize_pos(obs)

        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs

    def step(self, action):
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (self.traj_len) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.traj_len - 1), 'UnixTimeMillis']
        # action for new prediction
        predict_x=action % self.discrete_actionspace
        predict_yz=action // self.discrete_actionspace
        predict_z, predict_y = predict_yz // self.discrete_actionspace, predict_yz % self.discrete_actionspace
        predict_x = (predict_x - self.discrete_actionspace//2) * self.action_scale# RL调节范围 1e-6对应cm
        predict_y = (predict_y - self.discrete_actionspace//2) * self.action_scale
        predict_z = (predict_z - self.discrete_actionspace//2) * self.action_scale
        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':  # 用实时卡尔曼滤波
            obs_x = self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_KF_realtime']
            obs_y = self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'YEcefMeters_KF_realtime']
            obs_z = self.kf_realtime.loc[self.current_step + (self.traj_len - 1), 'ZEcefMeters_KF_realtime']
        elif self.baseline_mod == 'wls':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_wls_igst']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_wls_igst']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len - 1), 'XEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.traj_len-1), 'ecefZ']
        rl_x = obs_x + predict_x
        rl_y = obs_y + predict_y
        rl_z = obs_z + predict_z
        self.baseline.loc[self.current_step + (self.traj_len-1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.traj_len-1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.traj_len-1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.traj_len-1), ['Z_RLpredict']] = rl_z
        # reward function
        if self.reward_setting=='RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0#*1e5
        elif self.reward_setting=='RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2))*1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2))*1e0
        if step_print:
            print(f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                  f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum':self.tripIDnum, 'current_step':self.current_step, 'baseline':self.baseline} #self.info#, {}# , 'data_truth_dic':data_truth_dic

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')