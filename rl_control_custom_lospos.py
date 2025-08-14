import pandas as pd
#from env.GSDC_2022_LOSPOS import *
from env.GSDC_2022_LOS import *
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from model.ppo import PPO
from model.ppo_recurrent import RecurrentPPO
from env.env_param import *
from funcs.utilis import *
from funcs.PPO_SR import *
from collections import deque
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
training_stepnum=3e6#3000000#
# traj type: full 79 urban 32 highway 47 losangel 21 bayarea 58
traj_type='highway'
traj_ratio=0.5
trajdata_range=[0,7] # 训练范围
trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='kf_igst'
# around 37.52321	-122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100
continuous_action_scale=20e-1 #
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace=7
action_scale = 5e-1 # scale in meters
# training settings
learning_rate = 1e-6
# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom']
custom_lists=['ppo_discrete','ppo']
networkmod='continuous_lstm'
# select environment type
envlists=['latlon','ned','ecef','los','lospos']
envmod='los'
# recording parameters
running_date='1213'
tensorboard_log=f'./records_values/{running_date}/{networkmod}_{traj_type}_{envmod}_{baseline_mod}_{running_date}'
reward_setting='RMSEadv' # 'RMSE' ‘RMSEadv'
# data cata: KF: LatitudeDegrees; robust WLS: LatitudeDegrees_wls; standard WLS: LatitudeDegrees_bl
# parameters for customized ppo
# test settings
moretests=True #True False
more_test_trajrange='full'

traj_sum_df = pd.read_csv(f'{dir_path}/env/traj_summary.csv')
tripIDlist_full=traj_sum_df['tripId'].values.tolist()
traj_highway=traj_sum_df.loc[traj_sum_df['Type']=='highway']['tripId'].values.tolist()
traj_else=traj_sum_df.loc[traj_sum_df['Type']!='highway']['tripId'].values.tolist()
traj_losangel=[]
traj_bayarea=[]
for i in range(len(traj_sum_df)):
    if 'Los Angeles' in traj_sum_df.loc[i,'Descriptions']:
        traj_losangel.append(traj_sum_df.loc[i,'tripId'])
    else:
        traj_bayarea.append(traj_sum_df.loc[i, 'tripId'])
if traj_type == 'highway':
    tripIDlist = traj_highway
elif traj_type == 'full':
    tripIDlist = tripIDlist_full
elif traj_type == 'urban':
    tripIDlist = traj_else
elif traj_type == 'losangel':
    tripIDlist = traj_losangel
elif traj_type == 'bayarea':
    tripIDlist = traj_bayarea

trajdata_range=[0,int(np.ceil(len(tripIDlist)*traj_ratio))]
if more_test_trajrange == 'full':
    more_test_trajrange = [0, len(tripIDlist) - 1]

if networkmod in discrete_lists:
    print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
elif networkmod in continuous_lists:
    print(f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')
print(f'Learning rate: {learning_rate:8.2e}')
if envmod=='lospos':
    if networkmod=='discrete':
        env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(trajdata_range, traj_type, action_scale, discrete_actionspace,
                                                           reward_setting,trajdata_sort,baseline_mod)])
        obs = env.reset()
        # action = np.array(1)
        # obs, rewards, done, info = env.step(action)
        # model = PPO("MlpPolicy", env, verbose=2,learning_rate=learning_rate)
        # training_stepnum=10000
        model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
        # save initial params
        params_init=model.get_parameters()
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
    elif networkmod=='continuous_lstm':
        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                               reward_setting,trajdata_sort,baseline_mod)])
        # obs = env.reset()
        model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=2, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
        # action, _states = model.predict(obs)
        # obs, rewards, done, info = env.step(action)
        # save initial params
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)
    elif networkmod=='continuous':
        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                               reward_setting,trajdata_sort,baseline_mod)])
        obs = env.reset()
        model = PPO("MultiInputPolicy", env, verbose=2, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

elif envmod=='los':
    if networkmod=='continuous_lstm':
        env = DummyVecEnv([lambda: GPSPosition_continuous_los(trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                               reward_setting,trajdata_sort,baseline_mod)])
        # obs = env.reset()
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, tensorboard_log=tensorboard_log, learning_rate=learning_rate)
        # action, _states = model.predict(obs)
        # obs, rewards, done, info = env.step(action)
        # save initial params
        model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

#print and save training results
logdirname=model.logger.dir+'/train_'
# logdirname='./'
print('Training finished.')

#record model
# params=model.get_parameters()
if networkmod in discrete_lists:
    model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{discrete_actionspace}_{action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
elif networkmod in continuous_lists:
    model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
                                f"_{continuous_action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")

recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod)

# more tests
if moretests:
    test_trajlist=range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
    for test_traj in test_trajlist:
        test_trajdata_range = [test_traj, test_traj]
        if networkmod in discrete_lists:
            env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type, action_scale, discrete_actionspace,
                                                               reward_setting,trajdata_sort,baseline_mod)])
        elif networkmod in continuous_lists:
            env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(test_trajdata_range, traj_type, continuous_action_scale, continuous_actionspace,
                                                               reward_setting,trajdata_sort,baseline_mod)])
        obs = env.reset()
        maxiter = 10000
        for iter in range(maxiter):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            tmp = info[0]['tripIDnum']
            if iter <= 1 or iter % np.ceil(maxiter / 10) == 0:
                # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                print(f'Iter {iter}, traj {tmp} reward is {rewards}')
            elif done:
                print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                break

        pd_train = data_truth_dic[tripIDlist[int(info[0]['tripIDnum'])]]
        # pd_train=info[0]['baseline']
        if baseline_mod == 'bl':
            test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                    'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_bl', 'YEcefMeters_bl', 'ZEcefMeters_bl']]
        elif baseline_mod == 'wls':
            test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                    'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_wls', 'YEcefMeters_wls', 'ZEcefMeters_wls']]
        elif baseline_mod == 'kf':
            test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                    'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
        elif baseline_mod == 'kf_igst':
            test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                    'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']]
        test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[0], axis=1)
        test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[1], axis=1)
        test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
        test['count_rl_distance'] = test['rl_distance'].astype(float)
        test['count_or_distance'] = test['or_distance'].astype(float)
        print(test['error'].describe())
        print(test['count_rl_distance'].describe())
        print(test['count_or_distance'].describe())

    print('More Test finished.')
    logdirname=model.logger.dir+'/testmore_'
    recording_results_ecef(data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist,logdirname,baseline_mod)

cnt=1
# randnum=[]
# for i in range(168):
#     randnum.append(random.randint(0,168))
# if networkmod in discrete_lists:
#     predict_lat, predict_lng = action // discrete_actionspace, action % discrete_actionspace
#     predict_lat = (predict_lat - discrete_actionspace // 2) * action_scale  # RL调节范围 1e-6对应cm
#     predict_lng = (predict_lng - discrete_actionspace // 2) * action_scale
#     print(f'predict_lat: {predict_lat}, predict_lng: {predict_lng}')
#
#     action, _states = model.predict(obs)
#
#     predict_lat, predict_lng = action // discrete_actionspace, action % discrete_actionspace
#     predict_lat = (predict_lat - discrete_actionspace // 2) * action_scale  # RL调节范围 1e-6对应cm
#     predict_lng = (predict_lng - discrete_actionspace // 2) * action_scale
#     print(f'Another predict_lat: {predict_lat}, predict_lng: {predict_lng}')
# elif networkmod in continuous_lists:
#     print(f'action: {action}')
#     action, _states = model.predict(obs)
#     print(f'Another action: {action}')