import pandas as pd
#from env.GSDC_2022_LOSPOS import *
from envRLKF.GSDC_2022_LOSPOS_KF_V4multi import * # RL环境
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from model.ppo import PPO
from model.ppo_recurrent_ATF1_AKF_multi import RecurrentPPO
#from model.ppo_recurrent_ATF1_AKF_multi_onlyQoR import RecurrentPPO
from env.env_param import *
from funcs.utilis import *
from funcs.PPO_SR import *
from model.model_ATF_KF import *
from collections import deque
import time
import os
import torch.optim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
training_stepnum = 35000 # 20000
# parameter settings
learning_rate_list = [8e-5] #[5e-4,1e-4 8e-5,3e-5]
# traj type: full 79 urban 32 highway 47 losangel 21 bayarea 58
alltraj = False # use all traj data for training and testin
triptype = 'highway' # highway or urban
if alltraj:
    traj_type_target_train = [0, 0.7]  # 轨迹数据的比例
    traj_type_target_test = [0.7, 1]
else:
    traj_type_target_train = [0, 1]  # 轨迹数据的比例
    traj_type_target_test = [0, 1]

trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='wls_igst' # baseline方法
# around 37.52321	-122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100 #最大动作的范围
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace=7
action_scale = 5e-1 # scale in meters
"""
define parameter for RLKF
envmod: 
1. losposconvQR_onlyQR: use los pos covQR to only learn Q and R
2. losposcovR_onlyRNallcorrect: use los pos convR to learn R and correct in the end (or obs)
3. losposconvQR_V4: use los pos covQR to learn Q R and noise prediction
4. losposconvQR_QRcorrect_V4: use lospos covQ or R to learn Q R and noise prediction
5. losposconvQR_QRcorrect_fullobs_V4: use lospos covQ and R to learn Q R and noise prediction
"""
envmod = 'losposconvQR_QRcorrect_fullobs_V4' # losposconvQRre_V4 losposconvQR_V4 losposconvQRscale_V4
continuous_Xaction_scale= 40e-2 # 测量动作尺度
continuous_Vaction_scale= 20e-5 # 过程动作尺度
noise_scale_dic = {'process':1e-10,'measurement':1e-7} # {'process':5e-6,'measurement':0.01}
if triptype == 'highway':
    postraj_num_list = [5]  # [10,20,30,40]
    conv_corr = 'conv_corr_1' # conv_corr_1
    seed = 0
elif triptype == 'urban':
    postraj_num_list = [10]  # [10,20,30,40]
    conv_corr = 'conv_corr_2'  # conv_corr_1
allcorrect = False # it means correct in the end
network_unit = 64
detail = f'{conv_corr}_35000itr_noenf_seed0' # Vs={continuous_Vaction_scale}_maxmin

# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom','continuous_lstmATF1']
custom_lists=['ppo_discrete','ppo']
networkmod='continuous_lstmATF1'
# select environment type
envlists=['latlon','ned','ecef','los','lospos','losllAcat']
# recording parameters
running_date = 'RL4KF_halftest' # RL4KF_240303 RL4KF_halftrain
reward_setting='RMSEadv_kf' # 'RMSE' ‘RMSEadv' 'RMSEadv_kf'
# data cata: KF: LatitudeDegrees; robust WLS: LatitudeDegrees_wls; standard WLS: LatitudeDegrees_bl
# parameters for customized ppo
# test settings
moretests=True #True False

if triptype == 'highway':
    tripIDlist = traj_highway
    moreteststypelist = ['highway']
elif triptype == 'urban':
    tripIDlist = traj_urban
    moreteststypelist = ['urban']

# path for testing
posnum_test = 5
onlytesting = False
param_record = True
testdate = 'RL4KF_halftest'
if onlytesting:
    model_basefolder = 'source=highway_1_losposconvQR_QRcorrect_fullobs_V4/'
    model_basefolder=f'{dir_path}/records_values/{testdate}/{model_basefolder}'
    model_folderlist=os.listdir(model_basefolder)
    model_folderlist.sort()
# model_folderlist = ['continuous_lstmATF1_wls_igst_lr=5e-05_pos=10_QS=1e-09_RS=1e-05_XAS=2.0_VAS=0.0002'] # only for testing

if alltraj:
    ratio = 1
    trajdata_range = [0, len(tripIDlist) - 1]
else:
    ratio = 0.5
    trajdata_range= [0,int(np.ceil(len(tripIDlist)*ratio))] # train with half of data

if networkmod in discrete_lists:
    print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
elif networkmod in continuous_lists:
    print(f'Action scale {continuous_Xaction_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')

if onlytesting == False:
    for learning_rate in learning_rate_list:
        for posnum in postraj_num_list:
            QS,RS = noise_scale_dic['process'],noise_scale_dic['measurement']
            tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{envmod}/' \
                              f'{networkmod}_{baseline_mod}_lr={learning_rate}_pos={posnum}_QS={QS}_RS={RS}_XAS={continuous_Xaction_scale}_VAS={continuous_Vaction_scale}_{detail}'
            if envmod == 'losposconvQR_V4':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(trajdata_range, traj_type_target_train, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                                continuous_actionspace,reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                Q_encoder = CustomATF1_AKFRL_processQ
                R_encoder = CustomATF1_AKFRL_measurementR

            elif envmod == 'losposconvQRre_V4':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQRre(trajdata_range, traj_type_target_train, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                                continuous_actionspace,reward_setting,trajdata_sort,baseline_mod, posnum,noise_scale_dic, conv_corr)])
                Q_encoder = CustomATF1_AKFRL_processQ
                R_encoder = CustomATF1_AKFRL_measurementR
            elif envmod == 'losposconvQRscale_V4':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQRscale(trajdata_range,traj_type_target_train, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                            continuous_actionspace,reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic, conv_corr)])
                Q_encoder = CustomATF1_AKFRL_processQ
                R_encoder = CustomATF1_AKFRL_measurementR
            elif envmod == 'losposconvQR_onlyQR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_losposconvQR_onlyQR(trajdata_range,traj_type_target_train, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                            continuous_actionspace,reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic, conv_corr)])
                Q_encoder = CustomATF1_AKFRL_losposcovQ
                R_encoder = CustomATF1_AKFRL_losposcovR

                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                net_arch = [network_unit, network_unit]
                policy_kwargs_Q = dict(features_extractor_class=Q_encoder,  # CustomCNN CustomMLP
                    features_extractor_kwargs=dict(features_dim=features_dim_gnss+features_dim_Q+features_dim_pos),
                    ATF_trig=networkmod, net_arch=net_arch)
                policy_kwargs_R = dict(features_extractor_class=R_encoder,  # CustomCNN CustomMLP
                    features_extractor_kwargs=dict(features_dim=features_dim_gnss+features_dim_pos+features_dim_R),
                    ATF_trig=networkmod, net_arch=net_arch)
                policy_kwargs_dic = {'Q_policy': policy_kwargs_Q, 'R_policy': policy_kwargs_R}

            elif envmod == 'losposconvQR_QRcorrect_V4' or envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(trajdata_range,traj_type_target_train, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                            continuous_actionspace,reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic, conv_corr,allcorrect=allcorrect)])
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                net_arch = [network_unit, network_unit]
                Q_encoder = CustomATF1_AKFRL_losposcovQ
                R_encoder = CustomATF1_AKFRL_losposcovR
                dim_Q = features_dim_Q+features_dim_pos+features_dim_gnss
                dim_R = features_dim_gnss+features_dim_pos+features_dim_R
                if envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                    Q_encoder = CustomATF1_AKFRL
                    R_encoder = CustomATF1_AKFRL
                    dim_Q = features_dim_Q + features_dim_pos + features_dim_gnss + features_dim_R
                    dim_R = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_Q
                policy_kwargs_Q = dict(features_extractor_class=Q_encoder,  # CustomCNN CustomMLP
                    features_extractor_kwargs=dict(features_dim=dim_Q),ATF_trig=networkmod, net_arch=net_arch)
                policy_kwargs_R = dict(features_extractor_class=R_encoder,  # CustomCNN CustomMLP
                    features_extractor_kwargs=dict(features_dim=dim_R),ATF_trig=networkmod, net_arch=net_arch)
                policy_kwargs_dic = {'Q_policy': policy_kwargs_Q, 'R_policy': policy_kwargs_R}

            elif envmod == 'losposcovR_onlyRNallcorrectV4':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_Xaction_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])

                R_encoder = CustomATF1_AKFRL_measurementR
                correct_encoder = CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                feature_dim = features_dim_R + features_dim_pos + features_dim_gnss
                policy_kwargs_R = dict(features_extractor_class=R_encoder, features_extractor_kwargs=dict(features_dim=feature_dim),ATF_trig=networkmod)
                policy_kwargs_correct = dict(features_extractor_class=correct_encoder, features_extractor_kwargs=dict(features_dim=feature_dim), ATF_trig=networkmod)
                policy_kwargs_dic = {'R_policy': policy_kwargs_R, 'correct_policy': policy_kwargs_correct}

            model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, policy_kwargs_dic=policy_kwargs_dic, tensorboard_log=tensorboard_log,
                                 learning_rate=learning_rate, seed=seed) # ent_coef=0.01,  urban的时候有加这个吗
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
                                            f"_{continuous_Xaction_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                            f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")

            recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=True)

            # more tests
            if moretests:
                for testtype in moreteststypelist:
                    print(f'more test for {testtype} env begin here')
                    if testtype == 'highway':
                        tripIDlist_test = traj_highway
                    elif testtype == 'urban':
                        tripIDlist_test = traj_urban

                    more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*0.5))+1, len(tripIDlist_test) - 1]
                    if testtype == triptype:
                        traj_type = traj_type_target_test  # 独立同分布测试
                    else:
                        traj_type = [0, 1]  # 域外分布测试范围

                    test_trajlist=range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                    for test_traj in test_trajlist:
                        test_trajdata_range = [test_traj, test_traj]
                        if networkmod in discrete_lists:
                            env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type, action_scale, discrete_actionspace,
                                                                               reward_setting,trajdata_sort,baseline_mod)])
                        elif networkmod in continuous_lists:
                            if envmod == 'losposconvQR_V4':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale, continuous_Vaction_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])

                            elif envmod == 'losposconvQRre_V4':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQRre(test_trajdata_range,traj_type, testtype, continuous_Xaction_scale, continuous_Vaction_scale,continuous_actionspace,
                                                                                                reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic,conv_corr)])
                            elif envmod == 'losposconvQRscale_V4':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQRscale(test_trajdata_range,traj_type, testtype, continuous_Xaction_scale, continuous_Vaction_scale,continuous_actionspace,
                                                                                                reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic,conv_corr)])
                            elif envmod == 'losposconvQR_onlyQR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_losposconvQR_onlyQR(test_trajdata_range,traj_type, testtype, continuous_Xaction_scale, continuous_Vaction_scale,continuous_actionspace,
                                                                                                reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic,conv_corr)])
                            elif envmod == 'losposconvQR_QRcorrect_V4' or envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range,traj_type, testtype, continuous_Xaction_scale, continuous_Vaction_scale,continuous_actionspace,
                                                                                                reward_setting, trajdata_sort,baseline_mod, posnum,noise_scale_dic,conv_corr)])

                            elif envmod == 'losposcovR_onlyRNallcorrectV4':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(test_trajdata_range, traj_type_target_train, triptype, continuous_Xaction_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])

                        obs = env.reset()
                        maxiter = 100000
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

                    logdirname=model.logger.dir + f'/testmore_{testtype}_'
                    recording_results_ecef_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,baseline_mod,traj_record=True)

            cnt=1
    print('More Test for different phonetype finished.')

elif onlytesting:
    for model_folder in model_folderlist:
        if f'pos={posnum_test}' not in model_folder:
            continue
        # if f'QS=1e-11' not in model_folder:
        #     continue
        #record model
        # if networkmod in model_folder:
        model_sepfolderlist=os.listdir(f'{model_basefolder}{model_folder}') # PPO_1
        model_sepfolderlist.sort()
        # model_sepfolderlist=['RecurrentPPO_1','RecurrentPPO_2']

        for model_sepfolder in model_sepfolderlist:
            process_trig = False
            if ('csv' not in model_sepfolder) and ('txt' not in model_sepfolder):
                model_filelist=os.listdir(f'{model_basefolder}{model_folder}/{model_sepfolder}')
                model_filelist.sort()
                for model_file in model_filelist:
                    if networkmod in model_file:
                        model_filename=model_file
                        process_trig = True
                        break
                    else:
                        process_trig = False

            if process_trig:
                model_loggerdir=f'{model_basefolder}{model_folder}/{model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{model_loggerdir}, {t}')
                model_filenamepath=f'{model_loggerdir}/{model_filename}'
                if networkmod in {'continuous_lstmATF1'}:
                    if envmod == 'losposcovR_onlyRNallcorrectV4':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_Xaction_scale,
                                                                                                         continuous_actionspace,reward_setting,trajdata_sort,baseline_mod,
                                                                                                         posnum_test, noise_scale_dic,conv_corr,allcorrect=allcorrect)])

                        R_encoder = CustomATF1_AKFRL_measurementR
                        correct_encoder = CustomATF1_AKFRL_measurementR
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        feature_dim = features_dim_R + features_dim_pos + features_dim_gnss
                        policy_kwargs_R = dict(features_extractor_class=R_encoder,
                                               features_extractor_kwargs=dict(features_dim=feature_dim),
                                               ATF_trig=networkmod)
                        policy_kwargs_correct = dict(features_extractor_class=correct_encoder,
                                                     features_extractor_kwargs=dict(features_dim=feature_dim),
                                                     ATF_trig=networkmod)
                        policy_kwargs_dic = {'R_policy': policy_kwargs_R, 'correct_policy': policy_kwargs_correct}
                        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs_dic=policy_kwargs_dic)
                        model.policy_correct.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_correct.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_correct.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_correct.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_correct.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_correct.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention3.attwts.weight.squeeze())

                    elif envmod == 'losposconvQR_onlyQR':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_losposconvQR_onlyQR(trajdata_range,traj_type_target_train,triptype,continuous_Xaction_scale,
                                                                                              continuous_Vaction_scale,continuous_actionspace, reward_setting,trajdata_sort,
                                                                                              baseline_mod, posnum_test,noise_scale_dic,conv_corr)])
                        Q_encoder = CustomATF1_AKFRL_losposcovQ
                        R_encoder = CustomATF1_AKFRL_losposcovR
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        policy_kwargs_Q = dict(features_extractor_class=Q_encoder,  # CustomCNN CustomMLP
                                               features_extractor_kwargs=dict(features_dim=features_dim_gnss + features_dim_Q + features_dim_pos),
                                               ATF_trig=networkmod)
                        policy_kwargs_R = dict(features_extractor_class=R_encoder,  # CustomCNN CustomMLP
                                               features_extractor_kwargs=dict(features_dim=features_dim_gnss + features_dim_pos + features_dim_R),
                                               ATF_trig=networkmod)
                        policy_kwargs_dic = {'Q_policy': policy_kwargs_Q, 'R_policy': policy_kwargs_R}
                        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs_dic=policy_kwargs_dic)
                        model.policy_Q.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention4.attwts.weight.squeeze())

                    elif envmod == 'losposconvQR_QRcorrect_V4' or envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_Xaction_scale,
                                                                                              continuous_Vaction_scale,continuous_actionspace, reward_setting,trajdata_sort,
                                                                                              baseline_mod, posnum_test,noise_scale_dic,conv_corr)])
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        net_arch = [network_unit, network_unit]
                        Q_encoder = CustomATF1_AKFRL_losposcovQ
                        R_encoder = CustomATF1_AKFRL_losposcovR
                        dim_Q = features_dim_Q + features_dim_pos + features_dim_gnss
                        dim_R = features_dim_gnss + features_dim_pos + features_dim_R
                        if envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                            Q_encoder = CustomATF1_AKFRL
                            R_encoder = CustomATF1_AKFRL
                            dim_Q = features_dim_Q + features_dim_pos + features_dim_gnss + features_dim_R
                            dim_R = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_Q
                        policy_kwargs_Q = dict(features_extractor_class=Q_encoder,  # CustomCNN CustomMLP
                                               features_extractor_kwargs=dict(features_dim=dim_Q), ATF_trig=networkmod,
                                               net_arch=net_arch)
                        policy_kwargs_R = dict(features_extractor_class=R_encoder,  # CustomCNN CustomMLP
                                               features_extractor_kwargs=dict(features_dim=dim_R), ATF_trig=networkmod,
                                               net_arch=net_arch)
                        policy_kwargs_dic = {'Q_policy': policy_kwargs_Q, 'R_policy': policy_kwargs_R}
                        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs_dic=policy_kwargs_dic)
                        model.policy_Q.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_Q.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_Q.features_extractor.attention4.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy_R.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy_R.features_extractor.attention4.attwts.weight.squeeze())

                    model.load(model_filenamepath,env=env)

                    if param_record:
                        print(model.policy_Q) # 打印模型结构
                        params = list(model.policy_Q.parameters())
                        # 计算参数的总大小
                        total_params = sum(p.numel() for p in params)
                        print(f"模型的总参数数量为：{total_params}")
                        # 将模型保存到文件
                        torch.save(model.policy_Q.state_dict(), 'model.pth')
                        # 查看保存文件的大小
                        file_size = os.path.getsize('model.pth')
                        print(f"模型的存储大小为：{file_size} 字节")

                # more tests
                if moretests:
                    for testtype in moreteststypelist:
                        print(f'more test for {testtype} env begin here')
                        if testtype == 'highway':
                            tripIDlist_test = traj_highway
                        elif testtype == 'urban':
                            tripIDlist_test = traj_urban

                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*0.5))+1, len(tripIDlist_test) - 1]
                        if testtype == triptype:
                            traj_type = traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            if networkmod in discrete_lists:
                                env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type,action_scale, discrete_actionspace,
                                                                                       reward_setting, trajdata_sort,baseline_mod)])
                            elif networkmod in continuous_lists:
                                if envmod == 'losposcovR_onlyRNallcorrectV4':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,noise_scale_dic,conv_corr, allcorrect=allcorrect)])
                                elif envmod == 'losposconvQR_onlyQR':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_losposconvQR_onlyQR(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale,
                                        continuous_Vaction_scale, continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum_test, noise_scale_dic,conv_corr)])
                                elif envmod == 'losposconvQR_QRcorrect_V4' or envmod == 'losposconvQR_QRcorrect_fullobs_V4':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale,
                                        continuous_Vaction_scale, continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum_test, noise_scale_dic,conv_corr)])

                            obs = env.reset()
                            maxiter = 100000
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

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]], tripIDlist_test, logdirname,baseline_mod, traj_record=True)

    print('only test finish!')
