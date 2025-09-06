import pandas as pd
#from env.GSDC_2022_LOSPOS import *
from envRLKF.GSDC_2022_LOSPOS_KF import * # RL环境
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
# from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO
from model.ppo import PPO
from model.ppo_recurrent_ATF1_AKF import RecurrentPPO
from env.env_param import *
from funcs.utilis import *
from funcs.PPO_SR import *
from model.model_ATF_KF import *
from collections import deque
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
training_stepnum = 35000 # urban: 20000 highway: 35000
# parameter settings
learning_rate_list = [8e-5] #[5e-4,1e-4 8e-5,3e-5]
postraj_num_list = [5] # urban: 10 highway: 5
# traj type: full 79 urban 32 highway 47 losangel 21 bayarea 58
alltraj = False # use all traj data for training and testing
triptype = 'highway' # highway urban
if alltraj:
    traj_type_target_train = [0, 0.7]  # 轨迹数据的比例
    traj_type_target_test = [0.7, 1]
else:
    traj_type_target_train = [0, 1]  # 轨迹数据的比例
    traj_type_target_test = [0, 1]

trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='wls_igst' # baseline方法 wls_igst kf_igst
# around 37.52321	-122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100 #最大动作的范围
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace=7
action_scale = 3e-1 # scale in meters
"""
define parameter for RLKF
envmode: 
1. losconvR_onlyR: use gnss and covR features to learn only Q
2. losposconvQ_onlyQ: use pos and covQ features to learn only Q
3. losposconvR_onlyR: use los pos and covR features to learn only R
4. losposcovR_onlyRNallcorrect: use los pos convR to learn R and correct in the end (or obs)
5. losposcovR_RNobsendcorrect: use los pos covR to learn R, correct obs, and correct in the end 
6. poscovQ_onlyQNallcorrect: use pos convQ to learn Q and correct in the end (or vel)
7. losposcovQ_onlyQNallcorrect: use los pos convQ to learn Q and correct in the end (or vel)
8. losposcovQR_QRNallcorrect: use los pos covQR to learn Q and R and correct
"""
envmod = 'losposcovR_onlyRNallcorrect'
noise_scale_dic = {'process':1e-9,'measurement':1e-9} # {'process':5e-6,'measurement':0.01}
conv_corr = 'conv_corr_1' # conv_corr_1 conv_corr_2
continuous_action_scale=40e-2 # 测量噪声估计 动作尺度
continuous_Vaction_scale=20e-3 # 过程噪声估计 动作尺度
allcorrect = False # if correct in the end or correct in obs
if allcorrect:
    continuous_action_scale=20e-1
network_unit = 64
net_archppo = [network_unit, network_unit]
ent_coef = 0.0
detail = f'{trajdata_sort}_{conv_corr}' # _reversetraj

# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom','continuous_lstmATF1']
custom_lists=['ppo_discrete','ppo']
networkmod='continuous_lstmATF1'
# select environment type
envlists=['latlon','ned','ecef','los','lospos',' losllAcat']
# recording parameters
running_date = 'RL4KF_halftest' # RL4KF_240303 RL4KF_halftrain RL4KF_halftest
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
elif triptype == 'GooglePixel5':
    tripIDlist = traj_GooglePixel5
    moreteststypelist = ['GooglePixel5']
    net_archppo = [network_unit, network_unit]
elif triptype == 'full':
    tripIDlist = traj_full
    moreteststypelist = ['full']
    #net_archppo = [network_unit, network_unit, network_unit]

# path for testing
posnum_test = 10
onlytesting = True
testdate = 'RL4KF_halftest'
if onlytesting:
    model_summary = True # 打印模型结构
    model_basefolder = 'source=urban_1_losposcovR_onlyRNallcorrect_wls_igst_continuous_lstmATF1'
    model_basefolder=f'{dir_path}/records_values/{testdate}/{model_basefolder}'
    model_folderlist=os.listdir(model_basefolder)
    model_folderlist.sort()
    # model_folderlist = ['lr=8e-05_pos=5_QS=1e-09_RS=1e-07_XAS=0.2_randint_conv_corr_2_reversetraj'] # only for testing

if alltraj:
    ratio = 1
    trajdata_range = [0, len(tripIDlist) - 1]
else:
    ratio = 0.5
    trajdata_range= [0,int(np.ceil(len(tripIDlist)*ratio))] # train with half of data

# trajnum_test = 45
# trajdata_range = [trajnum_test,trajnum_test]
# detail = f'trajnum={trajnum_test}'

if networkmod in discrete_lists:
    print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
elif networkmod in continuous_lists:
    print(f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')

if onlytesting == False:
    for learning_rate in learning_rate_list:
        for posnum in postraj_num_list:
            QS,RS = noise_scale_dic['process'],noise_scale_dic['measurement']
            tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{ratio}_{envmod}_{baseline_mod}_{networkmod}/' \
                              f'lr={learning_rate}_pos={posnum}_QS={QS}_RS={RS}_XAS={continuous_action_scale}_{detail}'
            if allcorrect==True:
                tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{envmod}_{baseline_mod}_{networkmod}/' \
                                  f'lr={learning_rate}_pos={posnum}_QS={QS}_RS={RS}_XAS={continuous_action_scale}_{detail}'
            if envmod == 'lospos_convQR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL

            elif envmod == 'losposconvR_onlyR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyR(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_losposcovR  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod)

            elif envmod == 'losconvR_onlyR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_los_convQR_onlyR(trajdata_range,traj_type_target_train, triptype, continuous_action_scale,
                                                                                      continuous_actionspace,  reward_setting, trajdata_sort, baseline_mod, posnum,noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_loscovR  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                policy_kwargs = dict(features_extractor_class=encoder, features_extractor_kwargs=dict(features_dim=features_dim_R + features_dim_gnss),ATF_trig=networkmod)

            elif envmod == 'losposconvQ_onlyQ':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(trajdata_range,traj_type_target_train, triptype, continuous_action_scale,
                                                                                      continuous_actionspace,  reward_setting, trajdata_sort, baseline_mod, posnum,noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_losposcovQ  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder, features_extractor_kwargs=dict(features_dim=features_dim_gnss+features_dim_Q + features_dim_pos),
                                     ATF_trig=networkmod,net_arch=net_arch)

            elif envmod == 'losposcovR_onlyRNallcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL_losposcovR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'poscovQ_onlyQNallcorrect' or envmod == 'losposcovQ_onlyQNallcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_Vaction_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                encoder = CustomATF1_AKFRL_poscovQ
                policy_dim = features_dim_pos + features_dim_Q
                if envmod == 'losposcovQ_onlyQNallcorrect':
                    tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{envmod}_{baseline_mod}_{networkmod}/' \
                                      f'lr={learning_rate}_pos={posnum}_QS={QS}_VAS={continuous_Vaction_scale}_{detail}'
                    encoder = CustomATF1_AKFRL_losposcovQ
                    policy_dim = features_dim_pos + features_dim_Q + features_dim_gnss
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'losposcovQR_QRNallcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_Q
                net_arch = net_archppo
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'losposcovR_RNobsendcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_RNobsendcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL_losposcovR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod)

            elif envmod == 'lospos':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1

            model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef)
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

            recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=True)

            # more tests
            if moretests:
                for testtype in moreteststypelist:
                    print(f'more test for {testtype} env begin here')
                    if testtype == 'highway':
                        tripIDlist_test = traj_highway
                    elif testtype == 'urban':
                        tripIDlist_test = traj_urban
                    elif testtype == 'full':
                        tripIDlist_test = traj_full
                    elif testtype == 'GooglePixel5':
                        tripIDlist_test = traj_GooglePixel5

                    if alltraj:
                        more_test_trajrange = [0, len(tripIDlist) - 1]
                    else:
                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*ratio))+1, len(tripIDlist_test) - 1]

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
                            if envmod == 'lospos_convQR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losposconvR_onlyR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losconvR_onlyR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_los_convQR_onlyR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losposconvQ_onlyQ':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])

                            elif envmod == 'losposcovR_onlyRNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovQ_onlyQNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Vaction_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovQR_QRNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovR_RNobsendcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_RNobsendcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'lospos':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(test_trajdata_range,traj_type, testtype,continuous_action_scale,continuous_actionspace,reward_setting,
                                                                                                    trajdata_sort,baseline_mod,posnum,noise_scale_dic,conv_corr)])

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
        #record model
        # if networkmod in model_folder:
        model_sepfolderlist=os.listdir(f'{model_basefolder}/{model_folder}') # PPO_1
        model_sepfolderlist.sort()
        # model_sepfolderlist=['RecurrentPPO_1','RecurrentPPO_2']

        for model_sepfolder in model_sepfolderlist:
            process_trig = False
            if ('csv' not in model_sepfolder) and ('txt' not in model_sepfolder):
                model_filelist=os.listdir(f'{model_basefolder}/{model_folder}/{model_sepfolder}')
                model_filelist.sort()
                for model_file in model_filelist:
                    if networkmod in model_file:
                        model_filename=model_file
                        process_trig = True
                        break
                    else:
                        process_trig = False

            if process_trig:
                model_loggerdir=f'{model_basefolder}/{model_folder}/{model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{model_loggerdir}, {t}')
                model_filenamepath=f'{model_loggerdir}/{model_filename}'
                if networkmod=='discrete_A2C':
                    model = A2C.load(model_filenamepath)
                elif networkmod in {'continuous_lstmATF1'}:
                    if envmod == 'losposconvQ_onlyQ':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(trajdata_range,traj_type_target_train,triptype,continuous_action_scale,
                                                               continuous_actionspace,reward_setting, trajdata_sort,baseline_mod, posnum_test,noise_scale_dic, conv_corr)])

                        encoder = CustomATF1_AKFRL_losposcovQ  # CustomATF1_AKFRL_measurementR
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder, features_extractor_kwargs=dict(
                            features_dim=features_dim_gnss + features_dim_Q + features_dim_pos),ATF_trig=networkmod, net_arch=net_arch)

                    elif envmod == 'losposcovR_onlyRNallcorrect':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_action_scale,
                                                                continuous_actionspace,reward_setting,trajdata_sort,baseline_mod, posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                        encoder = CustomATF1_AKFRL_losposcovR
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,
                                             features_extractor_kwargs=dict(features_dim=policy_dim),
                                             ATF_trig=networkmod, net_arch=net_arch)

                    elif envmod == 'poscovQ_onlyQNallcorrect' or envmod == 'losposcovQ_onlyQNallcorrect':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range,traj_type_target_train,triptype,
                                                              continuous_Vaction_scale,continuous_actionspace,reward_setting,
                                                              trajdata_sort,baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])

                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        encoder = CustomATF1_AKFRL_poscovQ
                        policy_dim = features_dim_pos + features_dim_Q
                        if envmod == 'losposcovQ_onlyQNallcorrect':
                            encoder = CustomATF1_AKFRL_losposcovQ
                            policy_dim = features_dim_pos + features_dim_Q + features_dim_gnss
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,
                                             features_extractor_kwargs=dict(features_dim=policy_dim),
                                             ATF_trig=networkmod, net_arch=net_arch)


                    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs,)
                    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                    model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention1.attwts.weight.squeeze())
                    model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention2.attwts.weight.squeeze())
                    model.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention4.attwts.weight.squeeze())
                    model.load(model_filenamepath,env=env)
                    if model_summary:
                        print(model.policy)
                        # 获取模型的参数
                        params = list(model.policy.parameters())
                        # 计算参数的总大小
                        total_params = sum(p.numel() for p in params)
                        print(f"模型的总参数数量为：{total_params}")
                        # 将模型保存到文件
                        torch.save(model.policy.state_dict(), 'model.pth')


                elif networkmod in {'continuous_lstm'}:
                    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs,)
                    model = RecurrentPPO.load(model_filenamepath,env=env)

                # more tests
                if moretests:
                    for testtype in moreteststypelist:
                        print(f'more test for {testtype} env begin here')
                        if testtype == 'highway':
                            tripIDlist_test = traj_highway
                        elif testtype == 'urban':
                            tripIDlist_test = traj_urban
                        elif testtype == 'full':
                            tripIDlist_test = traj_full

                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * 0.5)) + 1, len(tripIDlist_test) - 1]
                        if testtype == triptype:
                            traj_type = traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            if networkmod in discrete_lists:
                                env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type,action_scale,discrete_actionspace,
                                                                                       reward_setting, trajdata_sort, baseline_mod)])
                            elif networkmod in continuous_lists:
                                if envmod == 'lospos_convQR':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(test_trajdata_range, traj_type, testtype,
                                                          continuous_action_scale,continuous_actionspace, reward_setting, trajdata_sort,
                                                           baseline_mod,posnum_test, noise_scale_dic,conv_corr)])
                                elif envmod == 'losposconvR_onlyR':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyR(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic, conv_corr)])
                                elif envmod == 'losconvR_onlyR':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_los_convQR_onlyR(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic, conv_corr)])
                                elif envmod == 'losposconvQ_onlyQ':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic, conv_corr)])
                                elif envmod == 'losposcovR_onlyRNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic,
                                        conv_corr, allcorrect=allcorrect)])
                                elif envmod == 'losposcovQ_onlyQNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic,conv_corr, allcorrect=allcorrect)])
                                elif envmod == 'losposcovQR_QRNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic,
                                        conv_corr, allcorrect=allcorrect)])
                                elif envmod == 'losposcovR_RNobsendcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_RNobsendcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,noise_scale_dic,conv_corr, allcorrect=allcorrect)])
                                elif envmod == 'lospos':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(test_trajdata_range,traj_type, testtype, continuous_action_scale,
                                                               continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum_test,noise_scale_dic,conv_corr)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                action, _states = model.predict(obs)
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 1 or iter % 10 == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif rewards == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]],tripIDlist_test, logdirname, baseline_mod, traj_record=True)

    print('only test finish!')