import sys
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import torch
from gym import spaces
# from stable_baselines3.common.buffers import RolloutBuffer
from sb3_161.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

# from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_161.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
# from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_161.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
# from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from sb3_161.ppo_rec.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy, \
    TransformerRecurrentActorCriticPolicy, GNNRecurrentActorCriticPolicy, GNNspRecurrentActorCriticPolicy, \
    GNNspRecurrentActorCriticPolicy_DirA


def SKL_sparsity(z, beta):
    beta_hat = torch.mean(z, 0)
    '''
    Clipped KL divergence between target Exp(beta) and observed Exp(beta_hat)
    '''
    mask = (beta_hat >= beta)#.detach()
    beta_hat_masked = beta_hat[mask]
    return beta / beta_hat_masked - torch.log(beta / beta_hat_masked) - 1

class RecurrentPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
        "TransformerLstmPolicy": TransformerRecurrentActorCriticPolicy,
        'GNNLstmPolicy': GNNRecurrentActorCriticPolicy,
        'GNNspLstmPolicy': GNNspRecurrentActorCriticPolicy,
        'GNNspLstmPolicyDirA': GNNspRecurrentActorCriticPolicy_DirA,
    }

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states = None
        # ATF weight recording for attention weight fusion 20230316
        if 'ATF_trig' in self.policy_kwargs:
            self.ATF_trig=policy_kwargs['ATF_trig']
            policy_kwargs.pop('ATF_trig')
        # Sparse coding loss for critic net 20230310
        if 'regulation_method' in self.policy_kwargs:
            self.regulation_method=policy_kwargs['regulation_method']
            policy_kwargs.pop('regulation_method')
            if 'L1' in self.regulation_method or 'L0' in self.regulation_method or 'Hoyer' in self.regulation_method:
                if 'sp' in self.regulation_method:
                    self.lambda_a=policy_kwargs['regulation_param1']
                    policy_kwargs.pop('regulation_param1')
                    self.lambda_c=policy_kwargs['regulation_param2']
                    policy_kwargs.pop('regulation_param2')
                else:
                    self.lambda1=policy_kwargs['regulation_param1']
                    policy_kwargs.pop('regulation_param1')
                # self.critic_loss_func=my_loss_L1_w(policy_kwargs['regulation_param1'])
            elif 'Log' in self.regulation_method or 'SKL' in self.regulation_method:
                if 'sp' in self.regulation_method:
                    self.lambda_a=policy_kwargs['regulation_param1']
                    policy_kwargs.pop('regulation_param1')
                    self.lambda_c=policy_kwargs['regulation_param2']
                    policy_kwargs.pop('regulation_param2')
                    self.delta_a=policy_kwargs['regulation_param3']
                    policy_kwargs.pop('regulation_param3')
                    self.delta_c=policy_kwargs['regulation_param4']
                    policy_kwargs.pop('regulation_param4')
                else:
                    self.lambda1=policy_kwargs['regulation_param1']
                    policy_kwargs.pop('regulation_param1')
                    self.delta1=policy_kwargs['regulation_param3']
                    policy_kwargs.pop('regulation_param3')
            # elif 'Hoyer' in self.regulation_method:
            #     if 'sp' in self.regulation_method:
            #         self.lambda_a=policy_kwargs['regulation_param1']
            #         policy_kwargs.pop('regulation_param1')
            #         self.lambda_c=policy_kwargs['regulation_param2']
            #         policy_kwargs.pop('regulation_param2')
            #     else:
            #         self.lambda1=policy_kwargs['regulation_param1']
            #         policy_kwargs.pop('regulation_param1')

        if _init_setup_model:
            self._setup_model()
    # add for SR 230403
    def SKL_sparsity_abs(self, z, beta):
        beta_hat = torch.mean(th.abs(z), dim=0)
        '''
        Clipped KL divergence between target Exp(beta) and observed Exp(beta_hat)
        '''
        mask = (beta_hat >= beta)  # .detach()
        beta_hat_masked = beta_hat[mask]
        return beta / beta_hat_masked - torch.log(beta / beta_hat_masked) - 1

    def SKL_sparsity_absrelu(self, hidden_states, beta):
        # Stack the hidden states and calculate the mean along the first dimension
        mean_hidden_states = torch.mean(torch.abs(hidden_states), dim=0)

        # Use the torch.relu function to create a differentiable mask
        masked_hidden_states = torch.relu(mean_hidden_states - beta) + beta

        # Calculate the sparsity loss using the masked hidden states
        sparsity_loss = beta / masked_hidden_states - torch.log(beta / masked_hidden_states) - 1

        # Sum the sparsity loss values
        return (sparsity_loss)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = (
            RecurrentDictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RecurrentRolloutBuffer
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        # remove check for custom pporecurrent 20230406
        # if not isinstance(self.policy, RecurrentActorCriticPolicy):
        #     raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
            (
                th.zeros(single_hidden_state_shape).to(self.device),
                th.zeros(single_hidden_state_shape).to(self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "RecurrentPPO",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """

        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        return total_timesteps, callback

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        # savemat = True
        savemat = False
        if savemat:
            if len(self._last_obs)==2:
                obs_gnss_all=[]
                obs_pos_all=[]
            else:
                obs_all = []
            cat_feature_all=[]
            lstm_hidden_pi_all=[]
            lstm_hidden_vf_all=[]
            mlp_hidden_pi_all=[]
            mlp_hidden_vf_all=[]

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts).float().to(self.device)
                # obtain mlp hidden states with forward propagation modified on 20230327
                # actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)
                actions, values, log_probs, lstm_states, lstm_hidden_pi, lstm_hidden_vf,\
                    mlp_hidden_pi, mlp_hidden_vf, cat_features = self.policy.forward(obs_tensor, lstm_states, episode_starts)

            if savemat:
                if len(self._last_obs)==2:
                    obs_gnss_all.append(self._last_obs['gnss'])
                    obs_pos_all.append(self._last_obs['pos'])
                else:
                    obs_all = []
                cat_feature_all.append(cat_features.cpu().numpy())
                lstm_hidden_pi_all.append(lstm_hidden_pi.cpu().numpy())
                lstm_hidden_vf_all.append(lstm_hidden_vf.cpu().numpy())
                mlp_hidden_pi_all.append(mlp_hidden_pi.cpu().numpy())
                mlp_hidden_vf_all.append(mlp_hidden_vf.cpu().numpy())

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # recording rewards per step # remote edition 20221021
            self.logger.record("values/rewards per step", np.mean(rewards))

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :],
                            lstm_states.vf[1][:, idx : idx + 1, :],
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False]).float().to(self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            # can not find mask 221021 (no mask in standard ppo
            # # check all zero obs in buffer for graphlos 230306
            # if np.all(self._last_obs['gnss']==0):#.item():
            #     cnt=1
            # likely add lstm hidden states in original ppo_recurrent
            if not savemat:
                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                    n_steps,
                    lstm_hidden_pi, lstm_hidden_vf,
                    mlp_hidden_pi, mlp_hidden_vf,
                    lstm_states=self._last_lstm_states,
                )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        if savemat:
            traj_type = 'urban'
            model_sepfolder='RecurrentPPO_3'
            # traj_type='highway'
            timesteps = 30000
            # timesteps = 2000
            if len(self._last_obs)==2:
                all_obs_gnss=np.concatenate(obs_gnss_all)[:timesteps,:]
                all_obs_pos=np.concatenate(obs_pos_all)[:timesteps,:]
            else:
                obs_all = []
            all_cat_feature = np.concatenate(cat_feature_all)[:timesteps,:]
            all_lstm_hidden_pi = np.concatenate(lstm_hidden_pi_all)[:timesteps,:]
            all_lstm_hidden_vf = np.concatenate(lstm_hidden_vf_all)[:timesteps,:]
            all_mlp_hidden_pi = np.concatenate(mlp_hidden_pi_all)[:timesteps,:]
            all_mlp_hidden_vf = np.concatenate(mlp_hidden_vf_all)[:timesteps,:]
            all_dic={'all_obs_gnss': all_obs_gnss,
                    'all_obs_pos':all_obs_pos,
                    'all_cat_feature':all_cat_feature,
                    'all_lstm_hidden_pi':all_lstm_hidden_pi,
                    'all_lstm_hidden_vf':all_lstm_hidden_vf,
                    'all_mlp_hidden_pi':all_mlp_hidden_pi,
                    'all_mlp_hidden_vf':all_mlp_hidden_vf,
                    }
            from scipy.io import savemat
            tensorboard_log=self.tensorboard_log.replace('SC4','SC')
            savemat(f'{tensorboard_log}/feature_{traj_type}_{model_sepfolder}_{timesteps:0.0f}.mat', all_dic)


        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones).float().to(self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        values_batchsum=0
        values_pred_batchsum=0
        returns_batchsum=0
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data,lstm_hidden_pi_buffer,lstm_hidden_vf_buffer,mlp_hidden_pi_buffer,mlp_hidden_vf_buffer in self.rollout_buffer.get(self.batch_size):
                # # check data has all zeros for graphlos 230306 padding batches with zeros?
                # # self.rollout_buffer.get batch_size=128 from 128 buffer but get 160 data
                # # results in 32 all zeros
                # mask_x = th.any(rollout_data.observations['gnss'] != 0, dim=2)
                # mask_1 = th.sum(mask_x, axis=1)
                # sum(mask_1 == 0)
                # if not min(mask_1):
                #     allzerocnt=1
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, lstm_hidden_pi,lstm_hidden_vf,mlp_hidden_pi,mlp_hidden_vf  = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                # Sparse coding loss for critic net 20230310
                invert_op = getattr(self, "regulation_method", None)
                regularization_loss=th.tensor(0.0).to('cuda:0')
                if invert_op:
                    state_dict = self.policy.state_dict()
                    # weight l1
                    weight_vf_l1=0
                    weight_pi_l1=0
                    for name, param in self.policy.named_parameters():
                        if ('value' in name or 'critic' in name) and 'weight' in name:
                            weight_vf_l1 += torch.norm(param, 1)
                    for name, param in self.policy.named_parameters():
                        if ('policy' in name or 'actor' in name) and 'weight' in name:
                            weight_pi_l1+=torch.norm(param, 1)
                    if self.regulation_method=='L1w_c':
                        regularization_loss += self.lambda1 * weight_vf_l1
                    elif self.regulation_method=='L1w_a':
                        regularization_loss += self.lambda1 * weight_pi_l1
                    elif self.regulation_method=='L1w_ac':
                        regularization_loss += self.lambda1 * weight_vf_l1
                        regularization_loss += self.lambda1 * weight_pi_l1
                    elif self.regulation_method=='L1w_ac_sp':
                        regularization_loss += self.lambda_c * weight_vf_l1
                        regularization_loss += self.lambda_a * weight_pi_l1
                    elif self.regulation_method=='Hoyerw_c':
                        for name, param in self.policy.named_parameters():
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*torch.square(torch.sum(torch.abs(param))) / (torch.sum(torch.square(param))+1e-15)
                    elif self.regulation_method=='Hoyerw_a':
                        for name, param in self.policy.named_parameters():
                            if ('policy' in name or 'actor' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*torch.square(torch.sum(torch.abs(param))) / (torch.sum(torch.square(param))+1e-15)
                    elif self.regulation_method=='Hoyerw_ac_sp':
                        for name, param in self.policy.named_parameters():
                            if ('policy' in name or 'actor' in name) and 'weight' in name:
                                regularization_loss+=self.lambda_a*torch.square(torch.sum(torch.abs(param))) / (torch.sum(torch.square(param))+1e-15)
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda_c*torch.square(torch.sum(torch.abs(param))) / (torch.sum(torch.square(param))+1e-15)
                    elif self.regulation_method=='L0w_c': # L0 norm has no gradients, so not applicable
                        for name, param in self.policy.named_parameters():
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*th.count_nonzero(param)
                    elif self.regulation_method=='Logw_c':
                        for name, param in self.policy.named_parameters():
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*th.sum(th.log(1 + th.abs(param)/self.delta1))
                    elif self.regulation_method=='Logw_a':
                        for name, param in self.policy.named_parameters():
                            if ('policy' in name or 'actor' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*th.sum(th.log(1 + th.abs(param)/self.delta1))
                    elif self.regulation_method=='Logw_ac_sp':
                        for name, param in self.policy.named_parameters():
                            if ('policy' in name or 'actor' in name) and 'weight' in name:
                                regularization_loss+=self.lambda_a*th.sum(th.log(1 + th.abs(param)/self.delta_a))
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda_c*th.sum(th.log(1 + th.abs(param)/self.delta_c))
                    elif self.regulation_method=='SKLabsw_c':
                        for name, param in self.policy.named_parameters():
                            if ('value' in name or 'critic' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*th.sum(self.SKL_sparsity_abs(param,self.delta1))
                    elif self.regulation_method=='SKLabsw_a':
                        for name, param in self.policy.named_parameters():
                            if ('policy' in name or 'actor' in name) and 'weight' in name:
                                regularization_loss+=self.lambda1*th.sum(self.SKL_sparsity_abs(param,self.delta1))
                    elif self.regulation_method=='L1mh_c':
                        regularization_loss+=self.lambda1*th.sum(th.abs((mlp_hidden_vf)))
                    elif self.regulation_method=='Hoyermh_c':
                        regularization_loss+=self.lambda1*torch.square(th.sum(th.abs((mlp_hidden_vf)))) / (torch.sum(torch.square(mlp_hidden_vf))+1e-15)
                    elif self.regulation_method=='L1mh_a':
                        regularization_loss+=self.lambda1*th.sum(th.abs((mlp_hidden_pi)))
                    elif self.regulation_method=='L1mh_ac_sp':
                        regularization_loss+=self.lambda_c*th.sum(th.abs((mlp_hidden_vf)))
                        regularization_loss+=self.lambda_a*th.sum(th.abs((mlp_hidden_pi)))
                    elif self.regulation_method=='L1lh_c':
                        regularization_loss+=self.lambda1*th.sum(th.abs((lstm_hidden_vf)))
                    elif self.regulation_method=='L1lh_a':
                        regularization_loss+=self.lambda1*th.sum(th.abs((lstm_hidden_pi)))
                    elif self.regulation_method=='L1lh_ac_sp':
                        regularization_loss+=self.lambda_c*th.sum(th.abs((lstm_hidden_vf)))
                        regularization_loss+=self.lambda_a*th.sum(th.abs((lstm_hidden_pi)))
                    elif self.regulation_method=='SKLmh_c':
                        regularization_loss+=self.lambda1*SKL_sparsity(mlp_hidden_vf,self.delta1)
                    elif self.regulation_method=='SKLmh_a':
                        regularization_loss+=self.lambda1*SKL_sparsity(mlp_hidden_pi,self.delta1)
                    elif self.regulation_method=='SKLmh_ac_sp':
                        regularization_loss+=self.lambda_c*SKL_sparsity(mlp_hidden_vf,self.delta_c)
                        regularization_loss+=self.lambda_a*SKL_sparsity(mlp_hidden_pi,self.delta_a)
                    elif self.regulation_method=='SKLabsmh_c':
                        regularization_loss+=self.lambda1*th.sum(self.SKL_sparsity_abs(mlp_hidden_vf,self.delta1))
                    elif self.regulation_method=='SKLabsmh_a':
                        regularization_loss+=self.lambda1*th.sum(self.SKL_sparsity_abs(mlp_hidden_pi,self.delta1))
                    elif self.regulation_method=='SKLabsmh_ac_sp':
                        regularization_loss+=self.lambda_c*th.sum(self.SKL_sparsity_abs(mlp_hidden_vf,self.delta_c))
                        regularization_loss+=self.lambda_a*th.sum(self.SKL_sparsity_abs(mlp_hidden_pi,self.delta_a))
                    # weightloss=self.critic_loss_func()

                    # added SR recordings 20230329
                    self.logger.record('sparsity/regularization_loss', regularization_loss.item())
                    self.logger.record('sparsity/weight_vf_L1', weight_vf_l1.item())
                    self.logger.record('sparsity/weight_pi_L1', weight_pi_l1.item())
                self.logger.record('sparsity/mlp_hidden_vf_L1', th.sum(th.abs((mlp_hidden_vf))).item())
                self.logger.record('sparsity/mlp_hidden_pi_L1', th.sum(th.abs((mlp_hidden_pi))).item())
                self.logger.record('sparsity/lstm_hidden_vf_L1', th.sum(th.abs((lstm_hidden_vf))).item())
                self.logger.record('sparsity/lstm_hidden_pi_L1', th.sum(th.abs((lstm_hidden_pi))).item())

                # added recordings 20221021
                self.logger.record('values/perepoch_values', th.mean(values[mask]).item())
                self.logger.record('values/perepoch_values_pred', np.float(th.mean(values_pred[mask])))
                self.logger.record('values/perepoch_returns', th.mean(rollout_data.returns[mask]).item())
                values_batchsum+=th.mean(values).item()
                values_pred_batchsum+=th.mean(values_pred).item()
                returns_batchsum+=th.mean(rollout_data.returns).item()

                # ATF weight recording for attention weight fusion 20230316
                invert_op1 = getattr(self, "ATF_trig", None)
                if invert_op1:
                    state_dict = self.policy.state_dict()
                    for name in list(state_dict.keys()):
                        if 'attwts' in name:
                            self.logger.record(f'attwts/{name}', th.mean(state_dict[name]).item())

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                # likely not added in loss and add regularization_loss on 230330
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + regularization_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # added recordings 20221021
        self.logger.record('values/values', th.mean(values[mask]).item())
        self.logger.record('values/values_pred', np.float(th.mean(values_pred[mask])))
        self.logger.record('values/returns', th.mean(rollout_data.returns[mask]).item())
        self.logger.record('values/values_batchsum', values_batchsum)
        self.logger.record('values/values_pred_batchsum', values_pred_batchsum)
        self.logger.record('values/returns_batchsum', returns_batchsum)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "RecurrentPPO":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

# Sparse coding loss for critic net 20230310
class my_loss_L1_w(th.nn.Module):
    def __init__(self, reg_nn):
        super().__init__()
        # self.model = model
        self.reg_nn = reg_nn

    def forward(self, x, y):
        mse = th.mean(th.pow((x - y), 2))
        regularization_loss = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                regularization_loss += th.sum(th.abs(param))
        return mse + self.reg_nn * regularization_loss