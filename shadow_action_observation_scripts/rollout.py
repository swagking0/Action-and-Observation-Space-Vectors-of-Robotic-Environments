from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, datetime
from PIL import Image
from mujoco_py import MujocoException
import csv

from baselines.her.util import convert_episode_to_batch_major, store_args

fig = plt.figure("Observation Space Plotter",figsize=(16.5, 7.0))
fig.set_facecolor('#74dd93')
ax1 = plt.subplot2grid((8,10), (0,0),rowspan=5,colspan=10)
ax2 = plt.subplot2grid((8,10), (6,0),colspan=10)

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()
        self.rendder_and_save_png = True #ndrw


    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):

            self.reset_rollout(i)


    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """

        directory_plot = '../shadow-hand-obervation-plot/shadow-hand-obs-png/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + os.sep
        directory_env = '../shadow-hand-observation-env/shadow-hand-env-png/' + datetime.datetime.now().strftime("%m%d_%H%M%S") + os.sep
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []

        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        x_bar = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
        x_lab = ["WR-J1-qpos","WR-J0-qpos","FF-J3-qpos","FF-J2-qpos","FF-J1-qpos","FF-J0-qpos","MF-J3-qpos","MF-J2-qpos","MF-J1-qpos", "MF-J0-qpos", "RF-J3-qpos", "RF-J2-qpos", "RF-J1-qpos", "RF-J0-qpos", "LF-J4-qpos", "LF-J3-qpos", "LF-J2-qpos", "LF-J1-qpos", "LF-J0-qpos", "TH-J4-qpos", "TH-J3-qpos", "TH-J2-qpos", "TH-J1-qpos", "TH-J0-qpos", "WR-J1-qvel","WR-J0-qvel","FF-J3-qvel","FF-J2-qvel","FF-J1-qvel","FF-J0-qvel","MF-J3-qvel","MF-J2-qvel","MF-J1-qvel", "MF-J0-qvel", "RF-J3-qvel", "RF-J2-qvel", "RF-J1-qvel", "RF-J0-qvel", "LF-J4-qvel", "LF-J3-qvel", "LF-J2-qvel", "LF-J1-qvel", "LF-J0-qvel", "TH-J4-qvel", "TH-J3-qvel", "TH-J2-qvel", "TH-J1-qvel", "TH-J0-qvel","object_qvel-0","object_qvel-1","object_qvel-2","object_qvel-3","object_qvel-4","object_qvel-5", "achieved_goal-0", "achieved_goal-1", "achieved_goal-2", "achieved_goal-3", "achieved_goal-4", "achieved_goal-5", "achieved_goal-6"]

        # List set used for appending values from eposide
        observation_catcher = []
        observation_catcher_1 = []
        observation_catcher_2 = []
        observation_catcher_3 = []
        observation_catcher_4 = [] # this is the max append used for observation parameters qpos and qvel
        observation_catcher_5 = [] # this is the max append used for observation parameter object qvel
        observation_catcher_6 = [] # this is the max append used for observation parameter achieved goal

        # List set used for appending values from csv file
        observation_catcher_f0 = []
        observation_catcher_f1 = []
        observation_catcher_f2 = []
        observation_catcher_f3 = []
        observation_catcher_f4 = [] # this is the max append used for observation parameters qpos and qvel from csv file
        observation_catcher_f5 = [] # this is the max append used for observation parameter object qvel from csv file
        observation_catcher_f6 = [] # this is the max append used for observation parameter achieved goal from csv file
        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            #u_check = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # used to check the observation value changes --> replace in step with u_check instead of u[i]

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i]) #u[i] & u_check
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                    elif self.rendder_and_save_png: #ndrw
                        rgb_array = self.envs[i].render(mode='rgb_array')
                        im = Image.fromarray(rgb_array)
                        lov = im.crop((230,180,780,730)) # the crop setting needed to be changed as per the direction and the required parameters are present in resource file.
                        observation_catcher.append(o_new[i][0]) # use to append the requried set values from the observation vector (0-60)
                        observation_catcher_1.append(o_new[i][1])
                        observation_catcher_2.append(o_new[i][2])
                        observation_catcher_3.append(o_new[i][3])
                        observation_catcher_4.append(o_new[i][4])
                        observation_catcher_5.append(o_new[i][5])
                        observation_catcher_6.append(o_new[i][6])

                        #this place to read csv file which has all the observation space values of shadow-hand
                        with open('two_finger_ac_values/foo_0.csv', 'r') as readcsv:
                            plots = csv.reader(readcsv, delimiter=',')
                            for row in plots:
                                observation_catcher_f0.append(float(row[0])) # use to append the requried set values from the observation vector (0-60) csv file
                                observation_catcher_f1.append(float(row[1]))
                                observation_catcher_f2.append(float(row[2]))
                                observation_catcher_f3.append(float(row[3]))
                                observation_catcher_f4.append(float(row[4]))
                                observation_catcher_f5.append(float(row[5]))
                                observation_catcher_f6.append(float(row[6]))

                        ax1.clear()
                        ax1.axvline(len(observation_catcher)-1,ymin=-1,ymax=1,color='k',linestyle=':',linewidth=3) # marker line for each step
                        ax1.plot(observation_catcher_f0,color='xkcd:coral',linewidth=4,label="achieved_goal-0") # full curve for all steps
                        ax1.plot(observation_catcher_f1,color='xkcd:green',linewidth=4,label="achieved_goal-1")
                        ax1.plot(observation_catcher_f2,color='xkcd:goldenrod',linewidth=4,label="achieved_goal-2")
                        ax1.plot(observation_catcher_f3,color='xkcd:orchid',linewidth=4,label="achieved_goal-3")
                        ax1.plot(observation_catcher_f4,color='xkcd:azure',linewidth=4,label="achieved_goal-4")
                        ax1.plot(observation_catcher_f5,color='xkcd:orangered',linewidth=4,label="achieved_goal-5")
                        ax1.plot(observation_catcher_f6,color='xkcd:tan',linewidth=4,label="achieved_goal-6")
                        ax1.plot(observation_catcher,'o', color='xkcd:coral',markevery=[-1], markersize=10,markeredgecolor='k') # ball marker for each step
                        ax1.plot(observation_catcher_1,'o', color='xkcd:green',markevery=[-1], markersize=10,markeredgecolor='k')
                        ax1.plot(observation_catcher_2,'o',color='xkcd:goldenrod',markevery=[-1], markersize=10,markeredgecolor='k')
                        ax1.plot(observation_catcher_3,'o',color='xkcd:orchid',markevery=[-1], markersize=10,markeredgecolor='k')
                        ax1.plot(observation_catcher_4,'o',color='xkcd:azure',markevery=[-1], markersize=10,markeredgecolor='k')
                        ax1.plot(observation_catcher_5,'o',color='xkcd:orangered',markevery=[-1], markersize=10,markeredgecolor='k')
                        ax1.plot(observation_catcher_6,'o',color='xkcd:tan',markevery=[-1], markersize=10,markeredgecolor='k')

                        ax1.set_xlabel('Time-Step',fontsize=15)
                        ax1.set_ylabel('Observation-Values',fontsize=15)
                        ax1.set_title('Observation Vector Of The Shadow Hand (NN-Input)',fontsize=18,loc="left")
                        ax1.legend(loc = 'upper right',facecolor='#74dd93',frameon=False,fontsize='large', ncol=3, bbox_to_anchor=(1.03,1.27))
                        ax1.set_facecolor('#74dd93')
                        ax1.set_xlim(xmin=-1)
                        ax1.set_xlim(xmax=99)
                        #ax1.set_ylim(ymin=-1.05) # default value --> should be checked according the y min in observed value - hard coded
                        #ax1.set_ylim(ymax=1.1)  # default value --> should be checked according the y max in observed value - hard coded

                        ax2.clear()
                        barlist = ax2.bar(x_bar,color='xkcd:silver',width=0.6,height=0.025)
                        barlist[0].set_color('xkcd:coral')
                        barlist[1].set_color('xkcd:green')
                        barlist[2].set_color('xkcd:goldenrod')
                        barlist[3].set_color('xkcd:orchid')
                        barlist[4].set_color('xkcd:azure')
                        barlist[5].set_color('xkcd:orangered')
                        barlist[6].set_color('xkcd:tan')
                        ax2.set_yticklabels([])
                        ax2.set_xticks(x_bar)
                        ax2.set_xticklabels(x_lab,rotation=90,fontsize=11)
                        ax2.set_facecolor('#74dd93')
                        ax2.set_frame_on(False)
                        ax2.axes.get_yaxis().set_visible(False)
                        if not os.path.exists(directory_plot):
                            os.makedirs(directory_plot)
                        if not os.path.exists(directory_env):
                            os.makedirs(directory_env)
                        plt.savefig(directory_plot + "pic_{0:05d}.png".format(t),facecolor=fig.get_facecolor(), edgecolor='none')
                        lov.save(directory_env + "pic_{0:05d}.png".format(t))
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        #this below code snap is helping to write the observation values to csv file using python zip and csv module combination
        #with open('two_finger_ac_values/foo_0.csv','w') as csvfile:
            #values = csv.writer(csvfile)
            #values.writerows(zip(observation_catcher,observation_catcher_1,observation_catcher_2,observation_catcher_3,observation_catcher_4,observation_catcher_5,observation_catcher_6,observation_catcher_7,observation_catcher_8,observation_catcher_9,observation_catcher_10,observation_catcher_11,observation_catcher_12,observation_catcher_13,observation_catcher_14,observation_catcher_15,observation_catcher_16,observation_catcher_17,observation_catcher_18,observation_catcher_19,observation_catcher_20,observation_catcher_21,observation_catcher_22,observation_catcher_23,observation_catcher_24,observation_catcher_25,observation_catcher_26,observation_catcher_27,observation_catcher_28,observation_catcher_29,observation_catcher_30,observation_catcher_31,observation_catcher_32,observation_catcher_33,observation_catcher_34,observation_catcher_35,observation_catcher_36,observation_catcher_37,observation_catcher_38,observation_catcher_39,observation_catcher_40,observation_catcher_41,observation_catcher_42,observation_catcher_43,observation_catcher_44,observation_catcher_45,observation_catcher_46,observation_catcher_47,observation_catcher_48,observation_catcher_49,observation_catcher_50,observation_catcher_51,observation_catcher_52,observation_catcher_53,observation_catcher_54,observation_catcher_55,observation_catcher_56,observation_catcher_57,observation_catcher_58,observation_catcher_59,observation_catcher_60))
            #csvfile.close()

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value
        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size
        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)


