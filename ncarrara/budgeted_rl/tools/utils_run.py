from collections import Iterable

import numpy as np

import ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq as pbf
from ncarrara.budgeted_rl.tools.policies import policy_factory
from ncarrara.utils.datastructure import merge_two_dicts
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.transition.transition import Transition

import logging

logger = logging.getLogger(__name__)


def datas_to_transitions(datas, env, feature, lambda_, normalize_reward):
    logger.info("data to transition ... ")
    max_r_ftq = 0
    max_r_bftq = 0
    e = env
    nbNone = 0
    for data in datas:
        reward_ftq = data.r_ - (lambda_ * data.info["c_"])
        max_r_ftq = np.abs(reward_ftq) if np.abs(reward_ftq) > max_r_ftq else max_r_ftq
        max_r_bftq = np.abs(data.r_) if np.abs(data.r_) > max_r_bftq else max_r_bftq
        if data.s_ is None: nbNone += 1
    logger.info("max_r_ftq : {:.2f} max_r_bftq : {:.2f}".format(max_r_ftq, max_r_bftq))
    transitions_ftq = []
    transitions_bftq = []
    for data in datas:
        # if not data.a in 'hello()':
        r_ = data.r_
        beta = data.info.get("beta", None)
        s = feature(data.s, e)
        if data.done:
            s_ = None
        else:
            s_ = feature(data.s_, e)
        a = data.a  # e.action_space().index(data.a)
        c_ = data.info["c_"]
        reward_ftq = r_ - (lambda_ * c_)
        reward_bftq = r_
        if normalize_reward:
            reward_ftq /= (1. if max_r_ftq == 0. else max_r_ftq)
            reward_bftq /= (1. if max_r_bftq == 0. else max_r_bftq)
        t_ftq = Transition(s, a, reward_ftq, s_)
        t_bftq = pbf.TransitionBFTQ(s, a, reward_bftq, s_, c_, beta, None)
        transitions_ftq.append(t_ftq)
        transitions_bftq.append(t_bftq)
    logger.info("nbdialogues : {}".format(nbNone))
    logger.info("data to transition ... done")
    return transitions_ftq, transitions_bftq


def format_results(results):
    N = len(results)
    rew_r, rew_c, ret_r, ret_c = np.mean(results, axis=0)
    std_rew_r, std_rew_c, std_ret_r, std_ret_c = np.std(results, axis=0)
    p = "R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f} , return : R={:.2f}+/-{:.2f} C={:.2f}+/-{:.2f}".format(
        rew_r, std_rew_r, rew_c, std_rew_c, ret_r, std_ret_r, ret_c, std_ret_c)
    confidence_r = 1.96 * (std_rew_r / np.sqrt(N))
    confidence_r_str = "[{:.2f};{:.2f}]".format(rew_r - confidence_r, rew_r + confidence_r)
    confidence_c = 1.96 * (std_rew_c / np.sqrt(N))
    confidence_c_str = "[{:.2f};{:.2f}]".format(rew_c - confidence_c, rew_c + confidence_c)
    pp = "R=" + confidence_r_str + " C=" + confidence_c_str
    return (pp + " " + p)


def execute_policy_one_trajectory(env, pi, gamma_r=1.0, gamma_c=1.0, beta=None):
    trajectory = []
    pi.reset()

    if hasattr(env, "ID") and env.ID == "gym_pydial":
        s = env.reset()
        a = env.action_space_str.index('hello')
        rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
        i = 0
        s_, r_, end, info_env = env.step(a)
        transition = (s, a, r_, s_, end, info_env)
        trajectory.append(transition)
        info_env = {}
        info_pi = {"beta": beta}
        i += 1
    else:
        s_ = env.reset()
        rew_r, rew_c, ret_r, ret_c = 0., 0., 0., 0.
        i = 0
        info_env = {}
        info_pi = {"beta": beta}
        end = False

    while not end:
        s = s_
        action_mask = get_action_mask(env)
        beta = info_pi["beta"]
        info_pi = merge_two_dicts(info_pi, info_env)

        a, is_master_action, info_pi = pi.execute(s, action_mask, info_pi)
        if hasattr(env, "ID") and env.ID == "gym_pydial":
            s_, r_, end, info_env = env.step(a, is_master_act=is_master_action)
        else:
            s_, r_, end, info_env = env.step(a)
        c_ = info_env["c_"]

        info = info_env.copy()
        info["beta"] = beta

        transition = (s, a if type(a) is str else int(a), r_, s_, end, info)
        rew_r += r_
        rew_c += c_
        ret_r += r_ * (gamma_r ** i)
        ret_c += c_ * (gamma_c ** i)
        trajectory.append(transition)
        i += 1

    return trajectory, rew_r, rew_c, ret_r, ret_c


def get_action_mask(env):
    action_mask = np.zeros(env.action_space.n)
    if hasattr(env, "action_space_executable"):
        raise Exception("Remove this expection please")
        actions = env.action_space_executable()
        action_mask[actions] = 1
    elif hasattr(env, "get_available_actions"):
        actions = env.get_available_actions()
        action_mask = np.ones(env.action_space.n)
        action_mask[actions] = 0
    return action_mask


def execute_policy(env, pi,
                   gamma_r=1.0,
                   gamma_c=1.0,
                   n_trajectories=10,
                   beta=1.,
                   epsilon_schedule=None,
                   save_path=None):
    """
        Execute a policy on an environment for several trajectories.
    :param env: environment
    :param pi: policy
    :param gamma_r: reward discount factor
    :param gamma_c: constraint discount factor
    :param n_trajectories: number of trajectories to generate
    :param beta: constraint threshold. Either a float, or an array of size n_trajectories for beta scheduling
    :param epsilon_schedule: array of size n_trajectories: schedule of epsilon to use for EpsilonGreedy policies
    :param save_path: if not none, results will be saved to that path
    :return: list of trajectories, array of [total reward,
                                             total constraint,
                                             discounted total reward,
                                             discounted total constraint]
    """
    trajectories = []
    results = np.zeros((n_trajectories, 4))
    for d in range(n_trajectories):
        # Beta schedule
        if isinstance(beta, Iterable):
            traj_beta = beta[d]
        else:
            traj_beta = beta
        # Epsilon schedule
        if epsilon_schedule is not None:
            pi.epsilon = epsilon_schedule[d]
        # Execution
        trajectory, rew_r, rew_c, ret_r, ret_c = execute_policy_one_trajectory(env, pi, gamma_r, gamma_c, traj_beta)
        trajectories.append(trajectory)
        results[d] = np.array([rew_r, rew_c, ret_r, ret_c])
    logger.info("[execute_policy] mean length : {}".format(np.mean([len(t) for t in trajectories])))
    if save_path is not None:
        logger.info("[execute_policy] saving results at : {}".format(save_path))
        with open(save_path, 'ab') as f:
            np.savetxt(f, results)
    return trajectories, results


def execute_policy_from_config(generate_envs,
                               policy_config,
                               seed=None,
                               gamma_r=1.0,
                               gamma_c=1.0,
                               n_trajectories=10,
                               beta=1.,
                               epsilon_schedule=None,
                               save_path=None,
                               logging_config={}):
    """
        Generate an environment and a policy from configurations, and collect trajectories.
    :param generate_envs: environment config
    :param policy_config: policy config
    :param seed: to seed the environment before execution
    :param gamma_r: see execute_policy()
    :param gamma_c: see execute_policy()
    :param n_trajectories: see execute_policy()
    :param beta: see execute_policy()
    :param epsilon_schedule: see execute_policy()
    :param save_path: see execute_policy()
    :param logging_config: the logging config of the process
    :return: the collected trajectories
    """
    if logging_config:
        import logging.config as config
        config.dictConfig(logging_config)

    envs, params = envs_factory.generate_envs(**generate_envs)
    env = envs[0]
    set_seed(seed, env)

    policy_config["env"] = env
    pi = policy_factory(policy_config)
    return execute_policy(env, pi, gamma_r, gamma_c, n_trajectories, beta, epsilon_schedule, save_path)


if __name__ == '__main__':
    envs = {
        "envs_str": "highway-v0",
        "envs_params": {
            "lanes_count": [2],
            "initial_spacing": [2],
            "vehicles_count": [5],
            "duration": [20],
            "other_vehicles_type": ["highway_env.vehicle.behavior.IDMVehicle"],
            "centering_position": [[0.3, 0.5]],
            "collision_reward": [0]
        }
    }
    n_trajectories = 10
    from ncarrara.budgeted_rl.tools.policies import RandomBudgetedPolicy
    trajs, res = execute_policy_from_config(envs,
                                            policy_config={"__class__": repr(RandomBudgetedPolicy)},
                                            seed=0,
                                            gamma_r=1,
                                            gamma_c=1,
                                            n_trajectories=n_trajectories,
                                            beta=np.linspace(0, 1, n_trajectories),
                                            epsilon_schedule= 1 - np.linspace(0, 1, n_trajectories),
                                            save_path=None,
                                            logging_config=None)
    print(len(trajs), "trajectories")
    print(res)
