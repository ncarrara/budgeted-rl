# # coding=utf-8
# from ncarrara.budgeted_rl.tools.configuration import C
# from ncarrara.budgeted_rl.tools.features import feature_factory
# from ncarrara.utils.math_utils import epsilon_decay, set_seed
# from ncarrara.utils.os import empty_directory, makedirs
# from ncarrara.utils_rl.algorithms.dqn import NetDQN, DQN
# from ncarrara.utils_rl.environments.envs_factory import generate_envs
# import logging
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# from ncarrara.utils_rl.transition.replay_memory import Memory
#
#
# def main(empty_previous_test=False):
#
#     logger = logging.getLogger(__name__)
#     if empty_previous_test:
#         empty_directory(C.path_ftq_results)
#
#     envs, params = generate_envs(**C["generate_envs"])
#     e = envs[0]
#     feature = feature_factory(C["feature_str"])
#
#     set_seed(C.seed, e)
#
#     size_state = len(feature(e.reset(), e))
#     logger.info("neural net input size : {}".format(size_state))
#     N = C["create_data"]["N_trajs"]
#     traj_max_size = np.inf
#     decays = epsilon_decay(**C["create_data"]["epsilon_decay"], N=N, show=True)
#     net = NetDQN(n_in=size_state, n_out=e.action_space.n, **C["net_params"])
#     dqn = DQN(policy_network=net, device=C.device, gamma=C["gamma"], **C["dqn_params"])
#     dqn.reset()
#     rrr = []
#     rrr_greedy = []
#     nb_samples = 0
#     rm = Memory()
#     result = np.zeros((N, 4))
#     for n in range(N):
#         if n % (N // 10) == 0:
#             logger.debug("DQN step {}/{}".format(n, N))
#         s = e.reset()
#         done = False
#         rr = 0
#         it = 0
#         trajectory = []
#         while (not done):
#             if np.random.random() < decays[n]:
#                 if hasattr(e, "action_space_executable"):
#                     a = np.random.choice(e.action_space_executable())
#                 else:
#                     a = e.action_space.sample()
#             else:
#                 if hasattr(e, "action_space_executable"):
#                     exec = e.action_space_executable()
#                     action_mask = np.ones(e.action_space.n)
#                     for ex in exec:
#                         action_mask[ex] = 0.
#                     a = dqn.pi(feature(s, e), action_mask)
#                 else:
#                     a = dqn.pi(feature(s, e), np.zeros(e.action_space.n))
#
#             s_, r_, done, info = e.step(a)
#             sample = (s, a if type(a) is str else int(a), r_, s_, done, info)
#             trajectory.append(sample)
#
#             rr += r_
#             t_dqn = (feature(s, e), a, r_, feature(s_, e), done, info)
#             # print("before",s_)
#             dqn.update(*t_dqn)
#             # print("after",s_)
#             s = s_
#             nb_samples += 1
#             it += 1
#             if it % 100 == 0:
#                 if it > 500:
#                     logger.warning("Number of trajectories overflowing : {}".format(it))
#             if traj_max_size is not None and it >= traj_max_size:
#                 logger.warning("Max size trajectory reached")
#                 break
#
#         rrr.append(rr)
#
#         for sample in trajectory:
#             rm.push(*sample)
#
#         s = e.reset()
#         done = False
#         rr = 0
#         it = 0
#         rew_r = 0.
#         ret_r = 0.
#         rew_c = 0.
#         ret_c = 0.
#         while not done:
#             if hasattr(e, "action_space_executable"):
#                 exec = e.action_space_executable()
#                 action_mask = np.ones(e.action_space.n)
#                 for ex in exec:
#                     action_mask[ex] = 0.
#                 a = dqn.pi(feature(s, e), action_mask)
#             else:
#                 a = dqn.pi(feature(s, e), np.zeros(e.action_space.n))
#
#             s_, r_, done, info = e.step(a)
#             c_ = info["c_"]
#             rew_r += r_
#             rew_c += c_
#             ret_r += r_ * (C["gamma"] ** it)
#             ret_c += c_ * (C["gamma_c"] ** it)
#             rr += r_
#
#             s = s_
#             it += 1
#             if it % 100 == 0:
#                 if it > 500:
#                     logger.warning("Number of trajectories overflowing : {}".format(it))
#             if traj_max_size is not None and it >= traj_max_size:
#                 logger.warning("Max size trajectory reached")
#                 break
#         rrr_greedy.append(rr)
#         result[n] = np.array([rew_r, rew_c, ret_r, ret_c])
#     logger.info("[execute_policy] saving results at : {}".format(C.path_dqn_results))
#     np.savetxt(C.path_dqn_results + "/greedy_lambda_=0.result", result)
#     if N > 100:
#         nb_traj_packet = 100
#         a = np.reshape(rrr, (int(N / nb_traj_packet), -1))
#         a = np.mean(a, 1)
#         x = np.asarray(range(len(a))) * nb_traj_packet
#         plt.plot(x, a)
#         a = np.reshape(rrr_greedy, (int(N / nb_traj_packet), -1))
#         a = np.mean(a, 1)
#         plt.plot(x, a)
#         plt.title("dqn results")
#         plt.show()
#         plt.savefig(C.workspace / "dqn_create_data")
#         plt.close()
#     rm.save_memory(C.workspace, "/" + C["create_data"]["filename_data"], C["create_data"]["as_json"])
#
#
# if __name__ == "__main__":
#     C.load("config/final.json").load_pytorch()
#     main()
