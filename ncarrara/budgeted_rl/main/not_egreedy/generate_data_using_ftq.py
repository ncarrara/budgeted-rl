# # coding=utf-8
# from ncarrara.budgeted_rl.tools.configuration import C
# # C.load_matplotlib('agg')
# from ncarrara.budgeted_rl.tools.features import feature_factory
# from ncarrara.utils.math_utils import epsilon_decay, set_seed
# from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
# from ncarrara.utils_rl.environments.envs_factory import generate_envs
# from ncarrara.utils_rl.transition.replay_memory import Memory
# from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy, RandomPolicy, HandcraftedSlotFillingEnv
# import ncarrara.budgeted_rl.tools.utils_run as urpy
# from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy
#
# import numpy as np
# import logging
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# def main():
#     # print(logger)
#     # logger.setLevel(C.logging_level)
#     envs, params = generate_envs(**C["generate_envs"])
#     e = envs[0]
#     set_seed(C.seed, e)
#
#     feature = feature_factory(C["feature_str"])
#
#     def process_between_epoch(pi):
#         logger.info("process_between_epoch ...")
#         pi = PytorchFittedPolicy(pi, e, feature)
#         _, results = urpy.execute_policy(e, pi, C["gamma"], C["gamma_c"], C["nb_trajs_between_epoch"], 1.)
#         return np.mean(results, axis=0)
#
#     size_state = len(feature(e.reset(), e))
#     logger.info("neural net input size : {}".format(size_state))
#
#     policy_network = NetFTQ(n_in=size_state, n_out=e.action_space.n, **C["net_params"])
#
#     ftq = PytorchFittedQ(
#         device=C.device,
#         action_str=e.action_space_str,
#         policy_network=policy_network,
#         test_policy=process_between_epoch,
#         gamma=C["gamma"],
#         **C["ftq_params"]
#     )
#
#     pi_greedy = None
#
#     decays = epsilon_decay(**C["create_data"]["epsilon_decay"], N=C["create_data"]["N_trajs"], show=True)
#     logger.info("decays={}".format(decays))
#
#     pi_epsilon_greedy = EpsilonGreedyPolicy(pi_greedy, decays[0])
#     pi_greedy = RandomPolicy()
#     rez = np.zeros((C["create_data"]["N_trajs"], 4))
#     rm = Memory()
#
#
#     for i in range(C["create_data"]["N_trajs"]):
#         if i % 50 == 0: logger.info(i)
#         pi_epsilon_greedy.epsilon = decays[i]
#         pi_epsilon_greedy.pi_greedy = pi_greedy
#         trajectory, rew_r, rew_c, ret_r, ret_c = urpy.execute_policy_one_trajectory(
#             e, pi_epsilon_greedy, gamma_r=C["gamma"], gamma_c=C["gamma_c"], beta=1.0)
#         rez[i] = np.array([rew_r, rew_c, ret_r, ret_c])
#         for sample in trajectory:
#             rm.push(*sample)
#         if C["create_data"]["handcrafted_greedy_policy"]:
#             pi_greedy = HandcraftedSlotFillingEnv(env=e, safeness=0.5)
#         else:
#             if i > 0 and i % C["create_data"]["trajs_by_ftq_batch"] == 0:
#                 transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
#                                                                              C["create_data"]["lambda_"],
#                                                                              C["create_data"]["normalize_reward"])
#                 logger.info("[LEARNING FTQ PI GREEDY] #samples={}".format(len(transitions_ftq)))
#
#                 ftq.reset(True)
#                 pi = ftq.fit(transitions_ftq)
#                 pi_greedy = PytorchFittedPolicy(pi, e, feature)
#     # for sample in rm.memory:
#     #     print(sample)
#     rm.save_memory(C.workspace, "/" + C["create_data"]["filename_data"], C["create_data"]["as_json"])
#     np.savetxt(C.workspace / C.id + ".results", rez)
#
#     _, rez = urpy.execute_policy(e, pi_epsilon_greedy,
#                                  gamma_r=C["gamma"], gamma_c=C["gamma_c"],
#                                  beta=1.0, n_trajectories=100)
#     print("greedy results")
#     print(urpy.format_results(rez))
#
#
# if __name__ == "__main__":
#     C.load("config/final.json").load_pytorch().create_fresh_workspace()
#     main()
