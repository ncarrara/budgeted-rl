# # coding=utf-8
# from ncarrara.budgeted_rl.tools.configuration import C
# from ncarrara.budgeted_rl.tools.features import feature_factory
# from ncarrara.utils.math_utils import set_seed
# from ncarrara.utils.os import empty_directory, makedirs
# from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
# from ncarrara.utils_rl.environments.envs_factory import generate_envs
# from ncarrara.utils_rl.transition.replay_memory import Memory
# from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy
# import ncarrara.budgeted_rl.tools.utils_run as urpy
# import logging
# import numpy as np
#
#
# def main(lambdas_, empty_previous_test=False):
#
#     logger = logging.getLogger(__name__)
#     if empty_previous_test:
#         empty_directory(C.path_ftq_results)
#
#     envs, params = generate_envs(**C["generate_envs"])
#     e = envs[0]
#
#     set_seed(C.seed, e)
#     feature = feature_factory(C["feature_str"])
#     size_state = len(feature(e.reset(), e))
#     logger.info("neural net input size : {}".format(size_state))
#
#     policy_network = NetFTQ(n_in=size_state,
#                             n_out=e.action_space.n,
#                             **C["net_params"])
#
#     def process_between_epoch(pi):
#         logger.info("process_between_epoch ...")
#         pi = PytorchFittedPolicy(pi, e, feature)
#         _, results = urpy.execute_policy(e, pi, C["gamma"], C["gamma_c"], C["nb_trajs_between_epoch"], 1.)
#         return np.mean(results, axis=0)
#
#     action_str = getattr(e, "action_space_str", map(str, range(e.action_space.n)))
#
#     ftq = PytorchFittedQ(
#         device=C.device,
#         test_policy=process_between_epoch,
#         workspace=C.path_ftq,
#         action_str=action_str,
#         policy_network=policy_network,
#         gamma = C["gamma"],
#         **C["ftq_params"]
#     )
#     rm = Memory()
#     if C["main"]["path_data"] is None:
#         path_data = C.workspace / C["main"]["filename_data"]
#     else:
#         path_data = C["main"]["path_data"] + "/" + C["main"]["filename_data"]
#     rm.load_memory(path_data, C["create_data"]["as_json"])
#     makedirs(C.path_ftq_results)
#     for lambda_ in lambdas_:
#         # learning
#         set_seed(C.seed, e)
#         transitions_ftq, transition_bftq = urpy.datas_to_transitions(rm.memory, e, feature,
#                                                                      lambda_,
#                                                                      C["main"]["normalize_reward"])
#
#         # Sub-workspace for saving plots
#         ftq.workspace = C.path_ftq + '/lambda_={}'.format(lambda_)
#         makedirs(ftq.workspace)
#
#         ftq.reset(True)
#         pi = ftq.fit(transitions_ftq)
#         pi = PytorchFittedPolicy(pi, e, feature)
#
#         # testing
#         set_seed(C.seed, e)
#         _, results = urpy.execute_policy(e, pi,
#                                          C["gamma"],
#                                          C["gamma_c"],
#                                          n_trajectories=C["main"]["N_trajs"],
#                                          save_path="{}/lambda_={}.results".format(C.path_ftq_results, lambda_))
#         print("FTQ({}) : {} ".format(lambda_, urpy.format_results(results)))
#
#
# if __name__ == "__main__":
#     C.load("config/test_slot_filling.json").load_pytorch()
#     main(lambda_=[0])
