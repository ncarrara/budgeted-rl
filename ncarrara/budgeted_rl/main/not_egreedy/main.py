# from ncarrara.budgeted_rl.main import run_dqn, run_ftq
# from ncarrara.budgeted_rl.main.not_egreedy import learn_bftq
# from ncarrara.budgeted_rl.main.utils import test_bftq, abstract_main
# from ncarrara.budgeted_rl.tools.configuration import C
# import sys
#
# seeds = None
# if len(sys.argv) > 1:
#     config_file = sys.argv[1]
#     if len(sys.argv) >2:
#         seed_start = int(sys.argv[2])
#         number_seeds = int(sys.argv[3])
#         seeds = range(seed_start, seed_start + number_seeds)
# else:
#     # config_file = "config/test.json"
#     config_file = "config/test_highway.json"
#
#
#
# def main():
#     print("Learning DQN")
#     run_dqn.main()
#     print("Learning and testing FTQ")
#     lambdas = eval(C["lambdas"])
#     run_ftq.main(lambdas_=lambdas, empty_previous_test=True)
#     print("Learning BFTQ")
#     betas_test = eval(C["betas_test"])
#     learn_bftq.main()
#     print("Testing BFTQ")
#     test_bftq.main(betas_test=betas_test)
#
#
# override_param_grid = {
#     'general.seed': seeds,
# }
#
# abstract_main.main(config_file, override_param_grid, main)
#
