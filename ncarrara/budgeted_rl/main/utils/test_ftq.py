# coding=utf-8
import os
from multiprocessing.pool import Pool

import numpy as np

from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, format_results
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy

import logging

from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(device, workspace, policy_path, generate_envs, feature_str, gamma, gamma_c,
         N_trajs, seed, lambda_, path_results, general, **args):
    if not os.path.isabs(policy_path):
        policy_path = workspace / policy_path

    mock_env = envs_factory.generate_envs(**generate_envs)[0][0]
    pi_config = {
        "__class__": repr(PytorchFittedPolicy),
        "feature_str": feature_str,
        "network_path": policy_path,
        "device": device
    }

    makedirs(path_results)
    set_seed(seed)
    # Prepare workers
    cpu_processes = min(general["cpu"]["processes_when_linked_with_gpu"] or os.cpu_count(), N_trajs)
    workers_n_trajectories = near_split(N_trajs, cpu_processes)
    workers_seeds = np.random.randint(0, 10000, cpu_processes).tolist()
    workers_params = list(zip_with_singletons(
        generate_envs, pi_config, workers_seeds, gamma, gamma_c, workers_n_trajectories, None,
        None, "{}/lambda={}.results".format(path_results, lambda_), general["dictConfig"]))
    # Collect trajectories
    logger.info("Collecting trajectories with {} workers...".format(cpu_processes))
    with Pool(cpu_processes) as pool:
        results = pool.starmap(execute_policy_from_config, workers_params)
        rez = np.concatenate([result for _, result in results], axis=0)
        trajs = []
        for t, _ in results:
            trajs += t

    if isinstance(mock_env, EnvGridWorld):
        from ncarrara.utils_rl.environments.gridworld.world import World
        w = World(mock_env)
        w.draw_frame()
        w.draw_lattice()
        w.draw_cases()
        w.draw_test_trajectories(trajs)
        makedirs(workspace / "trajs")
        pp = (workspace / "trajs" / "trajs_lambda").as_posix()
        w.save(pp + "={:.2f}".format(lambda_))

    print("FTQ({}) : {}".format(lambda_, format_results(rez)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "../config/debug_bftq.json"
    if len(sys.argv) > 2:
        id = sys.argv[2]
    else:
        id = None
    from ncarrara.budgeted_rl.tools.configuration_bftq import C
    C.load(config_file).load_pytorch().load_matplotlib('agg')
    if id:
        C.workspace /= "{}".format(id)
        C.update_paths()

    print("-------- test_ftq_greedy --------")
    lambdas = eval(C.dict["learn_ftq_egreedy"]["lambdas"])
    for lambda_ in lambdas:
        print("test_ftq_greed lambda={}".format(lambda_))
        workspace = C.path_ftq_egreedy + "/lambda={}".format(lambda_)
        main(
            lambda_=lambda_, device=C.device, seed=C.seed,
            workspace=workspace,
            path_results=C.path_ftq_egreedy_results,
            **C.dict["test_ftq"], **C.dict
        )

    print("-------- test_ftq_duplicate --------")
    try:
        lambdas = C.dict["learn_ftq_duplicate"]["lambdas"]
        if type(lambdas) is str:
            lambdas = eval(lambdas)
    except KeyError:
        lambdas = []
    for lambda_ in lambdas:
        print("test_ftq_duplicate, lambda={}".format(lambda_))
        workspace = C.path_ftq_duplicate / "lambda={}".format(lambda_)
        main(
            lambda_=lambda_, device=C.device, seed=C.seed,
            workspace=workspace,
            path_results=C.path_ftq_duplicate_results,
            **C.dict["test_ftq_duplicate"], **C.dict
        )
