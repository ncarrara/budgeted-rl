# coding=utf-8
import os
from multiprocessing.pool import Pool

from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, format_results
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy
import numpy as np

import logging

from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(betas_test, policy_path, generate_envs, feature_str, device, workspace, gamma, gamma_c,
         bftq_params, seed, N_trajs, path_results, general, **args):
    if not os.path.isabs(policy_path):
        policy_path = workspace / policy_path

    pi_config = {
        "__class__": repr(PytorchBudgetedFittedPolicy),
        "feature_str": feature_str,
        "network_path": policy_path,
        "betas_for_discretisation": eval(bftq_params["betas_for_discretisation"]),
        "device": device,
        "hull_options": general["hull_options"],
        "clamp_Qc": bftq_params["clamp_Qc"]
    }
    mock_env = envs_factory.generate_envs(**generate_envs)[0][0]
    makedirs(workspace / "trajs")

    makedirs(path_results)
    set_seed(seed)
    try:
        for beta in eval(betas_test):
            # Prepare workers
            cpu_processes = min(general["cpu"]["processes_when_linked_with_gpu"] or os.cpu_count(), N_trajs)
            workers_n_trajectories = near_split(N_trajs, cpu_processes)
            workers_seeds = np.random.randint(0, 10000, cpu_processes).tolist()
            workers_params = list(zip_with_singletons(
                generate_envs, pi_config, workers_seeds, gamma, gamma_c, workers_n_trajectories, beta,
                None, "{}/beta={}.results".format(path_results, beta), general["dictConfig"]))
            logger.info("Collecting trajectories with {} workers...".format(cpu_processes))
            with Pool(cpu_processes) as pool:
                results = pool.starmap(execute_policy_from_config, workers_params)
                rez = np.concatenate([result for _, result in results], axis=0)

                trajs = []
                for t, _ in results:
                    trajs += t
            print("BFTQ({:.2f}) : {}".format(beta, format_results(rez)))

            if isinstance(mock_env, EnvGridWorld):
                from ncarrara.utils_rl.environments.gridworld.world import World
                w = World(mock_env)
                w.draw_frame()
                w.draw_lattice()
                w.draw_cases()
                w.draw_test_trajectories(trajs)
                pp = (workspace / "trajs" / "trajs_beta").as_posix()
                w.save(pp+"={:.2f}".format(beta))
        if isinstance(mock_env, EnvGridWorld):
            os.system(
                "convert -delay 10 -loop 0 " + workspace.as_posix() + "/trajs/" + "*.png " + workspace.as_posix() + "/out.gif")

    except FileNotFoundError as e:
        logger.warning("Could not load policy: {}".format(e))


if __name__ == "__main__":
    import sys

    # Workspace from config and id
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

    main(device=C.device,
         seed=C.seed,
         workspace=C.workspace,
         path_results=C.path_bftq_results,
         **C.dict["test_bftq"],
         **C.dict)
