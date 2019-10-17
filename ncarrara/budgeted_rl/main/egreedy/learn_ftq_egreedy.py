# coding=utf-8
from multiprocessing.pool import Pool

from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, datas_to_transitions
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld
from ncarrara.utils_rl.environments.gridworld.world import World
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy, RandomPolicy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(generate_envs, feature_str, gamma, gamma_c, ftq_params, ftq_net_params,
         device, epsilon_decay, N_trajs, trajs_by_ftq_batch, normalize_reward,
         workspace, seed, save_memory, general, lambda_=0, **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)
    rm = Memory()
    feature = feature_factory(feature_str)

    def build_fresh_ftq():
        ftq = PytorchFittedQ(
            device=device,
            policy_network=NetFTQ(n_in=len(feature(e.reset(), e)), n_out=e.action_space.n, **ftq_net_params),
            action_str=None if not hasattr(e, "action_str") else e.action_str,
            test_policy=None,
            gamma=gamma,
            **ftq_params
        )
        return ftq

    # Prepare learning
    i_traj = 0
    decays = math_utils.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)
    batch_sizes = near_split(N_trajs, size_bins=trajs_by_ftq_batch)
    pi_epsilon_greedy_config = {
        "__class__": repr(EpsilonGreedyPolicy),
        "pi_greedy": {"__class__": repr(RandomPolicy)},
        "pi_random": {"__class__": repr(RandomPolicy)},
        "epsilon": decays[0]
    }

    # Main loop
    trajs = []
    for batch, batch_size in enumerate(batch_sizes):
        # Prepare workers
        cpu_processes = min(general["cpu"]["processes_when_linked_with_gpu"] or os.cpu_count(), batch_size)
        workers_n_trajectories = near_split(batch_size, cpu_processes)
        workers_start = np.cumsum(workers_n_trajectories)
        workers_traj_indexes = [np.arange(*times) for times in zip(np.insert(workers_start[:-1], 0, 0), workers_start)]
        workers_seeds = np.random.randint(0, 10000, cpu_processes).tolist()
        workers_epsilons = [decays[i_traj + indexes] for indexes in workers_traj_indexes]
        workers_params = list(zip_with_singletons(
            generate_envs, pi_epsilon_greedy_config, workers_seeds, gamma, gamma_c, workers_n_trajectories,
            None, workers_epsilons, None, general["dictConfig"]))

        # Collect trajectories
        logger.info("Collecting trajectories with {} workers...".format(cpu_processes))
        if cpu_processes == 1:
            results = [execute_policy_from_config(*workers_params[0])]
        else:
            with Pool(processes=cpu_processes) as pool:
                results = pool.starmap(execute_policy_from_config, workers_params)
        i_traj += sum([len(trajectories) for trajectories, _ in results])

        # Fill memory
        [rm.push(*sample) for trajectories, _ in results for trajectory in trajectories for sample in trajectory]
        transitions_ftq, _ = datas_to_transitions(rm.memory, e, feature, lambda_, normalize_reward)

        # Fit model
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        logger.info("[BATCH={}][learning ftq pi greedy] #samples={} #traj={}"
                    .format(batch, len(transitions_ftq), i_traj))
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        ftq = build_fresh_ftq()
        ftq.reset(True)
        ftq.workspace = workspace / "batch={}".format(batch)
        makedirs(ftq.workspace)

        if isinstance(e, EnvGridWorld):

            for trajectories, _ in results:
                for traj in trajectories:
                        trajs.append(traj)

            w = World(e)
            w.draw_frame()
            w.draw_lattice()
            w.draw_cases()
            w.draw_source_trajectories(trajs)
            w.save((ftq.workspace / "bftq_on_2dworld_sources").as_posix())

        ftq.fit(transitions_ftq)

        # Save policy
        network_path = ftq.save_policy()
        os.system("cp {}/policy.pt {}/final_policy.pt".format(ftq.workspace, workspace))

        # Update greedy policy
        pi_epsilon_greedy_config["pi_greedy"] = {
            "__class__": repr(PytorchFittedPolicy),
            "feature_str": feature_str,
            "network_path": network_path,
            "device": ftq.device
        }
    if save_memory is not None:
        rm.save(workspace / save_memory["path"], save_memory["as_json"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        lambda_ = float(sys.argv[2])
        force = False
    else:
        config_file = "../config/test_egreedy.json"
        lambda_ = 0.
        force = True
    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).create_fresh_workspace(force=force).load_pytorch().load_matplotlib('agg')
    main(lambda_=lambda_,
         seed=C.seed,
         device=C.device,
         workspace=C.path_learn_ftq_egreedy,
         **C.dict["learn_ftq_egreedy"],
         **C.dict)
