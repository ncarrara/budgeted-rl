# coding=utf-8
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, datas_to_transitions
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import get_memory_for_pid
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.envs_factory import get_actions_str
from ncarrara.utils_rl.environments.gridworld.envgridworld import EnvGridWorld
from ncarrara.utils_rl.environments.gridworld.world import World
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import RandomBudgetedPolicy, PytorchBudgetedFittedPolicy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy

import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(generate_envs, feature_str, betas_for_exploration, gamma, gamma_c, bftq_params, bftq_net_params, N_trajs,
         workspace, seed, device, normalize_reward, trajs_by_ftq_batch, epsilon_decay, general, **args):
    # Prepare BFTQ
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)
    rm = Memory()
    feature = feature_factory(feature_str)


    def build_fresh_bftq():
        bftq = PytorchBudgetedFittedQ(
            device=device,
            workspace=workspace / "batch=0",
            actions_str=get_actions_str(e),
            policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n,
                                   **bftq_net_params),
            gamma=gamma,
            gamma_c=gamma_c,
            cpu_processes=general["cpu"]["processes"],
            env=e,
            split_batches=general["gpu"]["split_batches"],
            hull_options=general["hull_options"],
            **bftq_params)
        return bftq

    # Prepare learning
    i_traj = 0
    decays = math_utils.epsilon_decay(**epsilon_decay, N=N_trajs, savepath=workspace)
    betas_for_exploration = np.array(eval(betas_for_exploration))
    memory_by_batch = [get_current_memory()]
    batch_sizes = near_split(N_trajs, size_bins=trajs_by_ftq_batch)
    pi_epsilon_greedy_config = {
        "__class__": repr(EpsilonGreedyPolicy),
        "pi_greedy": {"__class__": repr(RandomBudgetedPolicy)},
        "pi_random": {"__class__": repr(RandomBudgetedPolicy)},
        "epsilon": decays[0],
        "hull_options": general["hull_options"],
        "clamp_Qc": bftq_params["clamp_Qc"]
    }

    # Main loop
    trajs = []
    for batch, batch_size in enumerate(batch_sizes):
        # Prepare workers
        cpu_processes = min(general["cpu"]["processes_when_linked_with_gpu"] or os.cpu_count(), batch_size)
        workers_n_trajectories = near_split(batch_size, cpu_processes)
        workers_start = np.cumsum(workers_n_trajectories)
        workers_traj_indexes = [np.arange(*times) for times in zip(np.insert(workers_start[:-1], 0, 0), workers_start)]
        if betas_for_exploration.size:
            workers_betas = [betas_for_exploration.take(indexes, mode='wrap') for indexes in workers_traj_indexes]
        else:
            workers_betas = [np.random.random(indexes.size) for indexes in workers_traj_indexes]
        workers_seeds = np.random.randint(0, 10000, cpu_processes).tolist()
        workers_epsilons = [decays[i_traj + indexes] for indexes in workers_traj_indexes]
        workers_params = list(zip_with_singletons(
            generate_envs, pi_epsilon_greedy_config, workers_seeds, gamma, gamma_c, workers_n_trajectories,
            workers_betas, workers_epsilons, None, general["dictConfig"]))

        # Collect trajectories
        logger.info("Collecting trajectories with {} workers...".format(cpu_processes))
        if cpu_processes == 1:
            results = []
            for params in workers_params:
                results.append(execute_policy_from_config(*params))
        else:
            with Pool(processes=cpu_processes) as pool:
                results = pool.starmap(execute_policy_from_config, workers_params)
        i_traj += sum([len(trajectories) for trajectories, _ in results])

        # Fill memory
        [rm.push(*sample) for trajectories, _ in results for trajectory in trajectories for sample in trajectory]

        transitions_ftq, transition_bftq = datas_to_transitions(rm.memory, e, feature, 0, normalize_reward)

        # Fit model
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        logger.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}"
                    .format(batch, len(transition_bftq), i_traj))
        logger.info("[BATCH={}]---------------------------------------".format(batch))
        bftq = build_fresh_bftq()
        bftq.reset(True)
        bftq.workspace = workspace / "batch={}".format(batch)
        makedirs(bftq.workspace)
        if isinstance(e, EnvGridWorld):
            for trajectories, _ in results:
                for traj in trajectories:
                        trajs.append(traj)

            w = World(e)
            w.draw_frame()
            w.draw_lattice()
            w.draw_cases()
            w.draw_source_trajectories(trajs)
            w.save((bftq.workspace / "bftq_on_2dworld_sources").as_posix())
        q = bftq.fit(transition_bftq)

        # Save policy
        network_path = bftq.save_policy()
        os.system("cp {}/policy.pt {}/policy.pt".format(bftq.workspace, workspace))

        # Save memory
        save_memory(bftq, memory_by_batch, by_batch=False)

        # Update greedy policy
        pi_epsilon_greedy_config["pi_greedy"] = {
            "__class__": repr(PytorchBudgetedFittedPolicy),
            "feature_str": feature_str,
            "network_path": network_path,
            "betas_for_discretisation": bftq.betas_for_discretisation,
            "device": bftq.device,
            "hull_options": general["hull_options"],
            "clamp_Qc": bftq_params["clamp_Qc"]
        }

        if isinstance(e, EnvGridWorld):
            def pi(state, beta):
                import torch
                from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import convex_hull, \
                    optimal_pia_pib
                with torch.no_grad():
                    hull = convex_hull(s=torch.tensor([state], device=device, dtype=torch.float32),
                                       Q=q,
                                       action_mask=np.zeros(e.action_space.n),
                                       id="run_" + str(state), disp=False,
                                       betas=bftq.betas_for_discretisation,
                                       device=device,
                                       hull_options=general["hull_options"],
                                       clamp_Qc=bftq_params["clamp_Qc"])
                    opt, _ = optimal_pia_pib(beta=beta, hull=hull, statistic={})
                return opt

            def qr(state, a, beta):
                import torch
                s = torch.tensor([[state]], device=device)
                b = torch.tensor([[[beta]]], device=device)
                sb = torch.cat((s, b), dim=2)
                return q(sb).squeeze()[a]

            def qc(state, a, beta):
                import torch
                s = torch.tensor([[state]], device=device)
                b = torch.tensor([[[beta]]], device=device)
                sb = torch.cat((s, b), dim=2)
                return q(sb).squeeze()[e.action_space.n + a]

            w = World(e, bftq.betas_for_discretisation)
            w.draw_frame()
            w.draw_lattice()
            w.draw_cases()
            w.draw_policy_bftq(pi, qr, qc, bftq.betas_for_discretisation)
            w.save((bftq.workspace / "bftq_on_2dworld").as_posix())

    save_memory(bftq, memory_by_batch, by_batch=True)


def save_memory(bftq, memory_by_batch, by_batch=False):
    if not by_batch:
        plt.rcParams["figure.figsize"] = (30, 5)
        plt.grid()
        y_mem = np.asarray(bftq.memory_tracking)[:, 1]
        y_mem = [int(mem) for mem in y_mem]
        plt.plot(range(len(y_mem)), y_mem)
        props = {'ha': 'center', 'va': 'center', 'bbox': {'fc': '0.8', 'pad': 0}}
        for i, couple in enumerate(bftq.memory_tracking):
            id, memory = couple
            plt.scatter(i, memory, s=25)
            plt.text(i, memory, id, props, rotation=90)
        plt.savefig(bftq.workspace / "memory_tracking.png")
        plt.rcParams["figure.figsize"] = (5, 5)
        plt.close()
        memory_by_batch.append(get_current_memory())
    else:
        plt.plot(range(len(memory_by_batch)), memory_by_batch)
        plt.grid()
        plt.title("memory_by_batch")
        plt.savefig(bftq.workspace / "memory_by_batch.png")
        plt.close()


def get_current_memory():
    return sum(get_memory_for_pid(os.getpid()))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        config_file = sys.argv[1]
        force = bool(sys.argv[2])
    else:
        config_file = "../config/test_egreedy.json"
        force = True
    from ncarrara.budgeted_rl.tools.configuration import C

    C.load(config_file).create_fresh_workspace(force=force).load_pytorch().load_matplotlib('agg')
    main(device=C.device,
         seed=C.seed,
         workspace=C.path_learn_bftq_egreedy,
         **C.dict["learn_bftq_egreedy"],
         **C.dict)
