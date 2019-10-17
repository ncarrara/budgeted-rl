# coding=utf-8
import os
from multiprocessing.pool import Pool

from PIL import Image

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import PytorchBudgetedFittedQ, NetBFTQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.budgeted_rl.tools.utils_run import execute_policy_from_config, format_results, get_action_mask
from ncarrara.utils.datastructure import merge_two_dicts
from ncarrara.utils.math_utils import set_seed, near_split, zip_with_singletons
from ncarrara.utils.os import makedirs
from ncarrara.budgeted_rl.tools.policies import PytorchBudgetedFittedPolicy, policy_factory
import numpy as np

import logging

from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.environments.envs_factory import get_actions_str
from rl_agents.trainer.monitor import MonitorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(policy_path, generate_envs, feature_str, device, workspace, bftq_params, seed, general,
         betas_test, N_trajs, gamma, gamma_c, bftq_net_params, **args):
    if not os.path.isabs(policy_path):
        policy_path = workspace / policy_path

    env = envs_factory.generate_envs(**generate_envs)[0][0]
    feature = feature_factory(feature_str)

    bftq = PytorchBudgetedFittedQ(
        device=device,
        workspace=workspace,
        actions_str=get_actions_str(env),
        policy_network=NetBFTQ(size_state=len(feature(env.reset(), env)), n_actions=env.action_space.n,
                               **bftq_net_params),
        gamma=gamma,
        gamma_c=gamma_c,
        cpu_processes=general["cpu"]["processes"],
        env=env,
        hull_options=general["hull_options"],
        **bftq_params)
    bftq.reset(True)

    pi_config = {
        "__class__": repr(PytorchBudgetedFittedPolicy),
        "feature_str": feature_str,
        "network_path": policy_path,
        "betas_for_discretisation": eval(bftq_params["betas_for_discretisation"]),
        "device": device,
        "hull_options": general["hull_options"],
        "clamp_Qc": bftq_params["clamp_Qc"],
        "env": env
    }
    pi = policy_factory(pi_config)

    # Iterate over betas
    for beta in eval(betas_test):
        logger.info("Rendering with beta={}".format(beta))
        set_seed(seed, env)
        for traj in range(N_trajs):
            done = False
            pi.reset()
            info_env = {}
            info_pi = {"beta": beta}
            t = 0

            # Make a workspace for trajectories
            traj_workspace = workspace / "trajs" / "beta={}".format(beta) / "traj={}".format(traj)
            makedirs(traj_workspace)
            bftq.workspace = traj_workspace
            monitor = MonitorV2(env, traj_workspace, add_subdirectory=False)
            obs = monitor.reset()

            # Run trajectory
            while not done:
                action_mask = get_action_mask(env)
                info_pi = merge_two_dicts(info_pi, info_env)
                bftq.draw_Qr_and_Qc(obs, pi.network, "render_t={}".format(t), show=False)
                a, _, info_pi = pi.execute(obs, action_mask, info_pi)
                render(env, workspace, t, a)
                obs, _, done, info_env = monitor.step(a)
                t += 1
            monitor.close()


def render(env, workspace, time, action):
    im = Image.fromarray(env.render(mode="rgb_array"))
    im.save((workspace / "behavior").as_posix()+"/env_t={}_a={}.png".format(time, action))


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

    C.load(config_file).load_pytorch()
    if id:
        C.workspace /= str(id)
        C.update_paths()

    main(device=C.device,
         seed=C.seed,
         workspace=C.path_bftq_egreedy,
         **C.dict["render_bftq"],
         **C.dict)
