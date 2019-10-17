# coding=utf-8
import os

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import NetBFTQ, PytorchBudgetedFittedQ
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.environments import envs_factory
import ncarrara.budgeted_rl.tools.utils_run as urpy

import logging

from ncarrara.utils_rl.environments.envs_factory import get_actions_str
from ncarrara.utils_rl.transition.replay_memory import Memory


def main(load_memory, generate_envs, feature_str, gamma, gamma_c, bftq_params, bftq_net_params,
         workspace, seed, device, normalize_reward, general, **args):
    logger = logging.getLogger(__name__)

    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    e.reset()
    set_seed(seed, e)
    feature = feature_factory(feature_str)

    bftq = PytorchBudgetedFittedQ(
        device=device,
        workspace=workspace,
        actions_str=get_actions_str(e),
        policy_network=NetBFTQ(size_state=len(feature(e.reset(), e)), n_actions=e.action_space.n,
                               **bftq_net_params),
        gamma=gamma,
        gamma_c=gamma_c,
        split_batches=general["gpu"]["split_batches"],
        cpu_processes=general["cpu"]["processes"],
        env=e,
        **bftq_params)

    makedirs(workspace)
    rm = Memory()
    rm.load_memory(**load_memory)

    _, transitions_bftq = urpy.datas_to_transitions(rm.memory, e, feature, 0, normalize_reward)
    logger.info("[learning bftq with full batch] #samples={} ".format(len(transitions_bftq)))

    bftq.reset(True)
    _ = bftq.fit(transitions_bftq)

    bftq.save_policy()
