# coding=utf-8
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils import math_utils
from ncarrara.utils.math_utils import set_seed
from ncarrara.utils.os import makedirs
from ncarrara.utils_rl.algorithms.pytorch_fittedq import NetFTQ, PytorchFittedQ
from ncarrara.utils_rl.environments import envs_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.budgeted_rl.tools.policies import PytorchFittedPolicy, RandomPolicy
import ncarrara.budgeted_rl.tools.utils_run as urpy
from ncarrara.budgeted_rl.tools.policies import EpsilonGreedyPolicy
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(load_memory, generate_envs, feature_str, gamma, ftq_params, ftq_net_params, device, normalize_reward,
         workspace, seed,
         lambda_=0., **args):
    envs, params = envs_factory.generate_envs(**generate_envs)
    e = envs[0]
    set_seed(seed, e)

    feature = feature_factory(feature_str)

    ftq = PytorchFittedQ(
        device=device,
        policy_network=NetFTQ(n_in=len(feature(e.reset(), e)), n_out=e.action_space.n, **ftq_net_params),
        action_str=None if not hasattr(e, "action_str") else e.action_str,
        test_policy=None,
        gamma=gamma,
        **ftq_params
    )

    rm = Memory()
    rm.load_memory(**load_memory)

    transitions_ftq, _ = urpy.datas_to_transitions(rm.memory, e, feature, lambda_, normalize_reward)
    logger.info("[learning ftq with full batch] #samples={} ".format(len(transitions_ftq)))
    ftq.reset(True)
    ftq.workspace = workspace
    makedirs(ftq.workspace)
    ftq.fit(transitions_ftq)
    ftq.save_policy()


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
