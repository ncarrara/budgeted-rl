# coding=utf-8
from ncarrara.budgeted_rl.bftq.budgeted_network import BudgetedNetwork
from ncarrara.budgeted_rl.bftq.pytorch_budgeted_dqn import PytorchBudgetedDQN
from ncarrara.budgeted_rl.tools.configuration import C
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math import epsilon_decay, set_seed
from ncarrara.utils.os import empty_directory, makedirs
from ncarrara.utils_rl.algorithms.dqn import NetDQN, DQN
from ncarrara.utils_rl.environments.envs_factory import generate_envs
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from ncarrara.utils.math import generate_random_point_on_simplex_not_uniform

from ncarrara.utils_rl.transition.replay_memory import Memory

logger = logging.getLogger("run_bdqn")


def main(empty_previous_test=False):
    # betas = np.linspace(0, 1, 5)

    set_seed(C.seed)

    if empty_previous_test:
        empty_directory(C.path_ftq_results)

    envs, params = generate_envs(**C["generate_envs"])
    e = envs[0]
    e.reset()
    feature = feature_factory(C["feature_str"])

    size_state = len(feature(e.reset(), e))
    logger.info("neural net input size : {}".format(size_state))
    N = C["create_data"]["N_trajs"]
    traj_max_size = np.inf
    decays = epsilon_decay(**C["create_data"]["epsilon_decay"], N=N)
    net = BudgetedNetwork(size_state=size_state,
                          layers=C["bftq_net_params"]["intra_layers"] + [2 * e.action_space.n],
                          device=C.device,
                          **C["bftq_net_params"])

    betas_for_discretisation = eval(C["betas_for_discretisation"])
    betas = np.linspace(0, 1, 31)

    print(C["bdqn_params"])
    dqn = PytorchBudgetedDQN(policy_net=net,
                             workspace=C.path_bdqn,
                             device=C.device,
                             gamma=C["gamma"],
                             gamma_c=C["gamma_c"],
                             beta_for_discretisation=betas_for_discretisation,
                             **C["bdqn_params"])
    dqn.reset()
    e.seed(C.seed)
    rrr = []
    rrr_greedy = []
    nb_samples = 0
    rm = Memory()
    result = np.zeros((N, 4))
    for n in range(N):
        if len(betas) > 0:
            betas_to_explore = betas
        else:
            betas_to_explore = [np.random.random()]
        for beta in betas_to_explore:
            logger.info("beta {}".format(beta))
            if N // 10 > 0 and n % (N // 10) == 0:
                logger.debug("DQN step {}/{}".format(n, N))
            s = e.reset()
            done = False
            result_traj = np.zeros(4)
            it = 0
            trajectory = []
            while (not done):
                if np.random.random() < decays[n]:
                    if hasattr(e, "action_space_executable"):
                        raise NotImplementedError("TODO")
                    else:
                        action_repartition = np.random.random(e.action_space.n)
                        action_repartition /= np.sum(action_repartition)
                        budget_repartion = generate_random_point_on_simplex_not_uniform(
                            coeff=action_repartition,
                            bias=beta,
                            min_x=0,
                            max_x=1)
                        a = np.random.choice(a=range(e.action_space.n),
                                             p=action_repartition)
                        beta_ = budget_repartion[a]
                        logger.info(
                            "Random action : {} with a random budget : {:.2f} (from {})"
                                .format(a, beta_, ["{:.2f}".format(bud) for bud in budget_repartion]))
                else:
                    if hasattr(e, "action_space_executable"):
                        raise NotImplementedError("TODO")
                    else:
                        a, beta_ = dqn.pi(feature(s, e), beta, np.zeros(e.action_space.n))
                        logger.info("Greedy action : {} with corresponding budget {}".format(a, beta))

                s_, r_, done, info = e.step(a)
                c_ = info["c_"]
                sample = (s, a if type(a) is str else int(a), r_, s_, done, info)
                trajectory.append(sample)

                result_traj += np.array([r_, c_, r_ * (C["gamma"] ** it), c_ * (C["gamma_c"] ** it)])
                t_dqn = (feature(s, e), a, r_, feature(s_, e), c_, beta, done, info)
                dqn.update(*t_dqn)
                s = s_
                beta = beta_
                nb_samples += 1
                it += 1
                if it % 100 == 0:
                    if it > 500:
                        logger.warning("Number of trajectories overflowing : {}".format(it))
                if traj_max_size is not None and it >= traj_max_size:
                    logger.warning("Max size trajectory reached")
                    break
            logger.info("result_traj : {}".format(result_traj))

            for sample in trajectory:
                rm.push(*sample)

    logger.info("[execute_policy] saving results at : {}".format(C.path_dqn_results))
    np.savetxt(C.path_bdqn_results + "/greedy_lambda_=0.result", result)
    if N > 100:
        nb_traj_packet = 100
        a = np.reshape(rrr, (int(N / nb_traj_packet), -1))
        a = np.mean(a, 1)
        x = np.asarray(range(len(a))) * nb_traj_packet
        plt.plot(x, a)
        a = np.reshape(rrr_greedy, (int(N / nb_traj_packet), -1))
        a = np.mean(a, 1)
        plt.plot(x, a)
        plt.title("dqn results")
        plt.show()
        plt.savefig(C.workspace / "dqn_create_data")
        plt.close()
    rm.save(C.workspace, "/" + C["create_data"]["filename_data"], C["create_data"]["as_json"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/test3.json"
    C.load(config_file).load_pytorch().create_fresh_workspace(force=False)
    main()
