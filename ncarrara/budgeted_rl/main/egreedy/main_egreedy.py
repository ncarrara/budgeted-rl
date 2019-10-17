import os

import torch

from ncarrara.budgeted_rl.main.utils import plot_data
from ncarrara.budgeted_rl.tools.configuration_bftq import C
from ncarrara.budgeted_rl.main.utils import test_bftq, test_ftq, abstract_main
from ncarrara.budgeted_rl.main.egreedy import learn_ftq_egreedy, learn_ftq_full_batch, learn_bftq_full_batch, \
    learn_bftq_egreedy
import sys


def main(config):

    if "learn_bftq_egreedy" in config:
        print("-------- learn_bftq_greedy --------")
        config.dict["bftq_params"]["betas_for_duplication"] = \
            config.dict["learn_bftq_egreedy"]["betas_for_duplication"]
        learn_bftq_egreedy.main(
            device=config.device, seed=config.seed,
            workspace=config.path_bftq_egreedy,
            **config.dict["learn_bftq_egreedy"],
            **config.dict
        )
    torch.cuda.empty_cache()

    if "test_bftq" in config:
        print("-------- test_bftq_greedy --------")
        test_bftq.main(
            device=config.device, seed=config.seed,
            workspace=config.path_bftq_egreedy,
            path_results=config.path_bftq_egreedy_results,
            **config.dict["test_bftq"], **config.dict
        )

    torch.cuda.empty_cache()

    if "learn_ftq_duplicate" in config:
        print("-------- learn_ftq_duplicate --------")
        workspace = config.path_ftq_duplicate
        learn_ftq_egreedy.main(
            seed=config.seed, device=config.device,
            workspace=workspace,
            **config.dict["learn_ftq_duplicate"]["generate_data"], **config.dict
        )
        if "learn_ftq_full_batch" in  config.dict["learn_ftq_duplicate"]:
            lambdas = config.dict["learn_ftq_duplicate"]["lambdas"]
            if type(lambdas) is str:
                lambdas = eval(lambdas)
            config.dict["learn_ftq_duplicate"]["learn_ftq_full_batch"]["load_memory"]["path"] \
                = workspace / config.dict["learn_ftq_duplicate"]["learn_ftq_full_batch"]["load_memory"]["path"]
            for lambda_ in lambdas:
                print("learn_ftq_duplicate, lambda={}".format(lambda_))
                torch.cuda.empty_cache()
                learn_ftq_full_batch.main(
                    lambda_=lambda_,
                    seed=config.seed,
                    device=config.device,
                    workspace=workspace / "lambda={}".format(lambda_),
                    **config.dict["learn_ftq_duplicate"]["learn_ftq_full_batch"],
                    **config.dict)
    torch.cuda.empty_cache()

    if "test_ftq_duplicate" in config:
        print("-------- test_ftq_duplicate --------")
        try:
            lambdas = config.dict["learn_ftq_duplicate"]["lambdas"]
            if type(lambdas) is str:
                lambdas = eval(lambdas)
        except KeyError:
            lambdas = []
        for lambda_ in lambdas:
            print("test_ftq_duplicate, lambda={}".format(lambda_))
            torch.cuda.empty_cache()
            workspace = config.path_ftq_duplicate / "lambda={}".format(lambda_)
            test_ftq.main(
                lambda_=lambda_, device=config.device, seed=config.seed,
                workspace=workspace,
                path_results=config.path_ftq_duplicate_results,
                **config.dict["test_ftq_duplicate"], **config.dict
            )
    torch.cuda.empty_cache()



    if "learn_ftq_egreedy" in config:
        print("-------- learn_ftq_egreedy --------")
        lambdas = config.dict["learn_ftq_egreedy"]["lambdas"]
        if type(lambdas) is str:
            lambdas = eval(lambdas)
        for lambda_ in lambdas:
            print("learn_ftq_egreedy lambda={}".format(lambda_))
            torch.cuda.empty_cache()
            workspace = config.path_ftq_egreedy / "lambda={}".format(lambda_)
            learn_ftq_egreedy.main(
                lambda_=lambda_, seed=config.seed, device=config.device,
                workspace=workspace,
                **config.dict["learn_ftq_egreedy"], **config.dict
            )
    torch.cuda.empty_cache()

    if "test_ftq" in config:
        print("-------- test_ftq_greedy --------")
        lambdas = eval(config.dict["learn_ftq_egreedy"]["lambdas"])
        if type(lambdas) is str:
            lambdas = eval(lambdas)
        for lambda_ in lambdas:
            print("test_ftq_greed lambda={}".format(lambda_))
            torch.cuda.empty_cache()
            workspace = config.path_ftq_egreedy / "lambda={}".format(lambda_)
            test_ftq.main(
                lambda_=lambda_, device=config.device, seed=config.seed,
                workspace=workspace,
                path_results=config.path_ftq_egreedy_results,
                **config.dict["test_ftq"], **config.dict
            )

    if "learn_bftq_duplicate" in config:
        print("-------- learn_bftq_duplicate --------")
        workspace = config.path_bftq_duplicate
        config.dict["learn_bftq_duplicate"]["load_memory"]["path"] \
            = workspace / config.dict["learn_bftq_duplicate"]["load_memory"]["path"]
        torch.cuda.empty_cache()
        config.dict["bftq_params"]["betas_for_duplication"] = \
            config.dict["learn_bftq_duplicate"]["betas_for_duplication"]
        learn_bftq_full_batch.main(
            seed=config.seed,
            device=config.device,
            workspace=workspace,
            **config.dict["learn_bftq_duplicate"],
            **config.dict)
    torch.cuda.empty_cache()

    if "test_bftq_duplicate" in config:
        print("-------- test_bftq_duplicate --------")
        test_bftq.main(
            device=config.device, seed=config.seed,
            workspace=config.path_bftq_duplicate,
            path_results=config.path_bftq_duplicate_results,
            **config.dict["test_bftq_duplicate"], **config.dict
        )

    torch.cuda.empty_cache()


    plot_data.main(os.path.join(config.workspace, os.pardir))


if __name__ == "__main__":
    seeds = None
    override_device_str = None
    print(sys.argv)
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if len(sys.argv) > 2:
            seed_start = int(sys.argv[2])
            number_seeds = int(sys.argv[3])
            seeds = range(seed_start, seed_start + number_seeds)
            if len(sys.argv) > 4:
                override_device_str = sys.argv[4]
    else:
        config_file = "../config/test_egreedy.json"
        C.load(config_file).create_fresh_workspace(force=True)
        seeds = [0, 1]

    override_param_grid = {}
    if seeds is not None:
        override_param_grid['general.seed'] = seeds

    abstract_main.main(C,config_file, override_param_grid, override_device_str, main)
