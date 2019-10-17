import abc
import copy
import importlib
import logging
import torch
import numpy as np

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import convex_hull, optimal_pia_pib
from ncarrara.budgeted_rl.tools.features import feature_factory
from ncarrara.utils.math_utils import generate_random_point_on_simplex_not_uniform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Policy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        pass

    # must return Q function (s,a) -> double
    @abc.abstractmethod
    def execute(self, s, action_mask, info):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class HandcraftedSlotFillingEnv(Policy):
    def __init__(self, env, safeness=0.5, **kwargs):
        self.safeness = safeness
        self.env = env

    def reset(self):
        pass

    def ask(self, recos_status):
        not_valid_idx = []
        for idx_reco, reco in enumerate(recos_status):
            if reco is None or reco < 0.75:
                not_valid_idx.append(idx_reco)

        if len(not_valid_idx) == 0 :
            return "SUMMARIZE_AND_INFORM"
        else:
            # if np.random.rand() < 0.1:  # force some randomness just to see
            #     return "SUMMARIZE_AND_INFORM"

            idx_ask = np.random.choice(not_valid_idx)
            if np.random.rand() < self.safeness:
                return "ASK_ORAL({})".format(idx_ask)
            else:
                return "ASK_NUM_PAD({})".format(idx_ask)

    def execute(self, s, action_mask=None, info=None):
        turn = s["turn"]
        act = None
        if turn == 0:
            act = self.ask(s["recos_status"])
        else:
            usr_action = s['str_usr_actions'][s["turn"] - 1]
            if usr_action == "DENY_SUMMARIZE":
                min_reco = np.inf
                min_idx_reco = 0
                for ireco, reco in enumerate(s["recos_status"]):
                    if reco is None:
                        min_idx_reco = ireco
                        break
                    else:
                        if reco < min_reco:
                            min_idx_reco = ireco
                            min_reco = reco
                if np.random.rand() < self.safeness:
                    act = "ASK_ORAL({})".format(min_idx_reco)
                else:
                    act = "ASK_NUM_PAD({})".format(min_idx_reco)
            else:
                act = self.ask(s["recos_status"])
        return self.env.system_actions.index(act), False, info


class RandomPolicy(Policy):
    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        actions = []
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                actions.append(i)
        a = np.random.choice(actions)
        return a, False, info


class FittedPolicy(Policy):
    def __init__(self, pi, **kwargs):
        self.pi = pi

    def reset(self):
        pass

    def execute(self, s, actions, info):
        a = self.pi(s, actions)
        return a, False, info

    @classmethod
    def from_config(cls, config):
        config["pi"] = policy_factory(config["pi"])
        return super(PytorchFittedPolicy, cls).from_config(config)


class PytorchFittedPolicy(Policy):
    def __init__(self, env, feature_str, network_path, device, **kwargs):
        self.env = env
        self.feature = feature_factory(feature_str)
        self.device = device
        self.network = None
        if network_path:
            self.load_network(network_path)

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a = self.pi(self.feature(s, self.env), action_mask)
        return a, False, info

    def load_network(self, path):
        logger.info("loading ftq policy at {}".format(path))
        self.network = torch.load(path, map_location=self.device)

    def pi(self, state, action_mask):
        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([[state]], device=self.device, dtype=torch.float)
            a = self.network(s).sub(action_mask).max(1)[1].view(1, 1).item()
            return a


class EpsilonGreedyPolicy(Policy):
    def __init__(self, pi_greedy, epsilon, pi_random=RandomPolicy(), **kwargs):
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.epsilon = epsilon

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        if np.random.random_sample() > self.epsilon:
            a, is_master_action, info = self.pi_greedy.execute(s, action_mask, info)
        else:
            a, is_master_action, info = self.pi_random.execute(s, action_mask, info)

        return a, is_master_action, info

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["pi_greedy"]["env"] = config.get("env", None)
        config["pi_greedy"] = policy_factory(config["pi_greedy"])
        if config["pi_random"]:
            config["pi_random"]["env"] = config.get("env", None)
            config["pi_random"] = policy_factory(config["pi_random"])
        return super(EpsilonGreedyPolicy, cls).from_config(config)


class RandomBudgetedPolicy(Policy):
    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        beta = info["beta"]
        actions = []
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                actions.append(i)
        action_repartition = np.random.random(len(actions))
        action_repartition /= np.sum(action_repartition)
        budget_repartion = generate_random_point_on_simplex_not_uniform(
            coeff=action_repartition,
            bias=beta,
            min_x=0,
            max_x=1)
        index = np.random.choice(a=range(len(actions)),
                                 p=action_repartition)
        a = actions[index]
        beta_ = budget_repartion[index]
        info["beta"] = beta_
        return a, False, info


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, env, feature_str, network_path, betas_for_discretisation, device, hull_options, clamp_Qc=False,
                 **kwargs):
        self.env = env
        self.feature = feature_factory(feature_str)
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_Qc = clamp_Qc
        if network_path:
            self.load_network(network_path)

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a, beta = self.pi(self.feature(s, self.env), info["beta"], action_mask)
        info["beta"] = beta
        return a, False, info

    def load_network(self, network_path):
        logger.info("loading bftq policy at {}".format(network_path))
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def pi(self, state, beta, action_mask):
        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            hull = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                               Q=self.network,
                               action_mask=action_mask,
                               id="run_" + str(state), disp=False,
                               betas=self.betas_for_discretisation,
                               device=self.device,
                               hull_options=self.hull_options,
                               clamp_Qc=self.clamp_Qc)
            opt,_ = optimal_pia_pib(beta=beta, hull=hull,statistic={})
            rand = np.random.random()
            a = opt.id_action_inf if rand < opt.proba_inf else opt.id_action_sup
            b = opt.budget_inf if rand < opt.proba_inf else opt.budget_sup
            return a, b


def policy_factory(config, dynamically=False):
    if "__class__" in config:
        if dynamically:
            path = config['__class__'].split("'")[1]
            module_name, class_name = path.rsplit(".", 1)
            policy_class = getattr(importlib.import_module(module_name), class_name)
            return policy_class.from_config(config)
        elif config['__class__'] == repr(RandomPolicy):
            return RandomPolicy.from_config(config)
        elif config['__class__'] == repr(FittedPolicy):
            return FittedPolicy.from_config(config)
        elif config['__class__'] == repr(PytorchFittedPolicy):
            return PytorchFittedPolicy.from_config(config)
        elif config['__class__'] == repr(EpsilonGreedyPolicy):
            return EpsilonGreedyPolicy.from_config(config)
        elif config['__class__'] == repr(RandomBudgetedPolicy):
            return RandomBudgetedPolicy.from_config(config)
        elif config['__class__'] == repr(PytorchBudgetedFittedPolicy):
            return PytorchBudgetedFittedPolicy.from_config(config)
    else:
        raise ValueError("The configuration should specify the policy __class__")
