# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.budgeted_rl.bftq.budgeted_utils import BudgetedUtils, TransitionBudgeted
from ncarrara.utils.torch import optimizer_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

from ncarrara.utils_rl.visualization.toolsbox import fast_create_Q_histograms_for_actions

logger = logging.getLogger(__name__)


class PytorchBudgetedDQN():

    def __init__(self,

                 policy_net,
                 beta_for_discretisation,
                 workspace,
                 device,
                 loss_function,
                 loss_function_c,
                 optimizer_params={"learning_rate": 0.01,
                                   "weight_decay": 0.0,
                                   "optimizer_type": "ADAM"},
                 gamma=0.999,
                 gamma_c=1.0,
                 weights_losses=[1., 1.],
                 target_update=10,
                 size_minibatch=32,
                 state_to_unique_str="id",
                 action_to_unique_str="id",
                 actions_str=None
                 ):
        # TODO refaire proprement avec factory et functions
        if state_to_unique_str == "str":
            state_to_unique_str = lambda a: str(a)
        else:
            raise Exception("Unknown state_to_unique_str {}".format(state_to_unique_str))
        if action_to_unique_str == "str":
            action_to_unique_str = lambda a: str(a)
        else:
            raise Exception("Unknown action_to_unique_str {}".format(action_to_unique_str))

        # pattern, par composition (on ne veut pas d héritage pour les algos)
        self.budgeted_utils = BudgetedUtils(beta_for_discretization=beta_for_discretisation,
                                            workspace=workspace,
                                            device=device,
                                            loss_function_c=loss_function_c,
                                            loss_function=loss_function,
                                            id="NO_ID")
        self.device = device
        self.optimizer_params = optimizer_params
        self.state_to_unique_str = state_to_unique_str
        self.action_to_unique_str = action_to_unique_str
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.size_mini_batch = size_minibatch
        self.target_update = target_update
        self.weights_losses = weights_losses
        self.policy_net = policy_net
        self.memory = Memory(class_transition=TransitionBudgeted)
        self.i_episode = 0
        self.hulls_key_id = {}
        self.N_actions = policy_net.predict.out_features // 2
        logger.info('actions_str={}'.format(self.actions_str))
        logger.info('N_actions={}'.format(self.N_actions))
        self.reset()

    def reset(self):
        torch.cuda.empty_cache()
        self.budgeted_utils.reset()
        self.memory.reset()
        self.policy_net.reset()
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self._policy_network.parameters(),
                                           self.learning_rate,
                                           self.weight_decay)

        # print(self.optimizer_params)
        # self.optimizer = optimizer_factory(params=self.policy_net.parameters(), **self.optimizer_params)
        self.i_episode = 0
        self.current_hull_id = 0
        self.i_update = 0
        self.hulls_key_id = {}

    def _construct_batch(self):
        transitions = self.memory.sample(
            len(self.memory) if self.size_mini_batch > len(self.memory) else self.size_mini_batch)

        zipped = TransitionBudgeted(*zip(*transitions))
        A = torch.cat(zipped.action).to(self.device)
        R = torch.cat(zipped.reward).to(self.device)
        C = torch.cat(zipped.constraint).to(self.device)

        B = torch.cat(zipped.beta).to(self.device)
        S = torch.cat(zipped.state).to(self.device)
        NS = torch.cat(zipped.next_state).to(self.device)
        H = torch.cat(zipped.hull_id).to(self.device)
        SB = torch.cat((S, B), dim=2).to(self.device)

        _, mask_unique_hull_ns = np.unique(H, return_index=True)  # todo convertion numpy pytorch et cie
        mask_not_terminal_ns = np.where(NS != None)

        return SB, S, A, R, C, NS, H, B, mask_unique_hull_ns, mask_not_terminal_ns

    def _optimize_model(self):
        batch = self._construct_batch()
        loss = self.budgeted_utils.loss(Q=self.policy_net, Q_target=self.target_net, batch=batch)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def pi(self, state, beta, action_mask):
        return self.budgeted_utils.policy(self.policy_net, state, beta, action_mask)

    def update(self, *sample):
        self.budgeted_utils.change_id(self.i_update)
        state, action, reward, next_state, constraint, beta, done, info = sample
        state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        if not done:
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float).unsqueeze(0)
        else:
            next_state = None
        action = torch.tensor(action, device=self.device, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(float(reward), device=self.device, dtype=torch.float).unsqueeze(0)
        constraint = torch.tensor(float(constraint), device=self.device, dtype=torch.float).unsqueeze(0)
        beta = torch.tensor(float(beta), device=self.device, dtype=torch.float).unsqueeze(0)

        hull_key = self.state_to_unique_str(next_state)
        if hull_key in self.hulls_key_id:
            hull_id = self.hulls_key_id[hull_key]
        else:
            hull_id = self.current_hull_id
            self.current_hull_id += 1
        hull_id = torch.tensor(hull_id, device=self.device, dtype=torch.int).unsqueeze(0)
        self.memory.push(state, action, reward, next_state, constraint, beta, hull_id)
        self._optimize_model()
        if next_state is None:
            self.i_episode += 1
            if self.i_episode % self.target_update == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                # if logger.getEffectiveLevel() is logging.INFO:
                #     if self.i_episode % 10 == 0:
                #         with torch.no_grad():
                #             logger.info("Creating histograms ...")
                #             zipped = TransitionBudgeted(*zip(*self.memory.memory))
                #             state_batch = torch.cat(zipped.state)
                #             beta_batch = torch.cat(zipped.beta_batch)
                #             state_beta_batch = torch.cat((state_batch, beta_batch), dim=2)
                #             QQ = self.policy_net(state_beta_batch)
                #             QQr = QQ[:, 0:self.N_actions]
                #             QQc = QQ[:, self.N_actions:2 * self.N_actions]
                #             mask_action = np.zeros(self.N_actions)
                #             fast_create_Q_histograms_for_actions(
                #                 title="actions_Qr(s)_pred_target_e={}".format(self.i_episode),
                #                 QQ=QQr.cpu().numpy(),
                #                 path=self.workspace / "histogram",
                #                 labels=self.actions_str,
                #                 mask_action=mask_action)
                #             fast_create_Q_histograms_for_actions(
                #                 title="actions_Qc(s)_pred_target_e={}".format(self.i_episode),
                #                 QQ=QQc.cpu().numpy(),
                #                 path=self.workspace / "histogram",
                #                 labels=self.actions_str,
                #                 mask_action=mask_action)

        self.i_update += 1
