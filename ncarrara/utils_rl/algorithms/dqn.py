# -*- coding: utf-8 -*-
import numpy as np
import torch
import copy
import torch.nn.functional as F

from ncarrara.utils.os import makedirs
from ncarrara.utils.torch_utils import BaseModule, optimizer_factory, loss_fonction_factory
from ncarrara.utils_rl.transition.replay_memory import Memory
from ncarrara.utils_rl.transition.transition import TransitionGym
import logging

logger = logging.getLogger(__name__)


class NetDQN(BaseModule):
    def __init__(self, n_in, n_out, intra_layers, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super(NetDQN, self).__init__(activation_type, reset_type, normalize)
        all_layers = [n_in] + intra_layers + [n_out]
        self.layers = []
        for i in range(0, len(all_layers) - 2):
            module = torch.nn.Linear(all_layers[i], all_layers[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(all_layers[-2], all_layers[-1])

    def forward(self, x):
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)
        return x.view(x.size(0), -1)


class DQN:
    ALL_BATCH = "ALL_BATCH"
    ADAPTATIVE = "ADAPTATIVE"

    def __init__(self,
                 device,
                 policy_network,
                 gamma=0.999,
                 batch_size_experience_replay=128,
                 target_update=10,
                 optimizer=None,
                 loss_function=None,
                 workspace=None,
                 lr=None,
                 weight_decay=None,
                 transfer_module=None,
                 writer=None,
                 **kwargs):
        self.weights_over_time = []
        self.biais_over_time = []
        self.best_fit_over_time = []
        self.ae_errors_over_time = []
        self.probas_over_time = []
        self.writer = writer
        self.tranfer_module = transfer_module
        self.device = device
        self.size_mini_batch = batch_size_experience_replay
        self.GAMMA = gamma
        self.TARGET_UPDATE = target_update
        self.workspace = workspace
        self.policy_net = policy_network.to(self.device)

        self.memory = Memory()
        self.i_episode = 0
        self.n_actions = self.policy_net.predict.out_features
        self.loss_function = loss_fonction_factory(loss_function)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.optimizer = None
        self.reset()

    def save(self, path=None):
        import os
        makedirs(os.path.dirname(path))
        if path is None:
            path = self.workspace / "dqn.pt"
        logger.info("saving dqn at {}".format(path))
        torch.save(self.policy_net, path)
        return path

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self.policy_net.reset()

        if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
            # print(self.policy_net)
            self.policy_net.set_Q_source(self.tranfer_module.get_Q_source(), self.tranfer_module.get_best_error())

        self.optimizer = optimizer_factory(self.optimizer_type,
                                           self.policy_net.parameters(),
                                           self.lr,
                                           self.weight_decay)

        self.i_episode = 0
        self.no_need_for_transfer_anymore = False

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
            self.target_net.set_Q_source(self.policy_net.Q_source, self.tranfer_module.get_best_error())

    def _optimize_model(self):

        transitions = self.memory.sample(
            len(self.memory) if self.size_mini_batch > len(self.memory) else self.size_mini_batch)
        # print("transtions")
        # print(transitions)

        if self.tranfer_module is not None:
            # reeavalute errors
            self.tranfer_module.update()

            # transfert samples
            if self.tranfer_module.is_experience_replay_transfering():
                size_transfer = self.size_mini_batch - len(transitions)
                if size_transfer > 0 and not self.no_need_for_transfer_anymore:
                    transfer_transitions = self.tranfer_module.get_experience_replay_source().sample(size_transfer)
                    transitions = transitions + transfer_transitions
                self.no_need_for_transfer_anymore = size_transfer <= 0

            # transfer Q
            if self.tranfer_module.is_q_transfering():
                self.policy_net.set_Q_source(self.tranfer_module.get_Q_source(), self.tranfer_module.get_best_error())

                self.best_fit_over_time.append(self.tranfer_module.best_fit)

                self.ae_errors_over_time.append(self.tranfer_module.get_best_error())
                if self.writer is not None:
                    self.writer.add_scalar('ae_best_fit/episode', self.tranfer_module.best_fit, self.i_episode)
                    self.writer.add_scalar('ae_error/episode', self.tranfer_module.get_best_error(), self.i_episode)
        batch = TransitionGym(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states = [s for s in batch.s_ if s is not None]

        state_batch = torch.cat(batch.s)
        action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r_)
        Q = self.policy_net(state_batch)
        action_batch = action_batch.unsqueeze(1)
        state_action_values = Q.gather(1, action_batch)
        next_state_values = torch.zeros(len(transitions), device=self.device)
        if non_final_next_states:

            next_state_values[non_final_mask] = self.target_net(torch.cat(non_final_next_states)).max(1)[0].detach()
        else:
            logger.warning("Pas d'état non terminaux")
        self.expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = self.loss_function(state_action_values, self.expected_state_action_values.unsqueeze(1))
        if self.writer is not None:
            self.writer.add_scalar('loss/episode', loss.item(), self.i_episode)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
            weight = self.policy_net.get_weight_ae_layer().item()
            biais = self.policy_net.get_biais_ae_layer().item()
            proba = self.policy_net.get_p().item()
            self.weights_over_time.append(weight)
            self.biais_over_time.append(biais)
            self.probas_over_time.append(proba)
            if self.writer is not None:
                self.writer.add_scalar('ae_weight/episode', weight, self.i_episode)
                self.writer.add_scalar('ae_biais/episode', biais, self.i_episode)
                self.writer.add_scalar('p/episode', proba, self.i_episode)


    def pi(self, state, action_mask):

        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            action_mask[action_mask == 1.] = np.infty
            action_mask = torch.tensor([action_mask], device=self.device, dtype=torch.float)
            s = torch.tensor([state], device=self.device, dtype=torch.float)
            a = self.policy_net(s).squeeze().sub(action_mask).max(1)[1].view(1, 1).item()
            return a

    def update(self, *sample):
        state, action, reward, next_state, done, info = sample
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        if not done:
            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float)
        else:
            next_state = None
        action = torch.tensor([action], device=self.device, dtype=torch.long)
        reward = torch.tensor([float(reward)], device=self.device, dtype=torch.float)
        # print("next_state !!!! ",next_state)
        t = state, action, reward, next_state, done, info
        self.memory.push(*t)
        if self.tranfer_module is not None:
            self.tranfer_module.push(*t)
        self._optimize_model()
        if done:
            self.i_episode += 1
            if self.i_episode % self.TARGET_UPDATE == 0:
                logger.info("[update][i_episode={}] copying weights to target network".format(self.i_episode))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if self.tranfer_module is not None and self.tranfer_module.is_q_transfering():
                    self.target_net.set_Q_source(self.policy_net.Q_source, self.tranfer_module.get_best_error())

                # if self.i_episode % 100 == 0:
                #     with torch.no_grad():
                #         print("Creating histograms ...")
                #         QQ = self.policy_net(self._state_batch)
                #         utils_dqn.create_Q_histograms(
                #             title="Q(s)_pred_target_e={}".format(self.i_episode),
                #             values=[self.expected_state_action_values.cpu().numpy(),
                #                     QQ.gather(1, self._action_batch).cpu().numpy().flat],
                #             path=self.workspace,
                #             labels=["target", "prediction"])
                #
                #         utils_dqn.create_Q_histograms_for_actions(
                #             title="actions_Q(s)_pred_target_e={}".format(self.i_episode),
                #             QQ=QQ.cpu().numpy(),
                #             path=self.workspace,
                #             labels=[str(act) for act in range(self.n_actions)],
                #             mask_action=np.zeros(self.n_actions))
