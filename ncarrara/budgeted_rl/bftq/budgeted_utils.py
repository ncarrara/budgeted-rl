# !/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
from multiprocessing.pool import Pool

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
from scipy.spatial.qhull import ConvexHull, QhullError

from ncarrara.utils.math_utils import near_split
from ncarrara.utils.torch_utils import loss_fonction_factory, get_memory_for_pid
from ncarrara.utils_rl.visualization.toolsbox import create_Q_histograms
from pathlib import Path
import gc

import psutil
import sys

a: int = 5

logger = logging.getLogger(__name__)

OptimalPolicy = namedtuple('OptimalPolicy',
                           ('id_action_inf', 'id_action_sup', 'proba_inf', 'proba_sup', 'budget_inf', 'budget_sup'))

TransitionBFTQ = namedtuple('TransitionBFTQ',
                            ('state', 'action', 'reward', "next_state", 'constraint', 'beta', "hull_id"))

logger = logging.getLogger(__name__)


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())





def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def f(params):
    Qsb_, action_mask, betas_for_discretisation, path, hull_options, clamp_Qc = params
    hull, colinearity, true_colinearity, expection = compute_interest_points_NN_Qsb(
        Qsb=Qsb_,
        action_mask=action_mask,
        betas=betas_for_discretisation,
        disp=False,
        path=path,
        hull_options=hull_options,
        clamp_Qc=clamp_Qc)

    # force to do tolist, so object is pickable, because multiprocessing may freeze when it's not pickable
    # see  https://stackoverflow.com/questions/24537379/python-multiprocessing-script-freezes-seemingly-without-error
    # https://stackoverflow.com/questions/11854519/python-multiprocessing-some-functions-do-not-return-when-they-are-complete-que/11855207#11855207
    return hull.tolist(), colinearity, true_colinearity, expection


def compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=False, path="tmp", id="default",
                                   hull_options=None, clamp_Qc=None):
    with torch.no_grad():

        if clamp_Qc is not None:
            Qsb[:, len(action_mask):] = np.clip(Qsb[:, len(action_mask):],
                                                a_min=clamp_Qc[0],
                                                a_max=clamp_Qc[1])

        if not type(action_mask) == type(np.zeros(1)):
            action_mask = np.asarray(action_mask)
        N_OK_actions = int(len(action_mask) - np.sum(action_mask))

        dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

        if path:
            path = Path(path) / "interest_points"
        colinearity = False
        if disp:
            if not os.path.exists(path):
                os.makedirs(path)

        all_points = np.zeros((N_OK_actions * len(betas), 2))
        all_betas = np.zeros((N_OK_actions * len(betas),))
        all_Qs = np.zeros((N_OK_actions * len(betas),), dtype=int)
        max_Qr = -np.inf
        Qc_for_max_Qr = None
        l = 0
        x = np.zeros((N_OK_actions, len(betas)))
        y = np.zeros((N_OK_actions, len(betas)))
        i_beta = 0
        for ibeta, beta in enumerate(betas):
            QQ = Qsb[ibeta]
            for i_a, mask in enumerate(action_mask):
                i_a_ok_act = 0
                if mask == 1:
                    pass
                else:
                    Qr = QQ[i_a]  # TODO c'est peut etre l'inverse ici
                    Qc = QQ[i_a + len(action_mask)]
                    x[i_a_ok_act][i_beta] = Qc
                    y[i_a_ok_act][i_beta] = Qr
                    if Qr > max_Qr:
                        max_Qr = Qr
                        Qc_for_max_Qr = Qc
                    all_points[l] = (Qc, Qr)
                    all_Qs[l] = i_a
                    all_betas[l] = beta
                    l += 1
                    i_a_ok_act += 1

            i_beta += 1

        if disp:
            for i_a in range(0, N_OK_actions):  # len(Q_as)):
                if action_mask[i_a] == 1:
                    pass
                else:
                    plt.plot(x[i_a], y[i_a], linewidth=6, alpha=0.2)
        k = 0
        points = []
        betas = []
        Qs = []
        for point in all_points:
            Qc, Qr = point
            if not (Qr < max_Qr and Qc >= Qc_for_max_Qr):
                # on ajoute que les points non dominés
                points.append(point)
                Qs.append(all_Qs[k])
                betas.append(all_betas[k])
            k += 1
        if hull_options is not None and hull_options["decimals"] is not None:
            points = np.round(np.array(points), decimals=hull_options["decimals"])
        else:
            points = np.array(points)

        betas = np.array(betas)
        Qs = np.array(Qs)

        # on remove les duplications
        if hull_options is not None and hull_options["remove_duplicated_points"]:
            points, indices = np.unique(points, axis=0, return_index=True)
            betas = betas[indices]
            Qs = Qs[indices]

        if disp:
            plt.rcParams["figure.figsize"] = (5, 5)
            plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=7, color="blue", alpha=0.1)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, color="red")
            plt.grid()

        true_colinearity = False
        expection = False

        if len(points) < 3:
            colinearity = True
            true_colinearity = True
        else:
            if hull_options is None or "library" not in hull_options or hull_options["library"] == "scipy":
                try:
                    if hull_options is not None and hull_options["qhull_options"] is not None:
                        hull = ConvexHull(points, qhull_options=hull_options["qhull_options"])
                    else:
                        hull = ConvexHull(points)
                    vertices = hull.vertices
                except QhullError:
                    colinearity = True
                    expection = True
            elif hull_options is not None and hull_options["library"] == "pure_python":
                if not hull_options["remove_duplicated_points"]:
                    raise Exception("pure_python convexe_hull can't work without removing duplicate points")
                from ncarrara.budgeted_rl.tools.convex_hull_graham import convex_hull_graham
                hull = convex_hull_graham(points.tolist())
                vertices = []
                for vertex in hull:
                    vertices.append(np.where(np.all(points == vertex, axis=1)))
                vertices = np.asarray(vertices).squeeze()
            else:
                raise Exception("Wrong hull options : {}".format(hull_options))
        if colinearity:
            idxs_interest_points = range(0, len(points))  # tous les points d'intéret sont bon a prendre
        else:
            stop = False
            max_Qr = -np.inf
            corr_Qc = None
            max_Qr_index = None
            for k in range(0, len(vertices)):
                idx_vertex = vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qr > max_Qr:
                    max_Qr = Qr
                    max_Qr_index = k
                    corr_Qc = Qc
                else:
                    if Qr == max_Qr and Qc < corr_Qc:
                        max_Qr = Qr
                        max_Qr_index = k
                        corr_Qc = Qc
                        k += 1

                    stop = True
                stop = stop or k >= len(vertices)
            idxs_interest_points = []
            stop = False
            # on va itera a l'envers jusq'a ce que Qc change de sens
            Qc_previous = np.inf
            k = max_Qr_index
            j = 0
            # on peut procéder comme ça car c'est counterclockwise
            while not stop:
                idx_vertex = vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qc_previous >= Qc:
                    idxs_interest_points.append(idx_vertex)
                    Qc_previous = Qc
                else:
                    stop = True
                j += 1
                k = (k + 1) % len(vertices)  # counterclockwise
                if j >= len(vertices):
                    stop = True

        if disp:
            plt.title("interest_points_colinear={}".format(colinearity))
            plt.plot(points[idxs_interest_points, 0], points[idxs_interest_points, 1], 'r--', lw=1, color="red")
            plt.plot(points[idxs_interest_points][:, 0], points[idxs_interest_points][:, 1], 'x', markersize=15,
                     color="tab:pink")
            plt.grid()
            plt.savefig(path / "{}.png".format(id), dpi=300, bbox_inches="tight")
            plt.close()

        rez = np.zeros(len(idxs_interest_points), dtype=dtype)
        k = 0
        for idx in idxs_interest_points:
            Qc, Qr = points[idx]
            beta = betas[idx]
            action = Qs[idx]
            rez[k] = np.array([(Qc, Qr, beta, action)], dtype=dtype)
            k += 1
        if colinearity:
            rez = np.sort(rez, order="Qc")
        else:
            rez = np.flip(rez, 0)  # normalement si ya pas colinearité c'est deja trié dans l'ordre decroissant
        return rez, colinearity, true_colinearity, expection  # betas, points, idxs_interest_points, Qs, colinearity


def compute_interest_points_NN(s, Q, action_mask, betas, device, hull_options, clamp_Qc,
                               disp=False, path=None, id="default"):
    with torch.no_grad():
        ss = s.repeat((len(betas), 1, 1))
        bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
        sb = torch.cat((ss, bb), dim=2)
        Qsb = Q(sb).detach().cpu().numpy()
    return compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=disp, path=path, id=id,
                                          hull_options=hull_options, clamp_Qc=clamp_Qc)


def convex_hull(s, action_mask, Q, disp, betas, device, hull_options, clamp_Qc, path=None, id="default"):
    if not type(action_mask) == type(np.zeros(1)):
        action_mask = np.asarray(action_mask)
    hull, colinearity, true_colinearity, expection = compute_interest_points_NN(
        s=s,
        Q=Q,
        action_mask=action_mask,
        betas=betas,
        device=device,
        disp=disp,
        path=path,
        id=id,
        hull_options=hull_options,
        clamp_Qc=clamp_Qc)
    return hull


def optimal_pia_pib_parralle(args):
    return optimal_pia_pib(*args)


def optimal_pia_pib(beta, hull, statistic):
    with torch.no_grad():
        if len(hull) == 0:
            raise Exception("Hull is empty")
        elif len(hull) == 1:
            Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
            res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
            if beta == Qc_inf:
                status = "exact"
            elif beta > Qc_inf:
                status = "too_much_budget"
            else:
                status = "not_solvable"
        else:
            Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
            if beta < Qc_inf:
                status = "not_solvable"
                res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
            else:
                founded = False
                for k in range(1, len(hull)):
                    Qc_sup, Qr_sup, beta_sup, action_sup = hull[k]
                    if Qc_inf == beta:
                        founded = True
                        status = "exact"  # en realité avec Qc_inf <= beta and beta < Qc_sup ca devrait marcher aussi
                        res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
                        break
                    elif Qc_inf < beta and beta < Qc_sup:
                        founded = True
                        p = (beta - Qc_inf) / (Qc_sup - Qc_inf)
                        status = "regular"
                        res = OptimalPolicy(action_inf, action_sup, 1. - p, p, beta_inf, beta_sup)
                        break
                    else:
                        Qc_inf, Qr_inf, beta_inf, action_inf = Qc_sup, Qr_sup, beta_sup, action_sup
                if not founded:  # we have at least Qc_sup budget
                    status = "too_much_budget"
                    res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf,
                                        0.)  # action_inf = action_sup, beta_inf=beta_sup
        # statistic["status"] = status
        return res, status


class BudgetedUtils():

    def __init__(self,
                 beta_for_discretization,
                 workspace,
                 device,
                 loss_function,
                 loss_function_c,
                 gamma=0.999,
                 gamma_c=1.0,
                 weights_losses=[1., 1.],
                 id="NO_ID"):
        self.device = device
        self.gamma = gamma
        self.gamma_c = gamma_c
        self.weights_losses = weights_losses
        self.loss_function = loss_fonction_factory(loss_function)
        self.loss_function_c = loss_fonction_factory(loss_function_c)
        self.workspace = workspace
        self.betas_for_discretization = beta_for_discretization
        self.id = id
        raise Exception("Must be reviewed, way less computational efficient than classic BFTQ")

    def reset(self):
        # stateless
        pass

    def change_id(self, id):
        self.id = id

    def loss(self, Q, Q_target, batch, targets_must_be_zeros=False):
        SB, S, A, R, C, NS, H, B, mask_unique_hull_ns, mask_not_terminal_ns = batch
        N_batch = SB.shape[0]
        with torch.no_grad():
            ns_r = torch.zeros(N_batch, device=self.device)
            ns_c = torch.zeros(N_batch, device=self.device)
            if not targets_must_be_zeros:
                ns_r, ns_c = self.compute_next_values(Q_target, NS, H, B, mask_unique_hull_ns, mask_not_terminal_ns)

            label_r = R + (self.gamma * ns_r)
            label_c = C + (self.gamma_c * ns_c)
        loss = self._compute_loss(Q, SB, A, label_r, label_c, )
        return loss

    def _compute_loss(self, Q, sb_batch, a_batch, label_r, label_c):
        sb_batch = sb_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        label_r = label_r.to(self.device)
        label_c = label_c.to(self.device)
        output = Q(sb_batch)
        state_action_rewards = output.gather(1, a_batch)
        state_action_constraints = output.gather(1, a_batch + self.N_actions)
        loss_Qc = self.loss_function_c(state_action_constraints, label_c.unsqueeze(1))
        loss_Qr = self.loss_function(state_action_rewards, label_r.unsqueeze(1))
        w_r, w_c = self.weights_losses
        loss = w_c * loss_Qc + w_r * loss_Qr
        return loss

    def compute_next_values(self, Q, ns_batch, h_batch, b_batch, where_unique_hull_ns, where_not_terminal_ns):
        next_state_rewards = torch.zeros(self.size_batch, device=self.device)
        next_state_constraints = torch.zeros(self.size_batch, device=self.device)

        if len(where_not_terminal_ns) > 0:
            with torch.no_grad():
                ################################################
                # computing all unique hulls (no terminal next_state and no repetition among next state)
                ################################################
                self.track_memory("compute hulls")
                self.info("computing hulls ...")

                # gather all s,beta to compute in one foward pass (may increase the pic of memory used temporary)
                # but essential for multiprocessing hull computing bellow

                ns_batch_unique = ns_batch[where_unique_hull_ns]

                self.info("There are {} hulls to compute !".format(len(ns_batch_unique)))

                """ 
                sb = "duplicating each next states with each beta"
                if we have 2 states s0 and s1 and 2 betas 0 and 1. What do we want for sb is :
                (s0,0)
                (s0,1)
                (s1,0)
                (s1,1)
                """
                ss = ns_batch_unique \
                    .squeeze() \
                    .repeat((1, len(self.betas_for_discretisation))) \
                    .reshape(-1) \
                    .reshape((len(ns_batch_unique) * len(self.betas_for_discretisation), self.size_state))
                bb = torch.from_numpy(self.betas_for_discretisation).float().unsqueeze(1).to(device=self.device)
                bb = bb.repeat((len(ns_batch_unique), 1))
                sb = torch.cat((ss, bb), dim=1)
                sb = sb.unsqueeze(1)

                self.track_memory("Qsb (compute_hull) ")
                # self.info("Forward pass of couple (s',beta). Size of the batch : {}." +
                #           "It should be equals to #hulls({}) x #beta_for_discretisation({})  : {}"
                #           .format(len(sb), len(ns_batch_unique), len(self.betas_for_discretisation),
                #                   len(self.betas_for_discretisation) * len(ns_batch_unique)))
                num_bins = self.split_batches
                batch_sizes = near_split(x=len(sb), num_bins=num_bins)
                y = []
                self.info("Splitting x in minibatch to avoid out of memory")
                self.info("Batch sizes : {}".format(batch_sizes))
                offset = 0
                for i in range(num_bins):
                    self.info("mini batch {}".format(i))
                    self.track_memory("mini_batch={}".format(i))
                    mini_batch = sb[offset:offset + batch_sizes[i]]
                    offset += batch_sizes[i]
                    y.append(Q(mini_batch))
                    torch.cuda.empty_cache()
                Qsb = torch.cat(y)

                Qsb = Qsb.detach().cpu().numpy()
                self.track_memory("Qsb (compute_hull) (end)")

                args_for_ns_batch_unique = [
                    (
                        Qsb[(i_ns_unique * len(self.betas_for_discretisation)):
                            (i_ns_unique + 1) * len(self.betas_for_discretisation)],
                        np.zeros(self.N_actions),
                        self.betas_for_discretisation,
                        str(self.workspace),
                        self.hull_options,
                        self.clamp_Qc
                    )
                    for i_ns_unique in range(len(ns_batch_unique))
                ]

                if self.cpu_processes == 1:
                    hulls_for_ns_batch_unique = []
                    colinearities = []
                    true_colinearities = []
                    exceptions = []
                    for i_params, params in enumerate(args_for_ns_batch_unique):
                        if i_params % max(1, len(args_for_ns_batch_unique) // 10) == 0:
                            self.info("{} hulls processed (sequentially)".format(i_params))
                        hull, colinearity, true_colinearity, expection = f(params)
                        hulls_for_ns_batch_unique.append(hull)
                else:
                    self.info("Using multiprocessing")
                    with Pool(self.cpu_processes) as p:
                        # p.map return ordered fashion, so we're cool
                        rez = p.map(f, args_for_ns_batch_unique)
                        hulls_for_ns_batch_unique, colinearities, true_colinearities, exceptions = zip(*rez)

                self.info("exceptions : {:.4f} %".format(
                    np.sum(np.array(exceptions)) / len(hulls_for_ns_batch_unique) * 100.))
                self.info("true_colinearities : {:.4f} %".format(
                    np.sum(np.array(true_colinearities)) / len(hulls_for_ns_batch_unique) * 100.))
                self.info("colinearities : {:.4f} %".format(
                    np.sum(np.array(colinearities)) / len(hulls_for_ns_batch_unique) * 100.))
                self.info("#next_states : {}".format(len(ns_batch)))
                self.info("#non terminal next_states : {}".format(len(where_not_terminal_ns)))
                self.info("#hulls actually computed : {}".format(len(hulls_for_ns_batch_unique)))
                self.info("computing hulls [DONE] ")
                self.empty_cache()
                self.track_memory("compute hulls (end)")

                #####################################
                # for each next_state
                # computing optimal distribution among 2 actions,
                # with respectively 2 budget
                # given the hull of the next_state and the beta (in current state).
                #####################################

                self.track_memory("compute_opts")
                self.info("computing ops ... ")

                ##############################################################
                # we build couples (s',beta\top) and (s',beta\bot)
                # in order to compute Q(s')
                ##############################################################
                next_state_beta_not_terminal = torch.zeros((len(where_not_terminal_ns) * 2, 1, self.size_state + 1),
                                                           device=self.device)
                ns_batch_not_terminal = ns_batch[where_not_terminal_ns]
                h_batch_not_terminal = h_batch[where_not_terminal_ns]
                b_bath_not_terminal = b_batch[where_not_terminal_ns]

                args = [(beta.detach().item(), hulls_for_ns_batch_unique[hull_id], {})
                        for hull_id, beta in zip(h_batch_not_terminal, b_bath_not_terminal)]

                if self.cpu_processes == 1:
                    opts_and_statuses = []
                    for i_arg, arg in enumerate(args):
                        if i_arg % (max(1, len(args) // 10)) == 0:
                            self.info("{} opts proccessed (sequentially)".format(i_arg))
                        opts_and_statuses.append(optimal_pia_pib_parralle(arg))
                else:
                    self.info("computing optimal_pia_pib in parralle ...")
                    with Pool(self.cpu_processes) as p:
                        opts_and_statuses = p.map(optimal_pia_pib_parralle, args)
                    self.info("computing optimal_pia_pib in parralle ... done")

                self.info("computing opts ... end")
                self.track_memory("compute_opts (end)")
                status = {"regular": 0, "not_solvable": 0, "too_much_budget": 0, "exact": 0}

                for i_ns_nt, (ns_not_terminal, opt_and_status) in enumerate(
                        zip(ns_batch_not_terminal, opts_and_statuses)):
                    opt, stat = opt_and_status
                    status[stat] += 1
                    ns_beta_moins = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_inf]], device=self.device)),
                                              dim=1)
                    ns_beta_plus = torch.cat((ns_not_terminal, torch.tensor([[opt.budget_sup]], device=self.device)),
                                             dim=1)
                    next_state_beta_not_terminal[i_ns_nt * 2 + 0][0] = ns_beta_moins
                    next_state_beta_not_terminal[i_ns_nt * 2 + 1][0] = ns_beta_plus

                ##############################################################
                # Forwarding to compute the Q function in s' #################
                ##############################################################

                self.info("Q next")
                self.track_memory("Q_next")
                Q_next_state_not_terminal = Q(next_state_beta_not_terminal)
                Q_next_state_not_terminal = Q_next_state_not_terminal.detach()
                self.track_memory("Q_next (end)")
                self.info("Q next end")
                self.empty_cache()

                ###########################################
                ############  bootstraping   ##############
                ###########################################

                self.info("computing next values ...")
                self.track_memory("compute_next_values")

                warning_qc_negatif = 0.
                offset_qc_negatif = 0.
                warning_qc__negatif = 0.
                offset_qc__negatif = 0.
                next_state_c_neg = 0.
                offset_next_state_c_neg = 0.

                next_state_rewards_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)
                next_state_constraints_not_terminal = torch.zeros(len(where_not_terminal_ns), device=self.device)

                for i_ns_nt in range(len(where_not_terminal_ns)):
                    opt = opts_and_statuses[i_ns_nt][0]
                    qr_ = Q_next_state_not_terminal[i_ns_nt * 2][opt.id_action_inf]
                    qr = Q_next_state_not_terminal[i_ns_nt * 2 + 1][opt.id_action_sup]
                    qc_ = Q_next_state_not_terminal[i_ns_nt * 2][self.N_actions + opt.id_action_inf]
                    qc = Q_next_state_not_terminal[i_ns_nt * 2 + 1][self.N_actions + opt.id_action_sup]

                    if qc < 0.:
                        warning_qc_negatif += 1.
                        offset_qc_negatif = qc

                    if qc_ < 0.:
                        warning_qc__negatif += 1.
                        offset_qc__negatif = qc_

                    next_state_rewards_not_terminal[i_ns_nt] = opt.proba_inf * qr_ + opt.proba_sup * qr
                    next_state_constraints_not_terminal[i_ns_nt] = opt.proba_inf * qc_ + opt.proba_sup * qc

                    if next_state_constraints_not_terminal[i_ns_nt] < 0:
                        next_state_c_neg += 1.
                        offset_next_state_c_neg = next_state_constraints_not_terminal[i_ns_nt]

                next_state_rewards[where_not_terminal_ns] = next_state_rewards_not_terminal
                next_state_constraints[where_not_terminal_ns] = next_state_constraints_not_terminal

                if logger.getEffectiveLevel() <= logging.DEBUG:
                    self.info("printing some graphs in next_values ...")
                    self.info("\n[compute_next_values] Q(s') sur le batch")
                    create_Q_histograms("Qr(s')_e={}".format(self._id_ftq_epoch),
                                        values=next_state_rewards.cpu().numpy().flatten(),
                                        path=self.workspace / "histogram",
                                        labels=["next value"])
                    create_Q_histograms("Qc(s')_e={}".format(self._id_ftq_epoch),
                                        values=next_state_constraints.cpu().numpy().flatten(),
                                        path=self.workspace / "histogram",
                                        labels=["next value"])
                    self.info("printing some graphs in next_values ... done")

                mean_qc_neg = 0 if warning_qc_negatif == 0 else offset_qc_negatif / warning_qc_negatif
                mean_qc__neg = 0 if warning_qc__negatif == 0 else offset_qc__negatif / warning_qc__negatif
                mean_ns_neg = 0 if next_state_c_neg == 0 else offset_next_state_c_neg / next_state_c_neg
                self.info("qc < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    warning_qc_negatif / len(where_not_terminal_ns) * 100., mean_qc_neg))
                self.info("qc_ < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    warning_qc__negatif / len(where_not_terminal_ns) * 100., mean_qc__neg))
                self.info("next_state_constraints < 0 percentage {:.2f}% with a mean offset of {:.4f}".format(
                    next_state_c_neg / len(where_not_terminal_ns) * 100., mean_ns_neg))
                self.info("compute next values ... end")
                self.empty_cache()
                self.track_memory("compute_next_values (end)")

        return next_state_rewards, next_state_constraints

    def empty_cache(self):
        memoire_before = get_memory_for_pid(os.getpid())
        torch.cuda.empty_cache()
        memoire_after = get_memory_for_pid(os.getpid())
        self.info("empty cache {} -> {}".format(self.format_memory(memoire_before), self.format_memory(memoire_after)))

    def format_memory(self, memoire):
        for _ in range(len(self.devices) - len(memoire)):
            memoire.append(0)
        accolades = "".join(["{:05} " for _ in range(len(self.devices))])
        accolades = accolades[:-1]
        format = "[m=" + accolades + "]"
        return format.format(*memoire)

    def get_current_memory(self):
        memory = get_memory_for_pid(os.getpid())

        # self.memory_tracking.append(sum)
        return memory

    def track_memory(self, id):
        sum = 0
        for mem in self.get_current_memory():
            sum += mem
        self.memory_tracking.append([id, sum])

    def info(self, message):
        memoire = self.get_current_memory()

        if self._id_ftq_epoch is not None:
            logger.info("[e={:02}]{} {}".format(self._id_ftq_epoch, self.format_memory(memoire), message))
        else:
            logger.info("{} {} ".format(self.format_memory(memoire), message))
