#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

from gym.spaces import Discrete

from ncarrara.utils_rl.environments.gridworld.geometry import inRectangle
from ncarrara.utils_rl.environments.gridworld.noise import apply_noise


class EnvGridWorld(object):
    CARDINAL_ACTIONS = [(0., 0.), (0., 1.), (1., 0.), (-1., 0), (0., -1.)]
    CARDINAL_ACTIONS_STR = ["X", "v", ">", "<", "^"]

    def action_space(self):
        return self.actions

    def action_space_str(self):
        return self.actions_str

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # def random_state(self):
    #     return (random.random() * self.w, random.random() * self.h)

    def __init__(self, dim, std, cases, trajectoryMaxSize, walls_around, noise_type="gaussian_bis", id="default_id",
                 penalty_on_move=0, actions=CARDINAL_ACTIONS, actions_str=CARDINAL_ACTIONS_STR, init_s=(0.5, 0.5),
                 cost_on_move=0,blocks=[]):
        # random.seed(seed)
        # np.random.seed(seed)
        # self.seed = seed
        # List des actions possible et leur description textuelle
        self.actions = actions
        self.actions_str = actions_str
        self.blocks= blocks

        # identifiant de l'environement
        self.id = id

        # taille de la grille
        w, h = dim
        self.walls_around = walls_around
        self.w = float(w)
        self.h = float(h)

        # niveau et type de bruit
        self.std = (0., 0.)  # no noise
        if std is not None:
            self.std = std
        self.stdx, self.stdy = self.std
        self.noise_type = noise_type

        # liste des case specifiques de la grille
        self.cases = cases

        # reward/cout par defaut a chaque mouvement
        self.penalty_on_move = penalty_on_move
        self.cost_on_move = cost_on_move

        # attribut des trajectoires
        self.trajectoryMaxSize = trajectoryMaxSize
        self.current_case = None
        self.init_s = init_s
        # print self.cases

        self.action_space = Discrete(len(self.actions))
        self.action_space_str = actions_str

        self.reset()

    def reset(self):
        self.s = self.init_s
        self.t = 0
        self.ended = False;
        return np.array(self.s)

    def step(self, i_a):
        a = self.actions[i_a]
        if self.ended:
            raise Exception('game is ended')

        x, y = self.s
        s = (x, y)
        rp = 0.
        cp = 0.

        # on verifie qu on est pas dans une case absorbante
        for case in self.cases:
            rectangle, r, c, is_absorbing = case
            if inRectangle(s, rectangle):
                if is_absorbing:
                    cp = 0.
                    rp = 0.
                    self.ended = True
                break
        # on verifie qu'on est pas dans un mur
        if self.walls_around and (x < 0 or x > self.w or y < 0 or y > self.h):
            print("Boum!")
            cp = 0.
            rp = 0.
            if x < 0: x = -x
            if x > self.w: x = 2 * self.w - x
            if y < 0: y = -y
            if y > self.h: y = 2 * self.h - y

        # on se deplace
        ax, ay = a
        if not self.ended and not (ax == 0. and ay == 0.):
            xp, yp = apply_noise(x, y, ax, ay, self.std, self.noise_type)
        else:
            xp, yp = x, y

        sp = (xp, yp)

        block = None
        for wall in self.blocks:
            if inRectangle((xp, yp), wall):
                # on bouge pas
                block=wall
                break

        if (x == xp and y == yp):  # on a rien fait
            rp = 0.
            cp = 0.
            sp = (x, y)
        elif self.walls_around and block is None:
            # # On rebondit sur les murs
            if xp < 0: xp = -xp
            if yp < 0: yp = -yp
            if xp >= self.w: xp = 2 * self.w - xp
            if yp >= self.h: yp = 2 * self.h - yp
            # on bouge pas
            # sp = s
            sp = (xp, yp)
        elif block is not None:
            sp = (x,y)
        else:
            raise Exception("impossible")

        for case in self.cases:
            rectangle, r, c, is_absorbing = case
            if inRectangle(sp, rectangle):
                rp = r
                cp = c
                break
        s = self.s
        self.s = sp
        self.t += 1

        # if cp >0:
        #     print cp
        info = {"c_": cp}

        info["state_is_absorbing"] = False
        # print "--------------"
        # print "sp :", sp
        if (ax == 0. and ay == 0.):
            info["state_is_absorbing"] = True
            self.ended = True
        else:
            for case in self.cases:
                rectangle, r, c, is_absorbing = case
                if inRectangle(sp, rectangle):
                    info["state_is_absorbing"] = is_absorbing
                    self.ended = is_absorbing
                    break
        # if is_absorbing:
        #     observation = None
        # else:
        # observation = np.array(sp),
        self.ended = self.ended or self.t >= self.trajectoryMaxSize

        rp = rp - self.penalty_on_move
        observation, reward, done, info = np.array(sp), rp, self.ended, info

        return observation, reward, done, info
        # return t
