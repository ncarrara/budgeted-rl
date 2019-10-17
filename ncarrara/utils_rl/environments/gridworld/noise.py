#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import math


# Apply some noise to movement from x,y to x+ax, y+ay
def apply_noise(x, y, ax, ay, std, noise_type):
    stdx, stdy = std[0], std[1]
    xp, yp = 0, 0

    if noise_type == "gaussian":
        xp = np.random.normal(x + float(ax), stdx, 1)[0]
        yp = np.random.normal(y + float(ay), stdy, 1)[0]

    elif noise_type == "normalized_gaussian":
        dx, dy = ax, ay
        r = np.sqrt(dx ** 2 + dy ** 2)
        dx += np.random.normal(0, stdx, 1)[0]
        dy += np.random.normal(0, stdy, 1)[0]
        d = np.sqrt(dx ** 2 + dy ** 2)
        dx *= r / d
        dy *= r / d

        xp = x + dx
        yp = y + dy

    elif noise_type == "gaussian_bis":
        r = 1.
        rotation = 0
        if ax == 1. and ay == 0.:
            rotation = 0.
        elif ax == 0. and ay == 1.:
            rotation = np.pi / 2.
        elif ax == -1. and ay == 0.:
            rotation = np.pi
        elif ax == 0. and ay == -1.:
            rotation = 3 * np.pi / 2.
        else:
            raise Exception("Invalid action for this type of noise")

        theta = rotation + np.random.normal(0, stdx, 1)[0]
        # np.pi/2.#np.random.normal(0, stdx, 1)[0]
        xp = x + r * np.cos(theta)
        yp = y + r * np.sin(theta)

    elif noise_type == "uniform":
        rand = np.random.uniform(size=1)[0]
        xp = x + ax
        yp = y + ay
        # dx = xp - x
        # dy = yp - y
        # if rand < stdx:
        #     # on rotate a droite
        #     xp = x - dy
        #     yp = y + dx
        # elif rand < stdx + stdy:
        #     # on rotate a gauche
        #     xp = x + dy
        #     yp = y - dx
        # else:
        #     pass
        # translation vers l'origine
        x0 = xp - x
        y0 = yp - y
        if rand < stdx:
            # on rotate de -pi/2
            x0_ = x0 * 0 + y0 * 1
            y0_ = y0 * 0 - x0 * 1
        elif rand < stdx + stdy:
            # on rotate de pi/2
            x0_ = x0 * 0 - y0 * 1
            y0_ = y0 * 0 + x0 * 1
        else:
            x0_ = x0
            y0_ = y0
        xp = x0_ + x
        yp = y0_ + y

    else:
        xp, yp = apply_creepy_noise(x, y, ax, ay, std, noise_type)

    return xp, yp


# Apply a creepy/messy noise to movement from x,y to x+dy, y+dy
# You'better look away for its a terrible mess...
def apply_creepy_noise(x, y, ax, ay, std, noise_type):
    stdx, stdy = std[0], std[1]

    if noise_type == "test_death_trap":
        if ax == 0. and ay == 1.:
            xp = 0.5
            yp = 1.5
        else:
            if random.random() < 0.5:
                xp = 1.5
                yp = 0.5
            else:
                xp = 1.5
                yp = 1.5

    elif noise_type == "3xWidth":
        # seulement pour la grille 3xWidth
        xp = x + ax
        yp = y + ay
        if ax == 1 and ay == 0:
            rand = np.random.uniform(size=1)[0]
            dx = xp - x
            dy = yp - y
            if rand < 0.5:
                # on rotate a droite
                xp = x - dy
                yp = y + dx

    elif noise_type == "poele":
        if ax == 0 and ay == 1:
            if x == 0.5 and y == 0.5:
                xp = x + ax
                yp = y + ay
            elif x == 1.5 and y == 0.5:
                rand = np.random.uniform(size=1)[0]
                if rand < 0.01:
                    xp = 1.5
                    yp = 1.5
                else:
                    xp = x
                    yp = y
            elif ax == 1 and ay == 0:
                if x == 0.5 and y == 0.5:
                    xp = x + ax
                    yp = y + ay
                elif x == 1.5 and y == 0.5:
                    rand = np.random.uniform(size=1)[0]
                    if rand < 0.01:
                        xp = 1.5
                        yp = 1.5
                    else:
                        xp = x
                        yp = y
                else:
                    raise Exception("state : " + str(x) + " " + str(y) + "not possible for poele a frire")
            else:
                raise Exception("action not allowed for poele a frire")

    elif noise_type == "poele2":
        if ax == 0 and ay == 1 and x == 0.5 and y == 0.5:
            xp = 0.5
            yp = 1.5
        elif ax == 1 and ay == 0 and x == 0.5 and y == 0.5:
            rand = np.random.uniform(size=1)[0]
            if rand < 0.9:
                xp = 1.5
                yp = 1.5
            else:
                xp = 1.5
                yp = 0.5
        else:
            raise Exception("impossible case for " + noise_type)

    else:
        raise Exception("noise_type \"" + noise_type + "\" unknow")
    return xp, yp
