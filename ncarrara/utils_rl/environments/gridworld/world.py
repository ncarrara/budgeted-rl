#!/usr/bin/env python
import math
import cairocffi as cairo
import numpy as np
import logging


class World():
    def __init__(self, e, betas=None):  # ,options={"pi": None, "actions": None, "id": "default_id"}):

        # self.pi = options["pi"]
        # self.id = options["id"]
        # self.actions = options["actions"]
        self.model = e
        self.betas = betas
        self.w_model = e.w
        self.h_model = e.h
        self.w_txt_line, self.h_txt_line = (250, 16.66)  # size_txt_line
        self.w_block = self.w_txt_line
        self.h_block = self.h_txt_line * 8
        L = 1 if self.betas is None else len(self.betas)
        self.h_nb_blocks = np.ceil(np.sqrt(L * self.w_block / float(self.h_block)))
        self.w_nb_blocks = np.ceil(L / np.sqrt(L * self.w_block / float(self.h_block)))
        self.dh = self.h_nb_blocks * self.h_block + self.h_txt_line * 2
        self.dw = self.w_nb_blocks * self.w_block
        self.WIDTH = self.dw * self.w_model
        self.HEIGHT = self.dh * self.h_model
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(self.WIDTH), int(self.HEIGHT))
        self.ctx = cairo.Context(self.surface)

    def change_size(self, pixel_width, pixel_height):
        err_w = pixel_width / self.WIDTH
        err_h = pixel_height / self.HEIGHT
        self.WIDTH = pixel_width
        self.HEIGHT = pixel_height
        self.w_txt_line *= err_w
        self.h_txt_line *= err_h
        self.w_block *= err_w
        self.h_block *= err_h
        self.dw *= err_w
        self.dh *= err_h
        self.resize(self.WIDTH, self.HEIGHT)

    def draw_frame(self):
        self.ctx.rectangle(0, 0, self.WIDTH, self.HEIGHT)
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.fill()
        self.ctx.rectangle(0, 0, self.WIDTH + 100, self.HEIGHT + 100)
        self.ctx.set_source_rgb(0.0, 0.0, 0.0)
        self.ctx.fill()

    def draw_rectangle(self, rect, color):
        x1, y1, x2, y2 = rect
        r, g, b = color
        w = (x2 - x1) * self.dw
        h = (y2 - y1) * self.dh
        self.ctx.rectangle(x1 * self.dw, y1 * self.dh, w, h)
        self.ctx.set_source_rgb(r, g, b)
        self.ctx.fill()

    def format_p(self, p):
        str = ""
        for proba in p:
            if proba < 0.001:
                proba = 0
            str += "{0:.2f}".format(proba) + " "
        str = str[:len(str) - 1]
        return str

    def format_A_str(self, p):
        str = ""
        for proba in p:
            str += proba + "      "
        str = str[:len(str) - 1]
        return str

    def format_q(self, q, s, A, distrib_bn):
        str = ""
        for i in range(0, len(A)):
            str += "{0:.2f}".format(q(s, A[i], distrib_bn[i])) + " "
        str = str[:len(str) - 1]
        return str

    def format_q_ftq(self, q, s, A):
        str = ""
        for i in range(0, len(A)):
            str += "{0:.2f}".format(q(s, A[i])) + " "
        str = str[:len(str) - 1]
        return str

    def format_C(self, C, s, A):
        str = ""
        for a in A:
            str += "{0:.2f}".format(C(s, a)) + " "
        str = str[:len(str) - 1]
        return str

    def draw_trajectory(self, trajectory, rgba, line_width):
        self.ctx.stroke()
        r, g, b, a = rgba
        self.ctx.set_source_rgba(r, g, b, a)
        self.ctx.set_line_width(line_width)
        for sample in trajectory:
            s, _, _, sp, _, _ = sample
            sx, sy = s
            spx, spy = sp
            if sx != spx or sy != spy:
                self.ctx.move_to(sx * self.dw, sy * self.dh)
                self.ctx.line_to(spx * self.dw, spy * self.dh)
                self.ctx.move_to(spx * self.dw, spy * self.dh)
            else:
                self.ctx.arc(sx * self.dw, sy * self.dh, 20, 0, 2 * math.pi)
                self.ctx.move_to(sx * self.dw, sy * self.dh)
                # self.ctx.fill()
        self.ctx.stroke()

    def draw_lattice(self):
        self.ctx.set_source_rgb(1, 1, 1)
        for i in np.arange(0., self.w_model + 1, 1.):
            self.ctx.move_to(i * self.dw, 0)
            self.ctx.line_to(i * self.dw, self.h_model * self.dh)

        for i in np.arange(0., self.h_model + 1, 1.):
            self.ctx.move_to(0, i * self.dh)
            self.ctx.line_to(self.w_model * self.dw, i * self.dh)

    def draw_policy_bftq(self, pi, qr, qc, bs):
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.set_font_size(self.h_txt_line * 0.9)
        for i in np.arange(0.5, self.w_model, 1.):
            for j in np.arange(0.5, self.h_model, 1.):
                # i, j = (0.5, 0.5)
                s = (i, j)
                x_case = (i - 0.47) * self.dw
                y_case = (j - 0.5) * self.dh + 3 * self.h_txt_line
                self.ctx.move_to(x_case, y_case - 2 * self.h_txt_line)
                # self.ctx.show_text("A : " + self.format_A_str(A_str))
                x, y = None, None
                col = 0
                for k in range(0, len(bs)):
                    # print k
                    if k % self.h_nb_blocks == 0:
                        x = x_case + col * self.w_block
                        y = y_case
                        # print "youhou", x, y
                        col += 1
                    self.draw_block_bftq(pi, qr, qc, bs, k, x, y, s)
                    y = (y + self.h_block)
        self.ctx.stroke()

    def draw_policy_ftq(self, A, A_str, qr):
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.set_font_size(self.h_txt_line * 0.9)
        for i in np.arange(0.5, self.w_model, 1.):
            for j in np.arange(0.5, self.h_model, 1.):
                # i, j = (0.5, 0.5)
                s = (i, j)
                x_case = (i - 0.47) * self.dw
                y_case = (j - 0.5) * self.dh + 3 * self.h_txt_line
                self.ctx.move_to(x_case, y_case - 2 * self.h_txt_line)
                self.ctx.show_text("A : " + self.format_A_str(A_str))
                # print "case orgini:", x_case, y_case
                x, y = x_case, y_case
                col = 0
                self.draw_block_ftq(A, qr, x, y, s)
                y = (y + self.h_block)
        self.ctx.stroke()

    def draw_block_bftq(self, pi, qr, qc, bs, k, x, y, s):
        self.ctx.move_to(x, y)
        b = bs[k]
        # distrib, distrib_bn = pi(s, b)
        opt = pi(s, b)
        current_line = 0
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        # www2 = np.array([qr(s, A[a], distrib_bn[a]) for a in range(0, len(A))])
        www2 = np.array([qr(s, opt.id_action_inf, opt.budget_inf), qr(s, opt.id_action_sup, opt.budget_sup)])
        # www3 = np.array([qc(s, A[a], distrib_bn[a]) for a in range(0, len(A))])
        www3 = np.array([qc(s, opt.id_action_inf, opt.budget_inf), qc(s, opt.id_action_sup, opt.budget_sup)])
        self.ctx.set_source_rgb(1, 1, 0)
        # self.ctx.show_text("[b=" + "{0:.2f}".format(b) + "][p.b=" + "{0:.2f}".format(
        #     np.dot(distrib, distrib_bn)) + "]")
        self.ctx.show_text("[b=" + "{0:.2f}".format(b) + "][p.b=" + "{0:.2f}".format(
            opt.budget_inf * opt.proba_inf + opt.budget_sup * opt.proba_sup) + "]")
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        pqr = np.dot([opt.proba_inf, opt.proba_sup], www2)
        pqc = np.dot([opt.proba_inf, opt.proba_sup], www3)
        # print pqr, pqc
        self.ctx.show_text("[p.Qr={0:.2f}]".format(pqr) + "[p.Qc={0:.2f}]".format(pqc))
        # self.ctx.move_to(x, y + current_line * self.h_txt_line)
        # current_line += 1
        # self.ctx.show_text("[p.Qc = " + "{0: .2f}".format(np.dot(distrib, www3)) + "]")
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        self.ctx.show_text(
            "A: " + self.model.actions_str[opt.id_action_inf] + " " + self.model.actions_str[opt.id_action_sup])
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        self.ctx.show_text("p: " + self.format_p([opt.proba_inf, opt.proba_sup]))
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        self.ctx.show_text("b: " + self.format_p([opt.budget_inf, opt.budget_sup]))
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        self.ctx.show_text("Qr: " + str(
            ["{:.2f} ".format(ww) for ww in www2]))  # self.format_q(qr, s, A, [opt.proba_inf,opt.proba_sup]))
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        current_line += 1
        self.ctx.show_text("Qc: " + str(
            ["{:.2f} ".format(ww) for ww in www3]))  # self.format_q(qc, s, A, [opt.proba_inf,opt.proba_sup]))
        self.ctx.move_to(x, y + current_line * self.h_txt_line)

    def draw_block_ftq(self, A, qr, x, y, s):
        self.ctx.move_to(x, y)
        current_line = 0
        self.ctx.move_to(x, y + current_line * self.h_txt_line)
        self.ctx.set_source_rgb(1, 1, 1)
        current_line += 1
        self.ctx.show_text("Qr: " + self.format_q_ftq(qr, s, A))
        self.ctx.move_to(x, y + current_line * self.h_txt_line)

    def draw_goal(self, goal):
        rect, reward, constraint, isabsorbing = goal
        x1, y1, x2, y2 = rect
        self.draw_rectangle(rect, (0.17, 0.63, 0.17))
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.set_font_size(20)
        self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.3) * self.dh)
        self.ctx.show_text("R={:.2f}".format(reward))
        # self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.5) * self.dh)
        # self.ctx.show_text("C={:.2f}".format(constraint))

    def draw_wall(self, rect):
        x1, y1, x2, y2 = rect
        self.draw_rectangle(rect, (0.5, 0.5, 0.5))

    def draw_hole(self, hole):
        rect, reward, constraint, isabsorbing = hole
        x1, y1, x2, y2 = rect
        self.draw_rectangle(rect, (0.84, 0.15, 0.16))
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.set_font_size(20)
        self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.3) * self.dh)
        self.ctx.show_text("C={:.2f}".format(constraint))
        # self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.5) * self.dh)
        # self.ctx.show_text("C={:.2f}".format(constraint))

    def draw_mixte(self, hole):
        rect, reward, constraint, isabsorbing = hole
        x1, y1, x2, y2 = rect
        self.draw_rectangle(rect, (1.0, 0.5, 0.05))
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.set_font_size(20)
        self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.3) * self.dh)
        self.ctx.show_text("R={:.2f}".format(reward))
        self.ctx.move_to((x1 + 0.02) * self.dw, (y1 + 0.5) * self.dh)
        self.ctx.show_text("C={:.2f}".format(constraint))

    def draw_case(self, case):
        rect, reward, constraint, isabsorbing = case
        if reward > 0. and constraint == 0.:
            self.draw_goal(case)
        elif constraint > 0. and reward ==0.:
            self.draw_hole(case)
        else:
            self.draw_mixte(case)

    def draw_cases(self):
        for case in self.model.cases:
            self.draw_case(case)

        for case in self.model.blocks:
            self.draw_wall(case)

    def draw_bftq(self, source_trajectories=None, test_trajectories=None, A=None, A_str=None, qr=None, qc=None,
                  bs=None,
                  pi=None, C=None):
        for case in self.model.cases:
            self.draw_case(case)
        if source_trajectories is not None:
            self.draw_source_trajectories(source_trajectories)
        if test_trajectories is not None:
            self.draw_test_trajectories(test_trajectories)
        self.draw_policy_bftq(pi, qr, qc, bs)
        self.draw_lattice()

    def draw_test_trajectory(self, trajectory, alpha):
        self.draw_trajectory(trajectory, (0.12, 0.47, 0.71, alpha), 3)

    def draw_source_trajectory(self, trajectory, alpha=0.3):
        self.draw_trajectory(trajectory, (0.58, 0.40, 0.74, alpha), 3)

    def draw_source_trajectories(self, trajectories, alpha=1.0):
        for trajectory in trajectories:
            self.draw_source_trajectory(trajectory, alpha)

    def draw_test_trajectories(self, trajectories, alpha=1.0):
        for trajectory in trajectories:
            self.draw_test_trajectory(trajectory, alpha)

    def save(self, filename):
        file = filename+'.png'
        # logging.info("[WORLD] saving to {}".format(file))
        self.ctx.close_path()
        self.ctx.stroke()
        self.surface.write_to_png(file)
        return file
