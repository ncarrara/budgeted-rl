import random
from gym.spaces import Discrete
import numpy as np
from collections import namedtuple
import logging

import re

from ncarrara.utils_rl.environments.slot_filling_env.utils import plot_ctop_cbot

logger = logging.getLogger(__name__)





class SlotFillingEnv(object):
    CONS_ERROR = 0
    CONS_VALID = 1

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    ID = "slot_filling_env_v0"

    def __init__(self, user_params={"cerr": -1, "cok": 1, "ser": 0.3, "cstd": 0.2, "proba_hangup": 0.3},
                 max_turn=15,
                 size_constraints=3,
                 penalty_if_bye=0.,
                 penalty_if_hangup=0.,
                 penalty_if_summarize_fail=50.,
                 penalty_by_turn=-5.,
                 penalty_if_max_turn=0.,
                 reward_if_sucess=100.,
                 can_say_bye=True):
        if logger.getEffectiveLevel() is logging.INFO:
            plot_ctop_cbot(**user_params)
        self.penalty_if_bye = penalty_if_bye
        self.penalty_if_summarize_fail = penalty_if_summarize_fail
        self.penalty_if_max_turn = penalty_if_max_turn
        self.penalty_if_hangup = penalty_if_hangup
        self.pernalty_by_turn = penalty_by_turn
        self.reward_if_sucess = reward_if_sucess

        self.size_constraints = size_constraints
        self.user_params = user_params

        self.max_turn = max_turn

        self.system_actions = ["SUMMARIZE_AND_INFORM"]
        if can_say_bye:
            self.system_actions.append("BYE")

        for cons in range(size_constraints):
            self.system_actions.append("ASK_ORAL({})".format(cons))
        # for cons in range(size_constraints):
        #     self.system_actions.append("ASK_NUM_PAD({})".format(cons))
        logger.info("system actions : {}".format(self.system_actions))
        # print("system actions : {}".format(self.system_actions))

        self.action_space = Discrete(len(self.system_actions))
        self.action_space_str = self.system_actions
        # self.user_actions = ["INFORM", "HANGUP", "DENY_SUMMARIZE"]
        self.user_actions = ["INFORM", "DENY_SUMMARIZE"]


    def _get_user_(self):
        return self.__user

    def action_space_str(self):
        return [str(action) for action in self.system_actions]

    def close(self):
        self.__user.close()

    def score(self):
        return

    def __str__(self):
        return str(self.__user)

    def reset(self):

        self.turn = 0
        self.cons_status = [SlotFillingEnv.CONS_ERROR] * self.size_constraints
        self.recos_status = [None] * self.size_constraints
        self.recos = [None] * self.max_turn
        self.str_sys_actions = [None] * self.max_turn
        self.str_user_actions = [None] * self.max_turn
        observation = {
            "recos_status": self.recos_status.copy(),
            "turn": self.turn,
            "str_sys_actions": self.str_sys_actions.copy(),
            "str_usr_actions": self.str_user_actions.copy()

        }
        return observation

    def step(self, action):
        if self.turn==0:
            logger.info(("--------------- new dialogue -------------"))
        logger.info("[TURN {}] ---------------".format(self.turn))
        logger.info("[TURN {}] cons_status : {}".format(self.turn, self.cons_status))
        logger.info("[TURN {}] recos_status : [ {}]".format(self.turn,
                                                            "".join([("None " if rec is None
                                                                      else "{:.2f} ".format(rec)) for rec in
                                                                     self.recos_status])))
        if type(action) == str:
            str_sys_action = action
        else:
            str_sys_action = self.system_actions[action]
        logger.info("[TURN {}] system : \"{}\"".format(self.turn, str_sys_action))
        str_user_action = None
        success = False
        summarize_failed=False
        if str_sys_action == "SUMMARIZE_AND_INFORM":
            if np.all(np.asarray(self.cons_status) == SlotFillingEnv.CONS_VALID):
                success = True
            else:
                str_user_action = "DENY_SUMMARIZE"
                summarize_failed= True
        elif "ASK_NUM_PAD" in str_sys_action:
            if np.random.rand() < self.user_params['proba_hangup']:
                str_user_action = "HANGUP"
            else:
                str_user_action = "INFORM"
        elif "ASK_ORAL" in str_sys_action:
            str_user_action = "INFORM"
        elif str_sys_action == "BYE":
            pass
        else:
            raise Exception("Unknow action {}".format(str_sys_action))

        if str_user_action == "INFORM":
            m = re.search('\(([0-9]+)\)', str_sys_action)
            if m:
                cons_id = int(m.group(1))
            else:
                raise Exception("Malformed action : {}".format(str_sys_action))

            if "ASK_NUM_PAD" in str_sys_action:
                there_is_an_error = False
                reco = 1.
            elif "ASK_ORAL" in str_sys_action:
                reco, there_is_an_error = self.gen_error_reco(**self.user_params)
            else:
                raise Exception("Malformed action : {}".format(str_sys_action))
            if there_is_an_error:
                self.cons_status[cons_id] = SlotFillingEnv.CONS_ERROR
            else:
                self.cons_status[cons_id] = SlotFillingEnv.CONS_VALID
            self.recos_status[cons_id] = reco

        if success:
            logger.info("[TURN {}] success !".format(self.turn))
            return None, self.reward_if_sucess, True, {"c_": 0.}
        elif str_sys_action == "BYE":
            logger.info("[TURN {}] systeme ended dialogue premarturely !".format(self.turn))
            return None, self.penalty_if_bye, True, {"c_": 0.}
        elif self.turn >= self.max_turn-1:
            logger.info("[TURN {}] max size reached !".format(self.turn))
            return None, self.penalty_if_max_turn, True, {"c_": 0.}
        elif str_user_action == "HANGUP":
            logger.info("[TURN {}] user hang up !".format(self.turn))
            return None, self.penalty_if_hangup, True, {"c_": 1.}
        else:
            logger.info("[TURN {}] user : \"{}\"".format(self.turn, str_user_action))
            self.str_sys_actions[self.turn] = str_sys_action
            self.str_user_actions[self.turn] = str_user_action
            observation = {
                "recos_status": self.recos_status.copy(),
                "turn": self.turn + 1,
                "str_sys_actions": self.str_sys_actions.copy(),
                "str_usr_actions": self.str_user_actions.copy()

            }
            self.turn += 1

            reward = self.pernalty_by_turn
            if summarize_failed:
                reward+=self.penalty_if_summarize_fail

            return observation, reward, False, {"c_": 0.}

    def gen_error_reco(self, cerr, cok, ser, cstd, **kwargs):
        there_is_an_error = np.random.rand() < ser
        reco_err = 1. / (1 + np.exp(-np.random.normal(cerr, cstd)))
        reco_sucess = 1. / (1 + np.exp(-np.random.normal(cok, cstd)))
        reco = reco_err if there_is_an_error else reco_sucess
        return reco.tolist(), there_is_an_error


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    e = SlotFillingEnv(user_params={"cerr": -1, "cok": 1, "ser": 0.5, "cstd": 0.2, "proba_hangup": 0.3})
    e.seed(1)
    from ncarrara.budgeted_rl.tools.policies import HandcraftedSlotFillingEnv

    hdc_policy = HandcraftedSlotFillingEnv(env=e, safeness=1.0)

    for _ in range(10):
        hdc_policy.reset()
        s_ = e.reset()
        done = False
        rr = 0
        while not done:
            s = s_
            a, _, _ = hdc_policy.execute(s)
            s_, r, done, info = e.step(a)
            rr += r
        logger.info("reward : {}".format(rr))
