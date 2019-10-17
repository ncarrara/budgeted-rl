import logging

logger = logging.getLogger(__name__)


def feature_factory(feature_str):
    if feature_str == "feature_pydial":
        return feature_pydial
    elif feature_str == "feature_gaussian":
        return feature_gaussian
    elif feature_str == "feature_simple":
        return feature_simple
    elif feature_str == "feature_identity":
        return feature_identity
    else:
        return eval(feature_str)


def feature_pydial(s, e):
    N_actions = e.action_space.n
    if s is None:
        return None
    else:
        # pprint.pprint(s)
        summary_acts = s["system_summary_acts"]
        master_acts = s["system_master_acts"]
        user_acts = s["user_acts"]

        flatten = s["flatten"]
        last_act_onehot = [0.] * (N_actions)  # + 1)
        repeat_one_hot = [0.] * 11

        # what is my last summary act
        if not master_acts:
            pass
        else:
            last_act_onehot[e.action_space_str.index(summary_acts[-1])] = 1.

        # how many I ask twice the same thing
        concecutive_repetion = 0
        if not master_acts:
            pass
        else:
            i = 0
            while i < len(master_acts) - 1:
                if (master_acts[i] == master_acts[i + 1] and (
                        "repeat" not in user_acts[i])):  # or "null" in acts[i] or "badact" in acts[i]:
                    concecutive_repetion += 1
                i += 1
            repeat_one_hot[concecutive_repetion] = 1.

        rez = flatten + last_act_onehot + repeat_one_hot

        return rez


import numpy as np

import itertools


def lambda_rbf_quick(s, centers, sigma):
    exp = -1. / 2. * np.power(np.linalg.norm(np.array(s) - np.array(centers), axis=1) / sigma, 2)
    gauss = (1. / (sigma * np.sqrt(2. * np.pi))) * np.exp(exp)
    return gauss


def create_coding(n):
    return np.array([seq for seq in itertools.product((0., 0.75, 0.5, 0.25, 1.), repeat=n)])


def feature_basic(s, e):
    if s is None:
        return None
    else:
        reco = feat_reco_status(s, e)
        min_reco = [0] * len(reco)
        min_reco[np.argmin(np.asarray(reco))] = 1.
        return min_reco + reco + feat_turn(s, e) + feat_usr_act(s, e) + feat_sys_act(s, e)


def feature_simple(s, e):
    return feature_slot_filling(s, e, False)


def feature_gaussian(s, e):
    return feature_slot_filling(s, e, True)


def feat_turn(s, e):
    one_hot_turn = [0] * e.max_turn
    one_hot_turn[s["turn"]] = 1
    return one_hot_turn


def feat_sys_act(s, e):
    one_hot_sys_act = [0.] * len(e.system_actions)
    if s["turn"] == 0:
        pass
    else:
        one_hot_sys_act[e.system_actions.index(s["str_sys_actions"][s["turn"] - 1])] = 1.
    return one_hot_sys_act


def feat_usr_act(s, e):
    one_hot_usr_act = [0.] * len(e.user_actions)
    if s["turn"] == 0:
        pass
    else:
        one_hot_usr_act[e.user_actions.index(s["str_usr_actions"][s["turn"] - 1])] = 1.
    return one_hot_usr_act


def feat_reco_status(s, e):
    if s["turn"] == 0:
        recos_status = [0.] * e.size_constraints
    else:
        recos_status_numpy = np.nan_to_num(np.array(s["recos_status"], dtype=np.float))
        recos_status = recos_status_numpy.tolist()
    return recos_status


def feature_slot_filling(s, e, gaussian_reco):
    logger.info("[feature_slot_filling] ----------------------")
    if s is None:
        logger.info("\n\nNone\n")
        logger.info("[feature_slot_filling] None")
        return None
    else:
        import pprint
        logger.info("\n\n" + pprint.pformat(s) + "\n")

        recos_status = feat_reco_status(s, e)

        if gaussian_reco:
            coding = create_coding(e.size_constraints)
            if s["turn"] == 0:
                feat_reco = [0.] * len(coding)
            else:

                feat_reco = lambda_rbf_quick(np.asarray(recos_status), coding, 0.5).tolist()
        else:
            one_hot_min_reco = [0.] * e.size_constraints
            min_value = np.inf
            min_index = None
            for ireco, reco in enumerate(s["recos_status"]):
                if reco is None or reco < min_value:
                    min_value = reco
                    min_index = ireco
                    if reco is None:
                        break

            if min_index is not None:
                one_hot_min_reco[min_index] = 1.
                # min_reco[min_index] *= recos_status[min_index]

            all_slots_asked = 1.
            for reco in s["recos_status"]:
                if reco is None:
                    all_slots_asked = 0
                    break

            feat_reco = [all_slots_asked] + one_hot_min_reco + recos_status

        one_hot_turn = feat_turn(s, e)
        one_hot_usr_act = feat_usr_act(s, e)
        one_hot_sys_act = feat_sys_act(s, e)

        feat = feat_reco + one_hot_usr_act + one_hot_turn  # + one_hot_sys_act + one_hot_turn
        logger.info("[feature_slot_filling] feat_reco usr_act={} sys_act={} turn={}"
                    .format("".join(["{:.2f} ".format(x) for x in feat_reco]),
                            one_hot_usr_act,
                            one_hot_sys_act,
                            one_hot_turn))

        return feat


def feature_identity(s, e):
    return s.flatten().tolist()


if __name__ == "__main__":
    coding = np.array(create_coding(3))
    print(coding)
    print(len(coding))
    s = np.array([0., 0.5, 0.])
    feat_reco = lambda_rbf_quick(s, coding, 0.1).tolist()
    for i, code in enumerate(coding):
        print(s, code, feat_reco[i])
