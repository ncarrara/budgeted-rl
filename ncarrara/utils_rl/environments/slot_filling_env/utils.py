import logging
import numpy as np

import logging

logger = logging.getLogger(__name__)


def generate_proba_hangup(mean, std, k=0):
    xx = np.random.normal(mean, std)
    if xx < 0.:
        if k > 5:
            return 0.
        else:
            return generate_proba_hangup(mean / 2., std / 2., k=k + 1)
    if xx > 1.:
        if k > 5:
            return 1.0
        else:
            return generate_proba_hangup(mean + (1. - mean / 2.), std / 2., k=k + 1)
    return xx

def sucess(cok=1, std=0.2):
    reco_sucess = 1. / (1 + np.exp(-np.random.normal(cok, std)))
    return reco_sucess


def error(cerr=-1, std=0.2):
    reco_err = 1. / (1 + np.exp(-np.random.normal(cerr, std)))
    return reco_err


def plot_ctop_cbot(cerr, cok, cstd, **kwargs):
    import matplotlib.pyplot as plt
    succ = [sucess(cok, cstd) for _ in range(0, 100000)]
    err = [error(cerr, cstd) for _ in range(0, 100000)]
    plt.hist(succ,500,alpha=0.50)
    plt.hist(err,500,alpha=0.50)
    plt.title("choose ctop, cbottom")
    plt.savefig("plot_ctop_cbot")
    plt.show()
    plt.close()


# plot_ctop_cbot(-0.25,0.25,0.1)
# plot_ctop_cbot(-0.25,0.25,0.25)
for i in np.linspace(0.1,0.5,9):
    plot_ctop_cbot(-0.25,0.25,i)
# plot_ctop_cbot(-0.25,0.25,0.75)