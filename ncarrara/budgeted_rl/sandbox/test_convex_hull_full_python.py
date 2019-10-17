import numpy as np
from scipy.spatial import ConvexHull

from ncarrara.budgeted_rl.bftq.pytorch_budgeted_fittedq import compute_interest_points_NN_Qsb

a = np.array([
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13],
    [-9.39e-23, 3.51e-19],
    [5.83e-15, 9.34e-02],
    [2.32e-16, - 1.64e-18],
    [1.59e-30, - 7.92e-13]]
)

# z, indexes = np.unique(a,axis=0,return_index=True)

from ncarrara.budgeted_rl.tools.convex_hull_graham import convex_hull_graham

a = np.random.random_sample((100,2))

compute_interest_points_NN_Qsb(a, [0], np.linspace(0, 1, len(a)), disp=True, path="tmp", id="pure_python",
                               hull_options={
                                   "decimals": None,
                                   "qhull_options": None,
                                   "remove_duplicated_points": True,
                                   "library": "pure_python"
                               })

compute_interest_points_NN_Qsb(a, [0], np.linspace(0, 1, len(a)), disp=True, path="tmp", id="scipy",
                               hull_options={
                                   "decimals": None,
                                   "qhull_options": None,
                                   "remove_duplicated_points": True,
                                   "library": "scipy"
                               })

# hull = convex_hull_graham(a.tolist())
# for p in hull:
#     print(p)

# hull = ConvexHull((np.unique(a,axis=0)))
# hull = ConvexHull(a)
