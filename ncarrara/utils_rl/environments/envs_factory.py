from sklearn.model_selection import ParameterGrid
import gym
import logging

logger = logging.getLogger(__name__)


def generate_env_params_combination(env_params):
    for k, v in env_params.items():
        if type(v) is str:
            import numpy as np
            x = eval(v)
            if isinstance(x,np.ndarray):
                x=x.tolist()
            env_params[k] = x
            logger.info("[gepc] <{}> {} -> {}".format(k, v, x))
    grid = ParameterGrid(env_params)
    return grid


def generate_env_params_random(env_params, number_random_env):
    import numpy as np
    grid = []
    for ienv in range(0, number_random_env):
        params = {}
        for k, v in env_params.items():
            if isinstance(v, list):
                params[k] = np.random.choice(np.linspace(v[0], v[1], 1000), 1)[0].tolist()
            elif isinstance(v, float) or isinstance(v, int) or isinstance(v, str):
                params[k] = v
            else:
                raise Exception("This type can't be parsed : {}".format(type(v)))
        grid.append(params)
    return grid


def generate_envs(envs_str, envs_params, number_random_env=None):
    logger.info(
        "[generate_envs] params : \n\n{}".format("".join(["\t{} : {}\n".format(k, v) for k, v in envs_params.items()])))
    if number_random_env is None:
        grid = generate_env_params_combination(envs_params)
    else:
        grid = generate_env_params_random(envs_params, number_random_env)
    logger.info("[generate_envs] number of envs : {}".format(len(grid)))

    envs = []
    rez_params = []
    for param in grid:
        if envs_str == "CartPoleConfig-v0":
            from ncarrara.utils_rl.environments.cart_pole_config_env import CartPoleConfigEnv
            env = CartPoleConfigEnv(**param)
        elif envs_str == "MountainCarConfig-v0":
            from ncarrara.utils_rl.environments.mountain_car_config_env import MountainCarConfigEnv
            env = MountainCarConfigEnv(**param)
        elif envs_str == "LunarLanderConfig-v0":
            from ncarrara.utils_rl.environments.lunar_lander_config_env import LunarLanderConfigEnv
            env = LunarLanderConfigEnv(**param)
        elif envs_str == "LunarLanderBudgetedConfig-v0":
            from ncarrara.utils_rl.environments.lunar_lander_budgeted_config_env import LunarLanderBudgetedConfigEnv
            env = LunarLanderBudgetedConfigEnv(**param)
        elif envs_str == "gym_pydial":
            env_pydial = __import__("gym_pydial.env.env_pydial")
            env = env_pydial.EnvPydial(pydial_logging_level="ERROR", **param)
        elif envs_str == "slot_filling_env_v0":
            from ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env import SlotFillingEnv
            env = SlotFillingEnv(**param)
        elif envs_str == "slot_filling_env_v1":
            from ncarrara.utils_rl.environments.slot_filling_env.slot_filling_env import SlotFillingEnv
            user_params = {
                "cerr": param["cerr"],
                "cok": -param["cerr"],
                "ser": param["ser"],
                "cstd": param["cstd"],
                "proba_hangup": param["proba_hangup"]
            }
            param_bis = {**param}
            del param_bis["cerr"],param_bis["ser"], param_bis["cstd"], param_bis["proba_hangup"]
            env = SlotFillingEnv(user_params=user_params, **param_bis)
        elif envs_str == "highway-v0":
            __import__("highway_env")
            env = gym.make(envs_str)
            env.configure(dict(**param))
        elif envs_str == "test_death_trap":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_death_trap
            env, _ = generate_test_death_trap()
        elif envs_str == "3xWidth":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_3xWidth
            env, _ = generate_3xWidth(**param)
        elif envs_str == "continuous-3xWidth":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_continuous3xWidth
            env, _ = generate_continuous3xWidth(**param)
        elif envs_str == "test4":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_4
            env, _ = generate_test_4()
        elif envs_str == "test0":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_0
            env, _ = generate_test_0()
        elif envs_str == "test1":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_1
            env, _ = generate_test_1()
        elif envs_str == "test2":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_test_2
            env, _ = generate_test_2()
        elif envs_str == "generate_safe_explo":
            from ncarrara.utils_rl.environments.gridworld.model_generator import generate_safe_explo
            env, _ = generate_safe_explo(**param)
        elif envs_str == "double_path":
            from ncarrara.utils_rl.environments.gridworld.model_generator import double_path
            env, _ = double_path(**param)
        elif envs_str == "omega":
            from ncarrara.utils_rl.environments.gridworld.model_generator import omega
            env, _ = omega(**param)
        else:
            env = gym.make(envs_str)
            for k, v in param.items():
                setattr(env, k, v)
        envs.append(env)
        rez_params.append(param)
    logger.info("[generate_envs] actual params : \n\n{}".format("".join(
        ["{}\n".format("".join(["\t{} : {}\n".format(k, v) for k, v in param.items()])) for param in rez_params])))
    # for param in rez_params:
    #     logger.info("".join(["\t{} : {}\n".format(k, v) for k, v in param.items()]))
    return envs, rez_params


def get_actions_str(env):
    if hasattr(env, "action_str"):
        actions_str = env.action_str
    elif hasattr(env, "ACTIONS"):
        actions_str = [env.ACTIONS[a] for a in range(env.action_space.n)]
    else:
        actions_str = [str(a) for a in range(env.action_space.n)]
    return actions_str
