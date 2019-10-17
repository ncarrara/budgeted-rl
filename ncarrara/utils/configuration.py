import json
import os
from pathlib import Path
from ncarrara.utils.os import makedirs, empty_directory


class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        import logging
        self.logger = logging.getLogger(__name__)
        self.dict = {}
        self.device = None
        self.plt = None
        self.backend=None

    def __getitem__(self, arg):
        self.__check__()
        return self.dict[arg]

    def has_key(self,key):
        return key in self.dict

    def __contains__(self, key):
        return key in self.dict

    def __str__(self):
        import pprint
        return pprint.pformat(self.dict)

    def __check__(self):
        if not self.dict:
            raise Exception("please load the configuration file")

    def create_fresh_workspace(self,force=False):
        r = ''
        while r!='y' and r!='n':
            if force:
                r='y'
            else:
                r = input("are you sure you want to erase workspace {} [y/n] ?".format(self.workspace))
            from ncarrara.utils.os import makedirs
            if r=='y':
                self.__check__()
                self.__clean_workspace()

            elif  r=='n':
                makedirs(self.workspace)
            else:
                print("Only [y/n]")
        return self

    def __clean_workspace(self):
        self.__check__()
        import os
        os.system("rm -rf {}".format(self.workspace))
        return self

    def load(self, config):
        if type(config) == type(""):
            self.logger.info("reading config file at {}".format(config))
            with open(config, 'r') as infile:
                import json
                self.dict = json.load(infile)
        elif type(config) == type({}):
            self.dict = config
        else:
            raise TypeError("Wrong type for configuration, must be a path or a dict")

        self.id = self.dict["general"]["id"]
        self.workspace = Path(self.dict["general"]["workspace"])
        print(self.dict["general"])

        import logging.config as config
        config.dictConfig(self.dict["general"]["dictConfig"])

        self.seed = self.dict["general"]["seed"]
        if self.seed is not None:
            from ncarrara.utils.math_utils import set_seed
            set_seed(self.seed)

        import numpy as np
        np.set_printoptions(precision=2,suppress=True)

        self.writer = None
        self.is_tensorboardX = self["general"]["is_tensorboardX"]

        return self

    def load_matplotlib(self, backend=None):
        if self.plt is not None:
            self.logger.warning("matplotlib already loaded")
        else:
            self.backend = None
            import matplotlib
            if backend is not None:
                self.backend = backend
            elif self["general"]["matplotlib_backend"] is not None:
                self.backend =self["general"]["matplotlib_backend"]
            if self.backend is not None:
                matplotlib.use(self.backend)
            import matplotlib.pyplot as plt
            self.plt = plt
        return self

    def load_pytorch(self, override_device_str=None):

        self.logger.warning("Using import torch.multiprocessing as multiprocessing")
        self.logger.warning("Using multiprocessing.set_start_method('spawn')")
        import torch.multiprocessing as multiprocessing
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError as e:
            self.logger.warning(str(e))


        if self.device is not None:
            self.logger.warning("pytorch already loaded")
        else:
            if override_device_str is not None:
                import torch
                _device = torch.device(override_device_str)
            else:
                from ncarrara.utils.torch_utils import get_the_device_with_most_available_memory
                _device = get_the_device_with_most_available_memory()
            self.device = _device
            self.logger.info("DEVICE : {}".format(self.device))

        return self

    def load_tensorboardX(self):
        if self.is_tensorboardX:
            from tensorboardX import SummaryWriter
            empty_directory(self.workspace / "tensorboard")
            makedirs(self.workspace / "tensorboard")
            # exit()

            self.writer = SummaryWriter(str(self.workspace / "tensorboard"))
            command = "tensorboard --logdir {} --port 6009 &".format(str(self.workspace / "tensorboard"))
            self.logger.info("running command \"{}\"".format(command))
            os.system(command)

    def dump_to_workspace(self, filename="config.json"):
        """
        Dump the configuration a json file in the workspace.
        """
        makedirs(self.workspace)
        print(self.dict)
        with open(self.workspace / filename, 'w') as f:
            json.dump(self.dict, f, indent=2)
