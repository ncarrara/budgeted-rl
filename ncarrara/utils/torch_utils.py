import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import logging
import os
import re
logger = logging.getLogger(__name__)
from subprocess import PIPE, run

torch.set_printoptions(profile="short")

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.split()]
    return gpu_memory

def get_memory_for_pid(pid):
    command="nvidia-smi"
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout
    m=re.findall("\| *[0-9] *"+str(pid)+" *C *.*python.*? +([0-9]+).*\|",result,re.MULTILINE)
    return [int(mem) for mem in m]


def get_the_device_with_most_available_memory():#use_cuda_visible_devices=False):
    if str(torch.__version__) == "0.4.1.":
        logger.warning("0.4.1. is bugged regarding mse loss")
    logger.info("Pytorch version : {}".format(torch.__version__))

    if not torch.cuda.is_available():
        device_str = "cpu"
    else:
        memory_map = get_gpu_memory_map()
        device_id = 0
        min = np.inf
        for k, v in enumerate(memory_map):
            logger.info("device={} memory used={}".format(k, v))
            # print type(v)
            if v < min:
                device_id = k
                min = v
        torch.cuda.set_device(device_id)
        device_str = "cuda:{}".format(device_id)
        # import torch
        # if use_cuda_visible_devices :
        #     # this seems to be the correct way to do it
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        #     device = "cuda"
        # else:
        #     # but this one is 2x faster when calling module.to(device)
        #     device = "cuda:{}".format(device)

    # print("importing torch ...")
    # import torch
    # print("done ...")
    # exit()

    device = torch.device(device_str)
    logger.info("device with most available memory: {}".format(device))
    return device

def loss_fonction_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return  F.l1_loss
    elif loss_function == "bce":
        return  F.binary_cross_entropy
    else:
        raise Exception("Unknown loss function : {}".format(loss_function))

def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - activation factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super(BaseModule, self).__init__()
        self.activation = BaseModule.activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias.data, 0.)

    @staticmethod
    def activation_factory(activation_type):
        if activation_type == "RELU":
            return F.relu
        elif activation_type == "TANH":
            return torch.tanh
        else:
            raise Exception("Unknown activation_type: {}".format(activation_type))

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *input):
        return NotImplementedError
