import os
import logging

logger = logging.getLogger(__name__)

def makedirs(path):
    # path = os.path.dirname(path)
    if not os.path.exists(path):
        print("create \"{}\"".format(path))
        os.makedirs(path)
    else:
        logger.warning("Can't create \"{}\", folder exists".format(path))

def empty_directory(path_directory):
    os.system("rm -rf {}/*".format(str(path_directory)))