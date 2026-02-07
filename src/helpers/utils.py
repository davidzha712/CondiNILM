#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Utils

#
#################################################################################################################

import os


def create_dir(path):
    os.makedirs(path, exist_ok=True)

    return path
