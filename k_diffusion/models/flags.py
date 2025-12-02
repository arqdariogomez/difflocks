import os
import torch

# Desactivamos compilación para evitar crashes en T4 (INTERNAL ASSERT FAILED)
def compile_wrap(function):
    return function

use_compile = False
