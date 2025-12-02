import os
import torch

# --- CONFIGURACIÓN DE COMPILACIÓN (FIX T4 CRASH) ---
# Desactivamos torch.compile porque causa "INTERNAL ASSERT FAILED" en T4
def compile_wrap(function):
    return function

use_compile = False

# --- FLAGS DE CHECKPOINTING (RESTAURADOS) ---
# Estas funciones son requeridas por models/__init__.py
_checkpointing = False

def checkpointing(enable=True):
    global _checkpointing
    _checkpointing = enable

def get_checkpointing():
    global _checkpointing
    return _checkpointing
