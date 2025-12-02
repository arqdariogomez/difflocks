import os
import torch

# --- CONFIGURACIÓN DE COMPILACIÓN ---
# Desactivado para estabilidad en T4
use_compile = False

def get_use_compile():
    return use_compile

def compile_wrap(function):
    return function

# --- CONFIGURACIÓN DE CHECKPOINTING ---
_checkpointing = False

def checkpointing(enable=True):
    global _checkpointing
    _checkpointing = enable

def get_checkpointing():
    global _checkpointing
    return _checkpointing
