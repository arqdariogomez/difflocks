import os
import torch

# --- 1. COMPILACIÓN (JIT) ---
# Desactivado para evitar crash "INTERNAL ASSERT FAILED" en T4
use_compile = False

def get_use_compile():
    return use_compile

def compile_wrap(function):
    return function

# --- 2. CHECKPOINTING (Gradient Checkpointing) ---
_checkpointing = False

def checkpointing(enable=True):
    global _checkpointing
    _checkpointing = enable

def get_checkpointing():
    global _checkpointing
    return _checkpointing

# --- 3. FLASH ATTENTION 2 ---
# Controla si se intenta usar FA2. Lo ponemos en False por defecto para seguridad,
# ya que en attention.py tenemos nuestro propio fallback manual.
_use_flash_attention_2 = False

def use_flash_attention_2(enable=True):
    global _use_flash_attention_2
    _use_flash_attention_2 = enable

def get_use_flash_attention_2():
    global _use_flash_attention_2
    return _use_flash_attention_2
