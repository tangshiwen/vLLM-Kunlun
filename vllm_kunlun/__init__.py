"""vllm kunlun init"""
from .platforms import current_platform
import sys
import importlib
import warnings
import builtins
import os
import time
import vllm.envs as envs
OLD_IMPORT_HOOK = builtins.__import__
def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        module_mappings = {
            "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
            "vllm.v1.worker.utils": "vllm_kunlun.v1.worker.utils",
            "vllm.model_executor.model_loader.bitsandbytes_loader": "vllm_kunlun.models.model_loader.bitsandbytes_loader",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
            "vllm.model_executor.layers.sampler": "vllm_kunlun.ops.sample.sampler",
            "vllm.v1.sample.ops.topk_topp_sampler": "vllm_kunlun.v1.sample.ops.topk_topp_sampler",
        }

        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
            return module

        relative_mappings = {
            ("compressed_tensors_moe", "compressed_tensors"): "vllm_kunlun.ops.quantization.compressed_tensors_moe",
            ("layer", "fused_moe"): "vllm_kunlun.ops.fused_moe.layer",
        }

        if level == 1:
            parent = globals.get('__package__', '').split('.')[-1] if globals else ''
            key = (module_name, parent)
            if key in relative_mappings:
                if module_name in sys.modules:
                    return sys.modules[module_name]
                target_module = relative_mappings[key]
                module = importlib.import_module(target_module)
                sys.modules[module_name] = module
                sys.modules[target_module] = module
                return module

    except Exception:
        pass

    return OLD_IMPORT_HOOK(
        module_name,
        globals=globals,
        locals=locals,
        fromlist=fromlist,
        level=level
    )

def import_hook():
    """Apply import hook for VLLM Kunlun"""
    if not int(os.environ.get("DISABLE_KUNLUN_HOOK", "0")):
        builtins.__import__ = _custom_import
        
        try:
            modules_to_preload = [
                "vllm_kunlun.ops.quantization.compressed_tensors_moe",
                "vllm_kunlun.ops.fused_moe.custom_ops",
                "vllm_kunlun.ops.fused_moe.layer",
                "vllm_kunlun.ops.quantization.fp8",
            ]
            for module_name in modules_to_preload:
                importlib.import_module(module_name)
        except Exception:
            pass

def register():
    """Register the Kunlun platform"""
    from .utils import redirect_output
    from .vllm_utils_wrapper import direct_register_custom_op, patch_annotations_for_schema
    import_hook()
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"

def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg
    _reg()