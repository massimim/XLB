import os

from xlb.compute_backend import ComputeBackend
from dataclasses import dataclass


@dataclass
class DefaultConfig:
    default_precision_policy = None
    velocity_set = None
    default_backend = None


def init(velocity_set, default_backend, default_precision_policy):
    DefaultConfig.velocity_set = velocity_set
    DefaultConfig.default_backend = default_backend
    DefaultConfig.default_precision_policy = default_precision_policy

    if default_backend == ComputeBackend.WARP:
        import warp as wp
        wp.init()
    elif default_backend == ComputeBackend.JAX:
        check_multi_gpu_support()
    elif default_backend == ComputeBackend.NEON:
        import warp as wp
        import wpne
        import py_neon as ne
        from py_neon import Index_3d
        from py_neon.dense import dSpan


        # Get the path of the current script
        script_path = __file__
        # Get the directory containing the script
        script_dir = os.path.dirname(os.path.abspath(script_path)) +'/../'

        wp.config.mode = "debug"
        wp.config.llvm_cuda = False
        wp.config.verbose = True
        wp.verbose_warnings = True

        wp.init()

        wp.build.set_cpp_standard("c++17")
        wp.build.add_include_directory(script_dir)
        wp.build.add_include_directory(script_dir+"/../testing/Neon/")
        wp.build.add_include_directory(script_dir+"/../testing/")
        wp.build.add_preprocessor_macro_definition('NEON_WARP_COMPILATION')

        # It's a good idea to always clear the kernel cache when developing new native or codegen features
        wp.build.clear_kernel_cache()

        # !!! DO THIS BEFORE DEFINING/USING ANY KERNELS WITH CUSTOM TYPES
        wpne.init()
    else:
        raise ValueError(f"Unsupported compute backend: {default_backend}")


def default_backend() -> ComputeBackend:
    return DefaultConfig.default_backend


def check_multi_gpu_support():
    import jax

    gpus = jax.devices("gpu")
    if len(gpus) > 1:
        print("Multi-GPU support is available: {} GPUs detected.".format(len(gpus)))
    elif len(gpus) == 1:
        print("Single-GPU support is available: 1 GPU detected.")
    else:
        print("No GPU support is available; CPU fallback will be used.")
