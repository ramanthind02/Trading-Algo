"""
Setup script to compile Cython optimized modules.

Usage:
    python utils/compute/cython/setup_cython.py build_ext --inplace

This will compile the cython_optimized.pyx file into a shared library
that can be imported like a regular Python module.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extensions
extensions = [
    Extension(
        name="utils.compute.cython.cython_optimized",
        sources=["utils/compute/cython/cython_optimized.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",           # Maximum optimization
            "-march=native", # Optimize for your CPU
            "-ffast-math",   # Fast math operations
        ],
        extra_link_args=["-O3"],
    ),
    Extension(
        name="utils.compute.cython.cython_nodes",
        sources=["utils/compute/cython/cython_nodes.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
        ],
        extra_link_args=["-O3"],
    )
]

setup(
    name="cython_optimized",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,      # Disable bounds checking for speed
            'wraparound': False,       # Disable negative indexing
            'cdivision': True,         # Use C division (faster)
            'initializedcheck': False, # Disable initialization checks
            'nonecheck': False,        # Disable None checks
        },
        annotate=True  # Generate HTML file showing Python/C interaction
    ),
    zip_safe=False,
)
