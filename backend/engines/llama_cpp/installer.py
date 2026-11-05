"""Compatibility import for the shared CMake source host.

New code should import :class:`backend.engines.python.cmake_source.CmakeSourceInstaller`.
The historical name stays importable for routes/managers and tests.
"""

from backend.engines.python.cmake_source import CmakeSourceInstaller

LlamaNativeInstaller = CmakeSourceInstaller

__all__ = ["CmakeSourceInstaller", "LlamaNativeInstaller"]
