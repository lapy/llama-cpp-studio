/**
 * glibc >= 2.41 C23 math functions vs CUDA <= 12.9 host pass.
 *
 * glibc 2.41 (Ubuntu 25.04+ and 26.04) declares rsqrt/rsqrtf and
 * sinpi/cospi/sinpif/cospif through __THROW, which expands to
 * noexcept(true) in C++. CUDA's crt/math_functions.h declares the same
 * symbols without an exception specification, and noexcept is part of the
 * function type in C++17. nvcc's host pass therefore reports:
 *
 *   error: exception specification is incompatible with that of previous
 *   function "cospi" (declared at .../crt/math_functions.h)
 *
 * This header is force-included into the host pass only:
 *   -Xcompiler=-include,<this file>
 * It strips glibc's __THROW so its declarations carry no exception
 * specification and match CUDA's. The device pass never sees it because
 * -Xcompiler flags do not reach nvcc's device frontend.
 *
 * No-op on glibc before 2.41 and outside Linux.
 */
#ifndef STUDIO_GLIBC_C23_MATH_COMPAT_H
#define STUDIO_GLIBC_C23_MATH_COMPAT_H

#if defined(__cplusplus) && defined(__linux__)
#include <features.h>
#if defined(__GLIBC__) && defined(__THROW)
#undef __THROW
#define __THROW
#endif
#endif

#endif
