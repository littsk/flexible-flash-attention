/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

// Runtime check macro nFunc.
//
// DEVIATION FROM UPSTREAM (Tri Dao's FlashAttention): the original macro
// called ``exit(1)`` on failure. That is fatal for multi-rank PyTorch tests:
// when this fires inside a single rank's CUDA dispatch, that rank dies
// without ever returning to Python, so ``destroy_pg()`` (and its internal
// ``dist.barrier()``) is never reached on that rank -- the remaining ranks
// then block forever in the barrier and the whole test job hangs instead of
// reporting the error.
//
// Reason: a ``NFUNC_SWITCH`` runtime/compile-time mismatch (e.g. the
// FA4-arbitrary-mask kernel was built with ``FLASH_ATTENTION_NUM_FUNC=...,31``
// but a workload generates ``arbitrary_func_num=33``) is a build-config
// limitation, not a numerical bug; it should be recoverable so Python can
// catch it, surface a clear error, and let the surrounding ``finally``
// blocks (``destroy_pg`` etc.) run.
//
// Recovery: throwing ``std::runtime_error`` from a kernel-launch entrypoint
// propagates through pybind11 as a Python ``RuntimeError`` with the same
// message, which test code can either re-raise (fail loudly on a single-rank
// run) or pattern-match and SKIP (matching the existing
// ``Fa4MaxKvIntervalsExceededError`` skip pattern in
// ``tests/test_attn/test_flex_flash_attn.py``).
#define FLASH_CHECK(cond, ...)                                                                       \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            char _flash_check_msg[1024];                                                             \
            int _flash_check_prefix = snprintf(                                                      \
                _flash_check_msg, sizeof(_flash_check_msg),                                          \
                "FlashAttention Arbitrary Mask Function error (%s:%d): ",                            \
                __FILE__, __LINE__);                                                                 \
            if (_flash_check_prefix < 0) { _flash_check_prefix = 0; }                                \
            if (_flash_check_prefix > (int)sizeof(_flash_check_msg)) {                               \
                _flash_check_prefix = sizeof(_flash_check_msg);                                      \
            }                                                                                        \
            snprintf(                                                                                \
                _flash_check_msg + _flash_check_prefix,                                              \
                sizeof(_flash_check_msg) - _flash_check_prefix,                                      \
                __VA_ARGS__);                                                                        \
            throw std::runtime_error(std::string(_flash_check_msg));                                 \
        }                                                                                            \
    } while(0)

// Same recoverable-error treatment as ``FLASH_CHECK`` above: a CUDA runtime
// failure inside a kernel launch should surface as a Python exception, not
// kill the process and deadlock the other ranks in their next collective.
#define CHECK_CUDA(call)                                                                             \
    do {                                                                                             \
        cudaError_t status_ = (call);                                                                \
        if (status_ != cudaSuccess) {                                                                \
            char _check_cuda_msg[512];                                                               \
            snprintf(_check_cuda_msg, sizeof(_check_cuda_msg),                                       \
                "CUDA error (%s:%d): %s",                                                            \
                __FILE__, __LINE__, cudaGetErrorString(status_));                                    \
            throw std::runtime_error(std::string(_check_cuda_msg));                                  \
        }                                                                                            \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())
