// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
//

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                       \
  [&] {                                                                                          \
    if (COND) {                                                                                  \
      constexpr static bool CONST_NAME = true;                                                   \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static bool CONST_NAME = false;                                                  \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      constexpr static bool LOCAL_CONST_NAME = false;                                            \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#else
  #define CAUSAL_LOCAL_SWITCH(CAUSAL_COND, LOCAL_COND, CAUSAL_CONST_NAME, LOCAL_CONST_NAME, ...) \
    [&] {                                                                                        \
      if (CAUSAL_COND) {                                                                         \
        constexpr static bool CAUSAL_CONST_NAME = true;                                          \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      } else if (LOCAL_COND) {                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = true;                                           \
        return __VA_ARGS__();                                                                    \
      } else {                                                                                   \
        constexpr static bool CAUSAL_CONST_NAME = false;                                         \
        constexpr static bool LOCAL_CONST_NAME = false;                                          \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
    }()
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  #define PAGEDKV_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PAGEDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SPLIT
  #define SPLIT_SWITCH(COND, CONST_NAME, ...)                                                    \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define SPLIT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_APPENDKV
  #define APPENDKV_SWITCH(COND, CONST_NAME, ...)                                                 \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define APPENDKV_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_PACKGQA
  #define PACKGQA_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define PACKGQA_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_VARLEN
  #define VARLEN_SWITCH(COND, CONST_NAME, ...)                                                   \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VARLEN_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_CLUSTER
  #define CLUSTER_SWITCH(COND, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define CLUSTER_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SM8x
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    constexpr static int ARCH_NAME = 90;                                                         \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define ARCH_SWITCH(ARCH, ARCH_NAME, ...)                                                      \
  [&] {                                                                                          \
    if (ARCH == 86 || ARCH == 89) {                                                              \
      constexpr static int ARCH_NAME = 86;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (ARCH < 90) {                                                                      \
      constexpr static int ARCH_NAME = 80;                                                       \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int ARCH_NAME = 90;                                                       \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()
#endif

#ifndef FLASHATTENTION_ENABLE_VCOLMAJOR
  #define VCOLMAJOR_SWITCH(COND, CONST_NAME, ...)                                                \
  [&] {                                                                                          \
    constexpr static bool CONST_NAME = false;                                                    \
    return __VA_ARGS__();                                                                        \
  }()
#else
  #define VCOLMAJOR_SWITCH BOOL_SWITCH
#endif

#define HEADDIM_SWITCH(HEADDIM, ...)                                                             \
  [&] {                                                                                          \
    if (HEADDIM == 64) {                                                                         \
      constexpr static int kHeadSize = 64;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 128) {                                                                 \
      constexpr static int kHeadSize = 128;                                                      \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 96) {                                                                  \
      constexpr static int kHeadSize = 96;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (HEADDIM == 256) {                                                                 \
      constexpr static int kHeadSize = 256;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

#define NUM_WARP_SWITCH(VALUE, CONST_NAME, ...)                                                  \
  [&] {                                                                                          \
    if (VALUE <= 1) {                                                                            \
      constexpr static int CONST_NAME = 1;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 2) {                                                                     \
      constexpr static int CONST_NAME = 2;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 4) {                                                                     \
      constexpr static int CONST_NAME = 4;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 8) {                                                                     \
      constexpr static int CONST_NAME = 8;                                                       \
      return __VA_ARGS__();                                                                      \
    } else if (VALUE <= 16) {                                                                    \
      constexpr static int CONST_NAME = 16;                                                      \
      return __VA_ARGS__();                                                                      \
    } else {                                                                                     \
      constexpr static int CONST_NAME = 32;                                                      \
      return __VA_ARGS__();                                                                      \
    }                                                                                            \
  }()

// NFUNC_SWITCH - switch based on arbitrary_func_num value
// Controlled by FLASH_ATTENTION_NUM_FUNC environment variable
// Valid values are odd numbers from 1 to 33 (e.g., "1,3,5")
// When Is_arbitrary is false, kNFunc is 0
// When Is_arbitrary is true, kNFunc is determined by params.arbitrary_func_num
// Each supported value generates -DFLASHATTENTION_NFUNC_X compile flag

// Define NFUNC_SWITCH_CASE macros - only compile if corresponding macro is defined
#ifdef FLASHATTENTION_NFUNC_1
  #define NFUNC_SWITCH_CASE_1(NFUNC_VALUE, CONST_NAME, ...)                                      \
    if ((NFUNC_VALUE) == 1) { constexpr static int CONST_NAME = 1; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_1(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_3
  #define NFUNC_SWITCH_CASE_3(NFUNC_VALUE, CONST_NAME, ...)                                      \
    if ((NFUNC_VALUE) == 3) { constexpr static int CONST_NAME = 3; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_3(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_5
  #define NFUNC_SWITCH_CASE_5(NFUNC_VALUE, CONST_NAME, ...)                                      \
    if ((NFUNC_VALUE) == 5) { constexpr static int CONST_NAME = 5; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_5(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_7
  #define NFUNC_SWITCH_CASE_7(NFUNC_VALUE, CONST_NAME, ...)                                      \
    if ((NFUNC_VALUE) == 7) { constexpr static int CONST_NAME = 7; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_7(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_9
  #define NFUNC_SWITCH_CASE_9(NFUNC_VALUE, CONST_NAME, ...)                                      \
    if ((NFUNC_VALUE) == 9) { constexpr static int CONST_NAME = 9; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_9(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_11
  #define NFUNC_SWITCH_CASE_11(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 11) { constexpr static int CONST_NAME = 11; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_11(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_13
  #define NFUNC_SWITCH_CASE_13(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 13) { constexpr static int CONST_NAME = 13; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_13(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_15
  #define NFUNC_SWITCH_CASE_15(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 15) { constexpr static int CONST_NAME = 15; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_15(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_17
  #define NFUNC_SWITCH_CASE_17(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 17) { constexpr static int CONST_NAME = 17; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_17(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_19
  #define NFUNC_SWITCH_CASE_19(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 19) { constexpr static int CONST_NAME = 19; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_19(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_21
  #define NFUNC_SWITCH_CASE_21(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 21) { constexpr static int CONST_NAME = 21; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_21(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_23
  #define NFUNC_SWITCH_CASE_23(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 23) { constexpr static int CONST_NAME = 23; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_23(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_25
  #define NFUNC_SWITCH_CASE_25(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 25) { constexpr static int CONST_NAME = 25; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_25(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_27
  #define NFUNC_SWITCH_CASE_27(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 27) { constexpr static int CONST_NAME = 27; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_27(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_29
  #define NFUNC_SWITCH_CASE_29(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 29) { constexpr static int CONST_NAME = 29; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_29(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_31
  #define NFUNC_SWITCH_CASE_31(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 31) { constexpr static int CONST_NAME = 31; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_31(NFUNC_VALUE, CONST_NAME, ...)
#endif

#ifdef FLASHATTENTION_NFUNC_33
  #define NFUNC_SWITCH_CASE_33(NFUNC_VALUE, CONST_NAME, ...)                                     \
    if ((NFUNC_VALUE) == 33) { constexpr static int CONST_NAME = 33; return __VA_ARGS__(); }
#else
  #define NFUNC_SWITCH_CASE_33(NFUNC_VALUE, CONST_NAME, ...)
#endif

// Default value for fallback - use the first enabled NFUNC value
#if defined(FLASHATTENTION_NFUNC_1)
  #define NFUNC_SWITCH_DEFAULT_VALUE 1
#elif defined(FLASHATTENTION_NFUNC_3)
  #define NFUNC_SWITCH_DEFAULT_VALUE 3
#elif defined(FLASHATTENTION_NFUNC_5)
  #define NFUNC_SWITCH_DEFAULT_VALUE 5
#elif defined(FLASHATTENTION_NFUNC_7)
  #define NFUNC_SWITCH_DEFAULT_VALUE 7
#elif defined(FLASHATTENTION_NFUNC_9)
  #define NFUNC_SWITCH_DEFAULT_VALUE 9
#elif defined(FLASHATTENTION_NFUNC_11)
  #define NFUNC_SWITCH_DEFAULT_VALUE 11
#elif defined(FLASHATTENTION_NFUNC_13)
  #define NFUNC_SWITCH_DEFAULT_VALUE 13
#elif defined(FLASHATTENTION_NFUNC_15)
  #define NFUNC_SWITCH_DEFAULT_VALUE 15
#elif defined(FLASHATTENTION_NFUNC_17)
  #define NFUNC_SWITCH_DEFAULT_VALUE 17
#elif defined(FLASHATTENTION_NFUNC_19)
  #define NFUNC_SWITCH_DEFAULT_VALUE 19
#elif defined(FLASHATTENTION_NFUNC_21)
  #define NFUNC_SWITCH_DEFAULT_VALUE 21
#elif defined(FLASHATTENTION_NFUNC_23)
  #define NFUNC_SWITCH_DEFAULT_VALUE 23
#elif defined(FLASHATTENTION_NFUNC_25)
  #define NFUNC_SWITCH_DEFAULT_VALUE 25
#elif defined(FLASHATTENTION_NFUNC_27)
  #define NFUNC_SWITCH_DEFAULT_VALUE 27
#elif defined(FLASHATTENTION_NFUNC_29)
  #define NFUNC_SWITCH_DEFAULT_VALUE 29
#elif defined(FLASHATTENTION_NFUNC_31)
  #define NFUNC_SWITCH_DEFAULT_VALUE 31
#elif defined(FLASHATTENTION_NFUNC_33)
  #define NFUNC_SWITCH_DEFAULT_VALUE 33
#else
  #define NFUNC_SWITCH_DEFAULT_VALUE 1
#endif

// When ARBITRARY is disabled, only generate the Is_arbitrary=false path
#ifdef FLASHATTENTION_DISABLE_ARBITRARY
  #define NFUNC_SWITCH(IS_ARBITRARY, NFUNC_VALUE, CONST_NAME, ...)                               \
    [&] {                                                                                        \
      constexpr static int CONST_NAME = 0;                                                       \
      return __VA_ARGS__();                                                                      \
    }()
#else
  #define NFUNC_SWITCH(IS_ARBITRARY, NFUNC_VALUE, CONST_NAME, ...)                               \
    [&] {                                                                                        \
      if (!(IS_ARBITRARY)) {                                                                     \
        constexpr static int CONST_NAME = 0;                                                     \
        return __VA_ARGS__();                                                                    \
      }                                                                                          \
      /* Check each possible value, only compile if corresponding macro is defined */            \
      NFUNC_SWITCH_CASE_1(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                  \
      NFUNC_SWITCH_CASE_3(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                  \
      NFUNC_SWITCH_CASE_5(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                  \
      NFUNC_SWITCH_CASE_7(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                  \
      NFUNC_SWITCH_CASE_9(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                  \
      NFUNC_SWITCH_CASE_11(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_13(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_15(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_17(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_19(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_21(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_23(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_25(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_27(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_29(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_31(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      NFUNC_SWITCH_CASE_33(NFUNC_VALUE, CONST_NAME, __VA_ARGS__)                                 \
      /* Fallback: use default value to satisfy compiler */                                      \
      constexpr static int CONST_NAME = NFUNC_SWITCH_DEFAULT_VALUE;                              \
      return __VA_ARGS__();                                                                      \
    }()
#endif
