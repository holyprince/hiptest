#include <iostream>
#include <iomanip>
#if __CUDACC__
#include <sys/time.h>
#else
#include <chrono>
#endif
#include <stddef.h>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define HC __attribute__((hc))


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"


#ifdef __HIP_PLATFORM_HCC
#define TYPENAME(T) typeid(T).name()
#else
#define TYPENAME(T) "?"
#endif


#define passed()                                                                                   \
    printf("%sPASSED!%s\n", KGRN, KNRM);                                                           \
    exit(0);

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define warn(...)                                                                                  \
    printf("%swarn: ", KYEL);                                                                      \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("warn: TEST WARNING\n%s", KNRM);

#define HIP_PRINT_STATUS(status)                                                                   \
    std::cout << hipGetErrorName(status) << " at line: " << __LINE__ << std::endl;

#define HIPCHECK(error)                                                                            \
    {                                                                                              \
        hipError_t localError = error;                                                             \
        if ((localError != hipSuccess) && (localError != hipErrorPeerAccessAlreadyEnabled)) {      \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, hipGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }

#define HIPASSERT(condition)                                                                       \
    if (!(condition)) {                                                                            \
        failed("%sassertion %s at %s:%d%s \n", KRED, #condition, __FILE__, __LINE__, KNRM);        \
    }


#define HIPCHECK_API(API_CALL, EXPECTED_ERROR)                                                     \
    {                                                                                              \
        hipError_t _e = (API_CALL);                                                                \
        if (_e != (EXPECTED_ERROR)) {                                                              \
            failed("%sAPI '%s' returned %d(%s) but test expected %d(%s) at %s:%d%s \n", KRED,      \
                   #API_CALL, _e, hipGetErrorName(_e), EXPECTED_ERROR,                             \
                   hipGetErrorName(EXPECTED_ERROR), __FILE__, __LINE__, KNRM);                     \
        }                                                                                          \
    }

 //standard command-line variables:
 extern size_t N;
 extern char memsetval;
 extern int iterations;
 extern unsigned blocksPerCU;
extern unsigned threadsPerBlock;
 extern int p_gpuDevice;
extern unsigned p_verbose;
extern int p_tests;


