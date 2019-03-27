#ifndef ACC_SETTINGS_H_
#define ACC_SETTINGS_H_

#include "../macros.h"

#ifdef ACC_DOUBLE_PRECISION
	#define XFLOAT double
	#ifndef HIP
		typedef struct{ XFLOAT x; XFLOAT y;} double2;
	#endif
	#define ACCCOMPLEX double2
#else
	#define XFLOAT float
	#ifndef HIP
		typedef struct{ XFLOAT x; XFLOAT y;} float2;
	#endif
	#define ACCCOMPLEX float2
#endif
#ifdef ALTCPU
	#ifndef HIP
		typedef float hipStream_t;
		typedef double HipCustomAllocator;
		#define hipStreamPerThread 0
	#endif
#endif

#endif /* ACC_SETTINGS_H_ */
