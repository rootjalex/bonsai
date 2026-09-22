// The CUDA and buffer runtimes are header-only (see bonsai_cuda.h and
// bonsai_buffer.h), so that a driver has them by including the generated
// header. This translation unit is the compiler's copy: the JIT runs
// programs whose bound loops launch through it, and the PTX backend asks it
// which GPU the machine has.
#include "bonsai_buffer.h"
#include "bonsai_cuda.h"
