// The OptiX runtime is header-only (see bonsai_optix.h), so that a driver has
// it by including the generated header. This translation unit is the
// compiler's copy: the JIT runs programs whose ray queries launch through it.
#include "bonsai_buffer.h"
#include "bonsai_optix.h"
