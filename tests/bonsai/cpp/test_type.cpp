#include "IR/Printer.h"
#include "IR/Type.h"

using namespace bonsai::ir;
using namespace bonsai;

void test_example() {
    Type f32 = Float_t::make(32);
    Type f32x2 = Vector_t::make(f32, 2);
    Type f32x2x2 = Vector_t::make(f32x2, 2);
    std::cout << f32x2x2 << std::endl;
    Type p_f32x2x2 = Ptr_t::make(f32x2x2);
    std::cout << p_f32x2x2 << std::endl;
}

int main(void) { test_example(); }
