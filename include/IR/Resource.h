#pragma once

namespace bonsai {
namespace ir {

// A piece of hardware a cursor can be bound to.
//
// Its own header because both ends of a bind() need it: the schedule, which
// says what to bind (IR/Schedule.h), and the loop that carries the answer
// until code is generated (ParFor in IR/Stmt.h).
//
// The nesting these require of each other, checked where a bind is applied: a
// GPUThread runs inside a GPUBlock, an RTCore inside an OptixThread, and a
// vectorize() inside whichever thread owns it -- a gang of lanes is the
// innermost level there is.
enum class Resource {
    CPUThread,
    GPUThread,
    GPUBlock,
    RTCore,
    OptixThread,
    // The GPU's texture units: what a function bound to it runs on, as the
    // lambda the bind gives says (see Bind::lambda in IR/Schedule.h) --
    // `texture_filter.bind(TextureUnit, |t, st, dstdx, dstdy| ...)`.
    TextureUnit,
};

inline const char *to_string(Resource resource) {
    switch (resource) {
    case Resource::CPUThread:
        return "CPUThread";
    case Resource::GPUThread:
        return "GPUThread";
    case Resource::GPUBlock:
        return "GPUBlock";
    case Resource::RTCore:
        return "RTCore";
    case Resource::OptixThread:
        return "OptixThread";
    case Resource::TextureUnit:
        return "TextureUnit";
    }
    return "<unknown resource>";
}

} // namespace ir
} // namespace bonsai
