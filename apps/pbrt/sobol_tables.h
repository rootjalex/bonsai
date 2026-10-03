// pbrt's Sobol' generator matrices for dimensions zero and one, transcribed
// from pbrt-v4's src/pbrt/util/sobolmatrices.cpp, and the twenty-four
// permutations of four digits that ZSobolSampler::GetSampleIndex permutes a
// Morton index's base-4 digits with, from src/pbrt/samplers.h. The driver
// hands both to bonsai as extern arrays, the way the CIE curves arrive
// (cie_tables.h): bonsai has no file I/O, and tabulated constants reach it the
// same way scene data does. `scene_dump --check-tables` checks the matrices
// against pbrt's own table, word for word.
//
// Two dimensions rather than pbrt's thousand and twenty-four, because the
// ZSobol and PaddedSobol samplers -- pbrt's default among them -- draw every
// dimension from these two under a different scramble. The plain `sobol`
// sampler walks the dimensions and needs the whole table, and for it
// scene_dump, which links pbrt, copies pbrt's own array into the `.smp`
// sidecar beside the scene (scene_io.h) -- a header of 53,248 words would be
// a transcription nobody could check by reading.
//
// Dimension zero is the van der Corput sequence, each column one bit; the
// columns past the thirty-second are zero, since a sample index's bits above
// the word reverse to nothing. Dimension one's columns go on repeating with
// period thirty-two.
#pragma once

#include <cstdint>

inline constexpr int SOBOL_MATRIX_SIZE = 52;
inline constexpr int SOBOL_DIMENSIONS_SHIPPED = 2;

inline constexpr uint32_t SOBOL_MATRICES[SOBOL_DIMENSIONS_SHIPPED *
                                         SOBOL_MATRIX_SIZE] = {
    // Dimension zero.
    0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x08000000, 0x04000000,
    0x02000000, 0x01000000, 0x00800000, 0x00400000, 0x00200000, 0x00100000,
    0x00080000, 0x00040000, 0x00020000, 0x00010000, 0x00008000, 0x00004000,
    0x00002000, 0x00001000, 0x00000800, 0x00000400, 0x00000200, 0x00000100,
    0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004,
    0x00000002, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    // Dimension one.
    0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000,
    0xaa000000, 0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
    0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000,
    0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
    0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc,
    0xaaaaaaaa, 0xffffffff, 0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
    0x88000000, 0xcc000000, 0xaa000000, 0xff000000, 0x80800000, 0xc0c00000,
    0xa0a00000, 0xf0f00000, 0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
    0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
};

// pbrt: ZSobolSampler::GetSampleIndex's `permutations`, in pbrt's order. The
// order is the table's meaning: which row a hash picks is `hash % 24`, so a
// row moved is a different permutation for every digit that hashed to it.
inline constexpr uint8_t ZSOBOL_PERMUTATIONS[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 2, 1},
    {0, 3, 1, 2}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0},
    {1, 3, 2, 0}, {1, 3, 0, 2}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 0, 1, 3},
    {2, 0, 3, 1}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 1, 2, 0}, {3, 1, 0, 2},
    {3, 2, 1, 0}, {3, 2, 0, 1}, {3, 0, 2, 1}, {3, 0, 1, 2},
};
