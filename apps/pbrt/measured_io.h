#ifndef BONSAI_PBRT_MEASURED_IO_H
#define BONSAI_PBRT_MEASURED_IO_H

// Reading PBRT's `.bsdf` files, and the precomputation its PiecewiseLinear2D
// does in its constructor.
//
// Both are here rather than in the renderer for the reason everything else of
// this shape is: they run once per material and involve a file, and what the
// renderer should be left with is the part that runs per ray. It is also the
// only way round -- PBRT's `Tensor` and `MeasuredBxDFData` are both file-local
// to bxdfs.cpp, so linking PBRT does not give access to either, and the format
// and the CDF construction are transcribed from there rather than called.
//
// The format is Dupuy and Jakob's tensor container: a twelve-byte magic, a
// version, a field count, then a table of (name, ndim, dtype, offset, shape)
// with the bulk data at the offsets.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace measured_io {

struct Tensor {
    enum Type : uint8_t {
        Invalid = 0,
        UInt8,
        Int8,
        UInt16,
        Int16,
        UInt32,
        Int32,
        UInt64,
        Int64,
        Float16,
        Float32,
        Float64,
    };

    struct Field {
        Type dtype = Invalid;
        std::vector<size_t> shape;
        std::vector<uint8_t> data;

        const float *as_float() const {
            return reinterpret_cast<const float *>(data.data());
        }
    };

    std::map<std::string, Field> fields;
    std::string error;

    bool ok() const { return error.empty(); }

    const Field *find(const std::string &name) const {
        const auto it = fields.find(name);
        return it == fields.end() ? nullptr : &it->second;
    }
};

inline size_t tensor_type_size(Tensor::Type t) {
    switch (t) {
    case Tensor::UInt8:
    case Tensor::Int8:
        return 1;
    case Tensor::UInt16:
    case Tensor::Int16:
    case Tensor::Float16:
        return 2;
    case Tensor::UInt32:
    case Tensor::Int32:
    case Tensor::Float32:
        return 4;
    case Tensor::UInt64:
    case Tensor::Int64:
    case Tensor::Float64:
        return 8;
    default:
        return 0;
    }
}

inline Tensor read_tensor(const std::string &filename) {
    Tensor out;
    const auto fail = [&](const char *why) {
        out.error = filename + ": " + why;
        return out;
    };

    FILE *file = std::fopen(filename.c_str(), "rb");
    if (file == nullptr) {
        return fail("cannot open");
    }
    struct Closer {
        FILE *f;
        ~Closer() { std::fclose(f); }
    } closer{file};

    uint8_t header[12];
    uint8_t version[2];
    uint32_t n_fields = 0;
    if (std::fread(header, 1, 12, file) != 12 ||
        std::fread(version, 1, 2, file) != 2 ||
        std::fread(&n_fields, sizeof(n_fields), 1, file) != 1) {
        return fail("truncated header");
    }
    if (std::memcmp(header, "tensor_file", 12) != 0) {
        return fail("not a tensor file");
    }
    if (version[0] != 1 || version[1] != 0) {
        return fail("unknown tensor file version");
    }

    for (uint32_t i = 0; i < n_fields; i++) {
        uint16_t name_length = 0;
        uint16_t ndim = 0;
        uint8_t dtype = 0;
        uint64_t offset = 0;
        if (std::fread(&name_length, sizeof(name_length), 1, file) != 1) {
            return fail("truncated field table");
        }
        std::string name(name_length, '\0');
        if (std::fread(name.data(), 1, name_length, file) != name_length ||
            std::fread(&ndim, sizeof(ndim), 1, file) != 1 ||
            std::fread(&dtype, sizeof(dtype), 1, file) != 1 ||
            std::fread(&offset, sizeof(offset), 1, file) != 1) {
            return fail("truncated field table");
        }
        if (dtype == Tensor::Invalid || dtype > Tensor::Float64) {
            return fail("unknown field type");
        }

        Tensor::Field field;
        field.dtype = Tensor::Type(dtype);
        field.shape.resize(ndim);
        size_t total = tensor_type_size(field.dtype);
        for (uint16_t j = 0; j < ndim; j++) {
            uint64_t extent = 0;
            if (std::fread(&extent, sizeof(extent), 1, file) != 1) {
                return fail("truncated shape");
            }
            field.shape[j] = size_t(extent);
            total *= field.shape[j];
        }

        const long here = std::ftell(file);
        if (here < 0 || std::fseek(file, long(offset), SEEK_SET) != 0) {
            return fail("cannot seek to a field");
        }
        field.data.resize(total);
        if (total > 0 && std::fread(field.data.data(), 1, total, file) != total) {
            return fail("truncated field data");
        }
        if (std::fseek(file, here, SEEK_SET) != 0) {
            return fail("cannot seek back");
        }
        out.fields.emplace(std::move(name), std::move(field));
    }
    return out;
}

// PBRT's PiecewiseLinear2D, as the renderer needs it: the normalized data and,
// where it was asked for, the two CDFs.
//
// The construction is PBRT's constructor transcribed. What it does is turn a
// grid of densities into something the inversion method can sample: a marginal
// CDF over rows and a conditional CDF along each row, both trapezoidal because
// the function between samples is linear rather than constant.
struct PL2D {
    int size_x = 0;
    int size_y = 0;
    // One entry per parameter dimension, outermost first, as PBRT stores them.
    std::vector<int> param_size;
    std::vector<uint32_t> param_stride;
    std::vector<std::vector<float>> param_values;

    std::vector<float> data;
    std::vector<float> marginal_cdf;
    std::vector<float> conditional_cdf;
};

// `param_res` and `param_values` are outermost-first, matching the order PBRT
// passes them; the strides are computed innermost-first, as PBRT's loop does.
inline PL2D build_pl2d(const float *data, int x_size, int y_size,
                       const std::vector<int> &param_res,
                       const std::vector<const float *> &param_values,
                       bool normalize, bool build_cdf) {
    PL2D out;
    out.size_x = x_size;
    out.size_y = y_size;

    const size_t dim = param_res.size();
    out.param_size.assign(param_res.begin(), param_res.end());
    out.param_stride.assign(dim, 0);
    out.param_values.resize(dim);

    uint32_t slices = 1;
    for (int i = int(dim) - 1; i >= 0; i--) {
        out.param_values[i].assign(param_values[i],
                                   param_values[i] + param_res[i]);
        out.param_stride[i] = param_res[i] > 1 ? slices : 0;
        slices *= uint32_t(param_res[i]);
    }

    const uint32_t n_values = uint32_t(x_size) * uint32_t(y_size);
    out.data.assign(size_t(slices) * n_values, 0.f);

    const float inv_patch_x = float(x_size - 1);
    const float inv_patch_y = float(y_size - 1);

    if (build_cdf) {
        out.marginal_cdf.assign(size_t(slices) * y_size, 0.f);
        out.conditional_cdf.assign(size_t(slices) * n_values, 0.f);

        for (uint32_t slice = 0; slice < slices; slice++) {
            const float *in = data + size_t(slice) * n_values;
            float *cond = out.conditional_cdf.data() + size_t(slice) * n_values;
            float *marg = out.marginal_cdf.data() + size_t(slice) * y_size;
            float *out_data = out.data.data() + size_t(slice) * n_values;

            for (int y = 0; y < y_size; y++) {
                double sum = 0.0;
                size_t i = size_t(y) * x_size;
                cond[i] = 0.f;
                for (int x = 0; x < x_size - 1; x++, i++) {
                    sum += .5 * (double(in[i]) + double(in[i + 1]));
                    cond[i + 1] = float(sum);
                }
            }

            marg[0] = 0.f;
            double sum = 0.0;
            for (int y = 0; y < y_size - 1; y++) {
                sum += .5 * (double(cond[size_t(y + 1) * x_size - 1]) +
                             double(cond[size_t(y + 2) * x_size - 1]));
                marg[y + 1] = float(sum);
            }

            const float normalization = 1.f / marg[y_size - 1];
            for (size_t i = 0; i < n_values; i++) {
                cond[i] *= normalization;
            }
            for (int i = 0; i < y_size; i++) {
                marg[i] *= normalization;
            }
            for (size_t i = 0; i < n_values; i++) {
                out_data[i] = in[i] * normalization;
            }
        }
    } else {
        for (uint32_t slice = 0; slice < slices; slice++) {
            const float *in = data + size_t(slice) * n_values;
            float *out_data = out.data.data() + size_t(slice) * n_values;

            float normalization = 1.f / (inv_patch_x * inv_patch_y);
            if (normalize) {
                double sum = 0.0;
                for (int y = 0; y < y_size - 1; y++) {
                    size_t i = size_t(y) * x_size;
                    for (int x = 0; x < x_size - 1; x++, i++) {
                        const float avg =
                            .25f * (in[i] + in[i + 1] + in[i + x_size] +
                                    in[i + 1 + x_size]);
                        sum += double(avg);
                    }
                }
                normalization = float(1.0 / sum);
            }
            for (uint32_t k = 0; k < n_values; k++) {
                out_data[k] = in[k] * normalization;
            }
        }
    }
    return out;
}

} // namespace measured_io

#endif // BONSAI_PBRT_MEASURED_IO_H
