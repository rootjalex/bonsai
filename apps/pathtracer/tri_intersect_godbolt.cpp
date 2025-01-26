// flags: -std=c++17 --target=aarch64-linux-gnu -O3
// compare to: -mtriple=aarch64-linux-gnu -O3

#include <array>
#include <cmath>
#include <optional>

typedef float Float;
#define PBRT_CPU_GPU
#define DCHECK(...)

struct TriangleIntersection {
    Float b0, b1, b2;
    Float t;
    // std::string ToString() const;
};

template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Point3;
template <typename T>
class Point2;
template <typename T>
class Normal3;
using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;

template <typename T>
inline PBRT_CPU_GPU typename std::enable_if_t<std::is_floating_point_v<T>, bool>
IsNaN(T v) {
#ifdef PBRT_IS_GPU_CODE
    return isnan(v);
#else
    return std::isnan(v);
#endif
}

template <template <typename> class Child, typename T>
class Tuple3 {
  public:
    // Tuple3 Public Methods
    Tuple3() = default;
    PBRT_CPU_GPU
    Tuple3(T x, T y, T z) : x(x), y(y), z(z) { DCHECK(!HasNaN()); }

    PBRT_CPU_GPU
    bool HasNaN() const { return IsNaN(x) || IsNaN(y) || IsNaN(z); }

    PBRT_CPU_GPU
    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    PBRT_CPU_GPU
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0)
            return x;
        if (i == 1)
            return y;
        return z;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator+(Child<U> c) const
        -> Child<decltype(T{} + U{})> {
        DCHECK(!c.HasNaN());
        return {x + c.x, y + c.y, z + c.z};
    }

    static const int nDimensions = 3;

#ifdef PBRT_DEBUG_BUILD
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    PBRT_CPU_GPU
    Tuple3(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
    }

    PBRT_CPU_GPU
    Child<T> &operator=(Child<T> c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
        return static_cast<Child<T> &>(*this);
    }
#endif

    template <typename U>
    PBRT_CPU_GPU Child<T> &operator+=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x += c.x;
        y += c.y;
        z += c.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Child<U> c) const
        -> Child<decltype(T{} - U{})> {
        DCHECK(!c.HasNaN());
        return {x - c.x, y - c.y, z - c.z};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator-=(Child<U> c) {
        DCHECK(!c.HasNaN());
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_CPU_GPU
    bool operator==(Child<T> c) const {
        return x == c.x && y == c.y && z == c.z;
    }
    PBRT_CPU_GPU
    bool operator!=(Child<T> c) const {
        return x != c.x || y != c.y || z != c.z;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator*(U s) const -> Child<decltype(T{} * U{})> {
        return {s * x, s * y, s * z};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator*=(U s) {
        DCHECK(!IsNaN(s));
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_CPU_GPU auto operator/(U d) const -> Child<decltype(T{} / U{})> {
        DCHECK_NE(d, 0);
        return {x / d, y / d, z / d};
    }
    template <typename U>
    PBRT_CPU_GPU Child<T> &operator/=(U d) {
        DCHECK_NE(d, 0);
        x /= d;
        y /= d;
        z /= d;
        return static_cast<Child<T> &>(*this);
    }
    PBRT_CPU_GPU
    Child<T> operator-() const { return {-x, -y, -z}; }

    // std::string ToString() const { return internal::ToString3(x, y, z); }

    // Tuple3 Public Members
    T x{}, y{}, z{};
};

template <typename T>
class Point3 : public Tuple3<Point3, T> {
  public:
    // Point3 Public Methods
    using Tuple3<Point3, T>::x;
    using Tuple3<Point3, T>::y;
    using Tuple3<Point3, T>::z;
    using Tuple3<Point3, T>::HasNaN;
    using Tuple3<Point3, T>::operator+;
    using Tuple3<Point3, T>::operator+=;
    using Tuple3<Point3, T>::operator*;
    using Tuple3<Point3, T>::operator*=;

    Point3() = default;
    PBRT_CPU_GPU
    Point3(T x, T y, T z) : Tuple3<Point3, T>(x, y, z) {}

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    PBRT_CPU_GPU
    Point3<T> operator-() const { return {-x, -y, -z}; }

    template <typename U>
    PBRT_CPU_GPU explicit Point3(Point3<U> p)
        : Tuple3<Point3, T>(T(p.x), T(p.y), T(p.z)) {}
    template <typename U>
    PBRT_CPU_GPU explicit Point3(Vector3<U> v)
        : Tuple3<Point3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    PBRT_CPU_GPU auto operator+(Vector3<U> v) const
        -> Point3<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return {x + v.x, y + v.y, z + v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3<T> &operator+=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Vector3<U> v) const
        -> Point3<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return {x - v.x, y - v.y, z - v.z};
    }
    template <typename U>
    PBRT_CPU_GPU Point3<T> &operator-=(Vector3<U> v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    template <typename U>
    PBRT_CPU_GPU auto operator-(Point3<U> p) const
        -> Vector3<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return {x - p.x, y - p.y, z - p.z};
    }
};

template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
    // Vector3 Public Methods
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;

    Vector3() = default;
    PBRT_CPU_GPU
    Vector3(T x, T y, T z) : Tuple3<Vector3, T>(x, y, z) {}

    template <typename U>
    PBRT_CPU_GPU explicit Vector3(Vector3<U> v)
        : Tuple3<Vector3, T>(T(v.x), T(v.y), T(v.z)) {}

    template <typename U>
    PBRT_CPU_GPU explicit Vector3(Point3<U> p);
    // template <typename U>
    // PBRT_CPU_GPU explicit Vector3(Normal3<U> n);
};

// Vector3 Inline Functions
template <typename T>
template <typename U>
Vector3<T>::Vector3(Point3<U> p) : Tuple3<Vector3, T>(T(p.x), T(p.y), T(p.z)) {}

class Ray {
  public:
    // Ray Public Methods
    // PBRT_CPU_GPU
    bool HasNaN() const { return (o.HasNaN() || d.HasNaN()); }

    // std::string ToString() const;

    // PBRT_CPU_GPU
    Point3f operator()(Float t) const { return o + d * t; }

    Ray() = default;
    // PBRT_CPU_GPU
    Ray(Point3f o, Vector3f d, Float time = 0.f) : o(o), d(d), time(time) {}

    // Ray Public Members
    Point3f o;
    Vector3f d;
    Float time = 0;
    // Medium medium = nullptr;
};

PBRT_CPU_GPU inline float FMA(float a, float b, float c) {
    return std::fma(a, b, c);
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU inline auto DifferenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto differenceOfProducts = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return differenceOfProducts + error;
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> Cross(Vector3<T> v, Vector3<T> w) {
    DCHECK(!v.HasNaN() && !w.HasNaN());
    return {DifferenceOfProducts(v.y, w.z, v.z, w.y),
            DifferenceOfProducts(v.z, w.x, v.x, w.z),
            DifferenceOfProducts(v.x, w.y, v.y, w.x)};
}

template <typename T>
PBRT_CPU_GPU inline constexpr T Sqr(T v) {
    return v * v;
}

template <typename T>
PBRT_CPU_GPU inline T LengthSquared(Vector3<T> v) {
    return Sqr(v.x) + Sqr(v.y) + Sqr(v.z);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Abs(Tuple3<C, T> t) {
    using std::abs;
    return {abs(t.x), abs(t.y), abs(t.z)};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline int MaxComponentIndex(Tuple3<C, T> t) {
    return (t.x > t.y) ? ((t.x > t.z) ? 0 : 2) : ((t.y > t.z) ? 1 : 2);
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline C<T> Permute(Tuple3<C, T> t, std::array<int, 3> p) {
    return {t[p[0]], t[p[1]], t[p[2]]};
}

template <template <class> class C, typename T>
PBRT_CPU_GPU inline T MaxComponentValue(Tuple3<C, T> t) {
    using std::max;
    return max(t.x, max(t.y, t.z));
}

static constexpr Float MachineEpsilon =
    std::numeric_limits<Float>::epsilon() * 0.5;
inline constexpr Float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

std::optional<TriangleIntersection> IntersectTriangle(const Ray &ray,
                                                      Float tMax, Point3f p0,
                                                      Point3f p1, Point3f p2) {
    // Return no intersection if triangle is degenerate
    if (LengthSquared(Cross(p2 - p0, p1 - p0)) == 0)
        return {};

    // Transform triangle vertices to ray coordinate space
    // Translate vertices based on ray origin
    Point3f p0t = p0 - Vector3f(ray.o);
    Point3f p1t = p1 - Vector3f(ray.o);
    Point3f p2t = p2 - Vector3f(ray.o);

    // Permute components of triangle vertices and ray direction
    int kz = MaxComponentIndex(Abs(ray.d));
    int kx = kz + 1;
    if (kx == 3)
        kx = 0;
    int ky = kx + 1;
    if (ky == 3)
        ky = 0;
    Vector3f d = Permute(ray.d, {kx, ky, kz});
    p0t = Permute(p0t, {kx, ky, kz});
    p1t = Permute(p1t, {kx, ky, kz});
    p2t = Permute(p2t, {kx, ky, kz});

    // Apply shear transformation to translated vertex positions
    Float Sx = -d.x / d.z;
    Float Sy = -d.y / d.z;
    Float Sz = 1 / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    // Compute edge function coefficients _e0_, _e1_, and _e2_
    Float e0 = DifferenceOfProducts(p1t.x, p2t.y, p1t.y, p2t.x);
    Float e1 = DifferenceOfProducts(p2t.x, p0t.y, p2t.y, p0t.x);
    Float e2 = DifferenceOfProducts(p0t.x, p1t.y, p0t.y, p1t.x);

    // Fall back to double-precision test at triangle edges
    if (sizeof(Float) == sizeof(float) &&
        (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (float)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (float)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (float)(p1typ0tx - p1txp0ty);
    }

    // Perform triangle edge and determinant tests
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return {};
    Float det = e0 + e1 + e2;
    if (det == 0)
        return {};

    // Compute scaled hit distance to triangle and test against ray $t$ range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < tMax * det))
        return {};
    else if (det > 0 && (tScaled <= 0 || tScaled > tMax * det))
        return {};

    // Compute barycentric coordinates and $t$ value for triangle intersection
    Float invDet = 1 / det;
    Float b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
    Float t = tScaled * invDet;
    DCHECK(!IsNaN(t));

    // Ensure that computed triangle $t$ is conservatively greater than zero
    // Compute $\delta_z$ term for triangle $t$ error bounds
    Float maxZt = MaxComponentValue(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;

    // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
    Float maxXt = MaxComponentValue(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponentValue(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);

    // Compute $\delta_e$ term for triangle $t$ error bounds
    Float deltaE =
        2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
    Float maxE = MaxComponentValue(Abs(Vector3f(e0, e1, e2)));
    Float deltaT = 3 *
                   (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
                   std::abs(invDet);
    if (t <= deltaT)
        return {};

    // Return _TriangleIntersection_ for intersection
    return TriangleIntersection{b0, b1, b2, t};
}
