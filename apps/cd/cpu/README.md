# Collision Detection

This runs collision detection benchmarks, comparing to [FCL](https://github.com/flexible-collision-library/fcl).

## installation

Installation requires the `bonsai` compiler, [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), 
and [fcl](https://github.com/flexible-collision-library/fcl). FCL has additional requirements as well, namely 
[libccd](https://github.com/danfis/libccd). The octomap library is *not* necessary and can be omitted if built 
using Cmake: `-DBUILD_OCTOMAP=OFF`.

## run the benchmarks

```bash
./apps/cd/test.sh
```
