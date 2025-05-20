# Collision Detection

This runs collision detection benchmarks, comparing to [FCL](https://github.com/flexible-collision-library/fcl).

## installation

Installation requires the `bonsai` compiler, [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), 
and [fcl](https://github.com/flexible-collision-library/fcl). FCL has additional requirements as well, namely 
[libccd](https://github.com/danfis/libccd). The octomap library is *not* necessary and can be omitted if built 
using Cmake: `-DBUILD_OCTOMAP=OFF`.

## run the benchmarks

```bash
cd apps/cd/cpu/fcl/
./test.sh
```

Wavefront OBJ files are taken from [FCL](https://github.com/flexible-collision-library/fcl/tree/a3fbc9fe4f619d7bb1117dc137daa497d2de454b/test/fcl_resources) 
and from [`prims`](https://github.com/nickdesaulniers/prims/tree/master/meshes), the script assumes these are placed in the following directory: 

```
apps/cd/cpu/fcl/objects/
```