xcrun xctrace record \
  --template 'Time Profiler' \
  --output myprofile.trace \
  --target-stdout output.txt \
  --launch -- ./test_wos_bonsai \
    --dim=3 \
    --file=/Users/ajroot/projects/zombie-mirror/test/inputs/bunny.msh \
    --solveDoubleSided \
    --runSingleThreaded