# 20190820
## libgomp
when linking to `libceres.a`, I come up with some problems: 
```
/usr/bin/ld: /usr/local/lib/libceres.a(schur_eliminator_2_3_4.cc.o): undefined reference to symbol 'GOMP_loop_dynamic_start@@GOMP_1.0'
//usr/lib/x86_64-linux-gnu/libgomp.so.1: error adding symbols: DSO missing from command line
collect2: error: ld returned 1 exit status
CMakeFiles/calib.dir/build.make:189: recipe for target 'calib' failed
make[2]: *** [calib] Error 1
CMakeFiles/Makefile2:72: recipe for target 'CMakeFiles/calib.dir/all' failed
make[1]: *** [CMakeFiles/calib.dir/all] Error 2
Makefile:83: recipe for target 'all' failed
make: *** [all] Error 2
```

I **fix** this by manually refer to `libgomp.so.1` using `link_libraries`. 

## glog and gflag
I come up with many glog errors when compiling. I decide to re-install glog, gflag, and ceres (with suite sparse). 

I first install `glog` following the instructions of https://github.com/google/glog. Before run `autogen.sh`, install tools: `sudo apt-get install autogen autoconf libtool` (https://github.com/hishamhm/htop/issues/439). *BUT I FAILED!* due to wired bugs of `libtool`. 
So I instal them via `sudo apt-get install libgoogle-glog-dev libgflags-dev` 

## blas, lapack 
Just download http://www.netlib.org/lapack/, and follow `README.md`. 

## suitesparse 
VERY IMPORTANT for bundle adjustment. 

First I need to install openblas by `sudo apt-get install libopenblas-dev`

Then download suitesparse, and `make -j4` to build. 

install suitesparse at `/home/library4/SuiteSparse`, including `include` and `lib` dir. 

Infact, I finally install `libsuitesparse-dev` under the direction of `ceres-solver` document. 

## ceres 
