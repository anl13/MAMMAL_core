Download glad from https://github.com/Dav1dde/glad, and you know how to do. 
It compiles as static library. As you use it, you may come up with error similar to 

```Shell
/usr/bin/ld: //usr/local/lib/libglad.a(glad.c.o): undefined reference to symbol 'dlclose@@GLIBC_2.2.5'
//lib/x86_64-linux-gnu/libdl.so.2: error adding symbols: DSO missing from command line
collect2: error: ld returned 1 exit status
CMakeFiles/render.dir/build.make:250: recipe for target 'render' failed
make[2]: *** [render] Error 1
```
This is because `glfw` you use is dynamic library, which is imcompatible with `glad` you compiled. So, just download `glfw` and compile it as static library and reinstall it (Do not need to delete old one)