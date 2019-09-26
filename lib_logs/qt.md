# 20190916
when you compile your executable with opencv library support, you may need qt as some backend. 

I run into a bug with `imshow`: 
```Shell
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
```
This error appears at running your app. 

To solve it, 
`cp -r /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms build/.`
