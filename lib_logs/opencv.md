https://cv-tricks.com/installation/opencv-4-1-ubuntu18-04/

https://www.learnopencv.com/install-opencv-4-on-ubuntu-16-04/

2020/09/29
在windows上安装opencv4.5记录：
首先下载好opencv和opencv_contrib。
在cmake-gui中进行配置，配置的时候要注意，选择的编译器要是64位的。
这样在configure的时候，就能找到64位的cuda相关的库，否则可能会遇到错误。
生成sln之后，直接打开编译即可。有一个INSTALL工程，找到编译它，就能install了。