# MAMMAL core

This repository contains the core code of MAMMAL system. The system is written with C++17. 

# Environment
The code has been tested on Windows 10 system with NVIDIA RTX 2080Ti GPU and CUDA 10.2. 

After unzip this project, you need to download some necessary data from [Google Drive](https://drive.google.com/file/d/1m9hKCMhI_VJb1muM-sS_01TgYdDqaz3D/view?usp=sharing). The file you download is `data.zip`, just unzip it and put it under the MAMMAL_core main folder. Besides, download my pre-compiled third-party libraries from the link (https://drive.google.com/file/d/1Df-p3nQLE5lPo999eOvaKY59sMQurmW9/view?usp=sharing). Then unzip the third-party libraries and put it under the MAMMAL_core main folder. You will get a folder structure like  
```
\-- MAMMAL_core\
    |-- 3rdparty\
    |-- data\
    |-- annotator\
    |-- articulation\
    |-- configs\
    |-- posesolver\
    |-- props\
    |-- render\
    |-- utils\
    |-- .gitignore
    |-- README.md
    |-- MAMMAL_core.sln
```
Then, download data files for render from [Google Drive](https://drive.google.com/file/d/1xZqepoemvG6aPBnMBn69cvaUKz5JMlZd/view?usp=sharing) and unzip it under `render/` folder like 
```
\-- MAMMAL_core\
    \-- 3rdparty\
    |-- data\
    |-- annotator\
    |-- articulation\
    |-- configs\
    |-- posesolver\
    |-- props\
    |-- render\
        \-- data\
        |-- shader\
    |-- utils\
    |-- .gitignore
    |-- README.md
    |-- MAMMAL_core.sln
```
To open and run the project, we recommend to download and install Visual Studio 2017 Community from their official website (https://visualstudio.microsoft.com/). After installing Visual Studio 2017, you should install CUDA Runtime API 10.2 (or higher version. However, if you use higher version, you need to change some property configurations after opening the project. ). To install it, you should download from NVIDIA developer website (https://developer.nvidia.com/cuda-10.2-download-archive) and run the .exe file directly. Usually, both Visual Studio 2017 and CUDA are installed under C:/ drive by default. After installing above softwares, you can open the project now!

# Open the project
By simply clicking `MAMMAL_core.sln` file, the Visual Studio 2017 would open it.
Then, you will see five `projects` under the `MAMMAL_core` solution: 
```
annotator
articulation
posesolver
render
utils
```
You can choose to set one of `annotator`, `articulation`, `posesolver` or `render` as the "start project". We recommend to try `render` first because its dependencies are the least. The, set the project compile properties as `Release` + `x64`. Finally, click the green triangle button to compile and run the project. If everything goes right, you will see a rendering of pig liveing environment model. 


# Run demo on BamaPig3D dataset 
The tutorial on how to run demo on BamaPig3D will be provided soon. 