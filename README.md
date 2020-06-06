# emida
Employing GPU to Process Data from Electron Microscope


# How to build

Requirements:
- C++17 compiler (tested on MSVC from VS 16.4 and GCC 8.3)
- CUDA compute capability: 6.0

Tested on 2 platforms so far:
- Windows - Visual Studio 2019, CUDA 10.2
- Linux - GCC 8.3, CUDA 10.2

## Windows

1. Open the project folder in Visual Studio. It should detect CMake project and start configuration.
2. Choose the x64-Release configuration.
3. Build -> Build all (F6)
4. The resulting executable emida.exe is in `build/x64-Release/bin` folder.

## Linux

Prerequisities for linux build: ZLIB and LibLZMA (required by libtiff)

1. Run the following in the terminal from the project folder.
    ```
    mkdir build && cd build
    cmake ../
    cmake --build .
    ```
2. The resulting executable emida is in `build/bin` folder.


You may also run `bin/emida_test` to verify the build.

# How to run
The application compares one picture of the material patern with images of deformed material. The deformed images are specified by an input file that has lines in following format:
```
<image_pos_x> <image_pos_y> <image_file_name>
```
Example:
```
0.0 0.0 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x0y0.tif
600.0 0.0 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x600y0.tif
1200.0 0.0 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x1200y0.tif
1800.0 0.0 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x1800y0.tif
300.0 519.6152422706632 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x300y519.tif
900.0 519.6152422706632 E:/emida/data/FeAl/DEFORMED_FeAl/DEFORMED_x900y519.tif
```

Positions and size of regions that are compared in each picture are specified in a configuration file in fillowing format:
```
<size/2>\n
<roi_mid_x> <roi_mid_y>\n
<roi_mid_x> <roi_mid_y>\n
...
```
The size of each region is specified on the first line. The size is half ("radius") of the compared regions.


## Running emida

The `emida` executable cross-correlates parts of pictures(slices) in specified positions and then computes how much are the deformed picture's slices shifted compared to the initial picture.
```
emida -i data/FeAl/INITIAL_FeAl -d data/FeAl/DEFORMED_FeAl -o data/FeAl/OUT_FeAl -p 873,873 -r 0,0,5,5 -c 25,25 -s 64,64 > offsets.txt
```

Following options are mandatory:
- `-i,--initial` specifies path to the reference image
- `-d,--deformed` specifies path to file with list of the deformed pictures to process. The format of the file is described above.

Optional options:
- `-b,--slicepos` specifies path to file with positions of regions to be compared in each picture, as descibed above.
- `-o,--outpics` specifies folder to which pictures with resulting offsets will be written.
- `-s,--slicesize` specifies size of parts of pictures that will be cross-correlated against each other - slices. Default is 64,64.
- `-c,--crosssize` specifies size of neigbourhood of picture's center that will be analysed with cross correlation in the form X_SIZE,Y_SIZE. Both X_SIZE and Y_SIZE must be odd numbers. Small neighbourhood is faster to compute, but you may miss the best offset fit. Bigger neighbourhoods are faster to compute. By default, it is `2*slicesize-1`, so the cross correlation is done on whole slice.
- `-q` In addition to offsets, output also coefficients of parabola fitting.
- `--precision` specifies the floating type to be used. Allowed values: `double`, `float`.


## Process offsets

The `python/process_offsets.py` then can be used to compute matrix of linear transformation for each input picture.

```
emida -i data/FeAl/INITIAL_FeAl -d data/FeAl/DEFORMED_FeAl -r 0,0,5,5 -c 25,25 | py ..\..\python\process_offsets.py
```