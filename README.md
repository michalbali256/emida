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
The application processes files in the following form:
```
INITIAL_x0y0.tif
INITIAL_x60y0.tif
INITIAL_x120y0.tif
...
INITIAL_x30y51.tif
INITIAL_x90y51.tif
INITIAL_x150y51.tif
...
INITIAL_x0y103.tif
INITIAL_x60y103.tif
INITIAL_x120y103.tif
...
```
x and y are organised in a triangle raster:
```
x = i*60 + (j % 2) * 60/2          (1)
y = int(j*sqrt(0.75)*60)           (2)
```

The application takes two directories with such organisation:
1. initial pictures, all files must have prefix INITIAL_ followed by x and y position.
2. deformed pictures, all files must have prefix DEFORMED_ followed by x and y position.

## Running emida

The `emida` executable cross-correlates parts of pictures(slices) in specified positions and then computes how much are the deformed picture's slices shifted compared to the initial picture.
```
emida -i data/FeAl/INITIAL_FeAl -d data/FeAl/DEFORMED_FeAl -o data/FeAl/OUT_FeAl -p 873,873 -r 0,0,5,5 -c 25,25 -s 64,64 > offsets.txt
```

Following options are mandatory:
- `-i,--initial` specifies folder with the initial pictures
- `-d,--deformed` specifies folder with the deformed pictures
- `-r,--range` specifies range of pictures to be processed in the form `I_BEGIN,J_BEGIN,I_END,J_END`. It is specified in terms of `i` and `j` from the (1) and (2) formulae. For example range 0,0,3,3 would specify exactly the files listed in the example.
- `-p,--picsize` specifies size of input initial and deformed tiff pictures. All pictures should have the same size.

Optional options:
- `-o,--outpics` specifies folder to which pictures with resulting offsets will be written.
- `-s,--slicesize` specifies size of parts of pictures that will be cross-correlated against each other - slices. Default is 64,64.
- `-c,--crosssize` specifies size of neigbourhood of picture's center that will be analysed with cross correlation in the form X_SIZE,Y_SIZE. Both X_SIZE and Y_SIZE must be odd numbers. Small neighbourhood is faster to compute, but you may miss the best offset fit. Bigger neighbourhoods are faster to compute. By default, it is `2*slicesize-1`, so the cross correlation is done on whole slice.


By default, the slices are evenly distributed across the picture every 32 pixels (can be changed by `--slicestep`). You can also provide custom positions of the slices by writing a file that lists positions of the slices in the form `X_POS,Y_POS`. It looks like this:
```
50,300
100,300
150,300
200,300
250,300
50,350
```
The example specifies only 6 slices that would be cross-correlated in each picture. The path to the file can be specified using the `-b` option. An example file is `begins.txt` in the repository.


## Process offsets

The `python/process_offsets.py` then can be used to compute matrix of linear transformation for each input picture.

```
emida -i data/FeAl/INITIAL_FeAl -d data/FeAl/DEFORMED_FeAl -r 0,0,5,5 -c 25,25 | py ..\..\python\process_offsets.py
```