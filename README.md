# emida
Employing GPU to Process Data from Electron Microscope

# How to build

The project uses the CMake build system.

Requirements:
- git
- CMake version at least 3.9
- C++17 compiler (tested on MSVC from Visual Studio 16.8, GCC 8.3 and GCC 9)
- CUDA compute capability: 6.0

Tested on 2 platforms so far:
- Windows - Visual Studio 2019, CUDA 10.2
- Linux - GCC 8.3, CUDA 10.2


## Windows

1. Open the project folder in Visual Studio. It should detect CMake project and start configuration.
2. Choose the x64-Release configuration.
3. Build -> Build all (F6)
4. The resulting executable emida.exe is in `build/x64-Release/bin` folder.

You may also run `build/x64-Release/bin/emida_test.exe` to verify the build.

## Linux

On linux, it is preferred to use system libtiff (it is downloaded and built from source on Windows). Alternatively, there is `TIFF_FROM_SOURCE` switch that enforces building libtiff from source, but ZLIB and LibLZMA packages are still required by the libtiff library.
The following steps are an example how to build the project on Ubuntu. We assume that a compatible C++17 compiler is set as default. 

1. Install the libtiff, cmake and git packages:
   apt-get install -y cmake git libtiff-dev
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
Then, list of pairs follow, each of them specifies a middle of one subregion.


## Running emida

The `emida` executable cross-correlates parts of pictures(slices) in specified positions and then computes how much are the deformed picture's slices shifted compared to the initial picture.
```
emida -i data/FeAl/INITIAL_FeAl -d data/FeAl/DEFORMED_FeAl -o data/FeAl/OUT_FeAl -r 0,0,5,5 -c 25,25 -s 64,64 > offsets.txt
```

Following options are mandatory:
- `-i,--initial` specifies path to the reference image
- `-d,--deformed` specifies path to file with list of the deformed pictures to process. The format of the file is described above.
- `-b,--slicepos` specifies path to file with positions of regions to be compared in each picture, as descibed above.

Optional options:

- `-s,--slicesize` overrides the size of subregions specified in the subregion description passed by the `slicepos` parameter.
- `-q,--writecoefs` In addition to offsets, output also coefficients of parabola fitting.
- `--precision` specifies the floating type to be used. Allowed values: `double`, `float`.
- `f,fitsize` specifies the size of neighbourhood that is used to fit the quadratic function to cross--correlation and find subpixel maximum. Allowed values: 3, 5, 7, 9
- `crosspolicy`, Specifies whether to use FFT to compute cross correlation. Allowed values: brute, fft.
- `batchsize`, Specifies how many files are processed in one batch in parallel.
- `loadworkers`, "Specifies the number of workers that load input patterns simultaneously.
- `-a` When specified, the executable measures the duration of selected kernels and parts of processing.

## Test data
The package contains testing data in the `test_data` folder.  `test_data/initial` contains the reference, unmodified pattern. `test_data/deformed` contains the deformed patterns. Due to limited options of uploading large attachments, we attached only a subset of original testing data. `test_data/def.txt` contains list of the files in the format described above that can be loaded by the executable.

The deformed images have file names in the following format:
```
DEFORMED_x<X_position>y<X_position>.tif
```
The `X_position` and `Y_position` are integers that determine the position where the pattern was measured on the studied specimen.

The patterns were measured in a triangle raster so the `X_position` and `Y_position` can be iterated as follows:

```
x = i*60 + (j % 2) * 60/2
y = int(j*sqrt(0.75)*60)
```

We have prepared a python script `TEST_RUN/run.py`, that runs the executable on the test data. The only required change to the script is to set the correct path to the executable.

We also prepared a script in `test_data/expand_data.py` that copies some of the test files to create a larger dataset. Just set the size of the dataset directly in the script. In order to run the application on the larger dataset, set the same size in `TEST_RUN/run.py`.
