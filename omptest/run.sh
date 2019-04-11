
#hipcc -fopenmp -I/work/home/wangzh/software/llvmopenmp/include  -l/work/home/wangzh/software/llvmopenmp/lib/libomp.so  main.cpp
hipcc -I/work/home/wangzh/software/llvmopenmp/include -L/work/home/wangzh/software/llvmopenmp/lib -lomp main.cpp

