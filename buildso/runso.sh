
rm add.o  a.out  libadd.so  Mainadd  main.o

hipcc -fPIC -c add.hip.cpp -o add.o
hipcc -shared add.o -o libadd.so

hipcc -c main.cpp -o main.o
mpicc  -L. -ladd main.o  -o Mainadd
./Mainadd

