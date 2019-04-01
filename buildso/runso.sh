
rm add.o  a.out  libadd.so  Mainadd  main.o

hipcc -fPIC -c add.hip.cpp -o add.o
hipcc -shared add.o -o libadd.so

hipcc -c main.cpp -o main.o
hipcc main.o  -L. -ladd -o Mainadd
./Mainadd

