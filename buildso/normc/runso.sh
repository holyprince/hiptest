
hipcc -fPIC -c maxtest.cpp -o maxtest.o
hipcc -shared maxtest.o -o libmaxtest.so

hipcc -c mainmax.cpp -o mainmax.o
hipcc mainmax.o  -L. -lmaxtest -o Mainmax
./Mainmax

