CC=icpc
FLAGS=-std=c++11 -mkl -L /opt/intel/mkl/lib/ 

all: compile clean

compile:
	${CC} ${FLAGS} main.cpp -o main_exec

clean:
	mv main_exec bin/

