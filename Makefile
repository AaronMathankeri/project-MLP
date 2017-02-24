CC=icpc
FLAGS=-std=c++11 -mkl

all: compile clean

compile:
	${CC} ${FLAGS} main.cpp -o main_exec

clean:
	mv main_exec bin/

