CC=icpc
FLAGS=-std=c++11 -mkl -I ./include/
src = $(wildcard ./src/*.cpp)

all: compile clean

compile:
	${CC} ${FLAGS} ${src} -o main_exec
clean:
	mv main_exec bin/

