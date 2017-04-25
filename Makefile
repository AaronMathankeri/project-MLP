CC=icpc
FLAGS=-std=c++11 -mkl -I ./include/ -Wl,-rpath,${MKLROOT}/lib -L${MKLROOT}/lib -liomp5
src = $(wildcard ./src/*.cpp)

all: compile clean run

compile:
	${CC} ${FLAGS} ${src} -o main_exec
clean:
	mv main_exec bin/
run:
	./bin/main_exec
