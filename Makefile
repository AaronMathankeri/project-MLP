CC=icpc
FLAGS=-std=c++11 -mkl

all: compile clean

compile:
	${CC} ${FLAGS} ./src/main.cpp \
		./src/ioFunctions.cpp \
		./src/initializations.cpp \
		./src/networkAgnosticFunctions.cpp \
		./src/backpropFunctions.cpp \
		./src/feedForwardFunctions.cpp \
		./src/gradientDescent.cpp -o main_exec 

clean:
	mv main_exec bin/

