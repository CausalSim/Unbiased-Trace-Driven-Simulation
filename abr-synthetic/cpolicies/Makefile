all: clean build

default: build

############### FILES #################
FILES=mpc.pyx

############## COMMANDS ###############
build: ${FILES}
	python3 setup.py build_ext -i

clean:
	rm -rf build
	rm -f *.cpython*.so mpc.cpp
#######################################