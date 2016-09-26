QT       += core
QT       += gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport
#
TARGET    = OptiSDRQtCuda
CONFIG   += console
CONFIG   -= app_bundle
#
TEMPLATE  = app
#
# C++ source code
SOURCES += main.cpp\
           mainwindow.cpp \
           qcustomplot.cpp
#
HEADERS  += mainwindow.h \
         qcustomplot.h \
	 OptiSDRCuda.h \
	netradread.h \
	optisdrdevices.h \
	radardsp.h
#
FORMS    += mainwindow.ui
#
# Cuda sources
CUDA_SOURCES += OptiSDRCuda.cu
CUDA_HEARDERS += OptiSDRCuda.h \
		optisdrdevices.h \
		radardsp.h
#
# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3
#
# Path to cuda toolkit install
CUDA_DIR      = /usr
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# Note I am using a 64 bits Operating system
# libs used in your code
LIBS += -lcudart -lcuda -lcufft
# GPU architecture
CUDA_ARCH     = sm_50
#
# NVCC Flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -D_FORCE_INLINES
#
# Prepare the extra compiler configuration (taken from the nvidia forum - i am not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
#
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
#
cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
#
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
#
#
# OpenMP Flags and Libs
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS   +=  -fopenmp
LIBS += -fopenmp
