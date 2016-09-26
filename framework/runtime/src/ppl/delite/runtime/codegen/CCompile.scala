package ppl.delite.runtime.codegen

import collection.mutable.ArrayBuffer
import ppl.delite.runtime.Config
import xml.XML
import java.io.FileNotFoundException
import tools.nsc.io.{Directory, Path, File}
import ppl.delite.runtime.graph.targets.{OS, Targets}
import ppl.delite.runtime.graph.ops._
import java.io.FileWriter

trait CCompile extends CodeCache {
  
  protected class CompilerConfig(
    val compiler: String, 
    val make: String,
    val headerDir: Array[String],
    val sourceHeader: Array[String],
    val libs: Array[String],
    val headerPrefix: String, 
    val libPrefix: String
  )
	//
  protected def configFile: String // name of file, will always be searched for inside config/delite/
  protected def compileFlags: Array[String] // machine-independent flags that are always passed to the compiler for this lib
  protected def linkFlags: Array[String] // machine-independent flags that are always passed to the linker for this lib
  protected def outputSwitch: String // compiler parameter that allows us to specify destination dir (e.g. -o)
  protected lazy val config = loadConfig(configFile)
  //
  private val headerBuffer = new ArrayBuffer[(String, String)]
  private val kernelBuffer = new ArrayBuffer[String] // list of kernel filenames to be included in the compilation (non-multiloop kernels)
  protected def auxSourceList = List[String]() // additional source filenames to be included in the compilation
	//
  def headers = headerBuffer.map(_._2)

  protected def deliteLibs = Config.deliteBuildHome + sep + "libraries" + sep + target

  def addKernel(op: DeliteOP) { 
    op match {
      case _:EOP | _:Arguments => // kernelBuffer is used to hold the actual generated kernels to compile 
      case _ if kernelBuffer.contains (op.id + "." + ext) => // same kernel may be used multiple times in a deg 
      case _ => kernelBuffer += (op.id + "." + ext)
    }
  }

  def addHeader(source: String, name: String) {
    if (!headerBuffer.contains((source, name+".h")))
      headerBuffer += Pair(source, name+".h")
  }

  protected def loadConfig(f: String): CompilerConfig = {
    // parse XML, return configuration
    val configFile = File(Config.deliteHome + sep + "config" + sep + "delite" + sep + f)
    if (!configFile.isValid) throw new FileNotFoundException("could not load compiler configuration: " + configFile)

    val body = XML.loadFile(configFile.jfile)
    val compiler = (body \\ "compiler").text.trim
    val make = (body \\ "make").text.trim
    val headerPrefix = (body \\ "headers" \ "prefix").text.trim
    val headerDir = body \\ "headers" flatMap { e => val prefix = e \ "prefix"; e \\ "path" map { prefix.text.trim + _.text.trim } } toArray
    val sourceHeader = body \\ "headers" flatMap { e => e \\ "include" map { _.text.trim } } toArray
    val libPrefix = (body \\ "libs" \ "prefix").text.trim
    val libs = body \\ "libs" flatMap { e => val prefix = e \ "prefix"; (e \\ "path").map(p => prefix.text.trim + p.text.trim) ++ (e \\ "library").map(l => l.text.trim) } toArray

		println("Make Command::"+make)
		println("Compiler Command::"+compiler)
		//TODO: Run a QMake COmmand Here....
    new CompilerConfig(compiler, make, headerDir, sourceHeader, libs, headerPrefix, libPrefix)
  }

  def compile() {
    if (sourceBuffer.length == 0) return
    cacheRuntimeSources((sourceBuffer ++ headerBuffer).toArray)
    //TODO: Had to edit this to make it work with Qt QMake project file
    //println("HeaderPrefix:::"+config.headerDir.mkString(" "))
    // Array("/home/optisdr/OptiSDR/jdk1.8.0_31/include","/home/optisdr/OptiSDR/jdk1.8.0_31/include/linux")
    //
    val headersDir: Array[String] = new Array[String](config.headerDir.length)
    for(i<- 0 until config.headerDir.length)
    {
    	val tmp = config.headerDir(i).split("-I")
    	headersDir(i) = tmp(1);
    }
    if (modules.exists(_.needsCompile)) {
      val includes2 = modules.flatMap(m => List(config.headerPrefix + sourceCacheHome + m.name, config.headerPrefix + Compilers(Targets.getHostTarget(target)).sourceCacheHome + m.name)).toArray ++ 
                     config.headerDir ++ Array(config.headerPrefix + staticResources)
      val includes = modules.flatMap(m => List(sourceCacheHome + m.name, Compilers(Targets.getHostTarget(target)).sourceCacheHome + m.name)).toArray ++ 
                     headersDir ++ Array(staticResources) //
      val libs = config.libs ++ Directory(deliteLibs).files.withFilter(f => f.extension == OS.libExt || f.extension == OS.objExt).map(_.path)
      val sources = (sourceBuffer.map(s => sourceCacheHome + "runtime" + sep + s._2) ++ kernelBuffer.map(k => sourceCacheHome + "kernels" + sep + k) ++ auxSourceList).toArray
      val degName = ppl.delite.runtime.Delite.inputArgs(0).split('.')
      val dest = binCacheHome + target + "Host" + degName(degName.length-2) + "." + OS.libExt
      compile(dest, sources, includes2, libs)
    }
    sourceBuffer.clear()
    headerBuffer.clear()
    kernelBuffer.clear()
  }

  def compileInit() {
    val root = staticResources + sep + target + "Init."
    val source = root + ext
    val dest = root + OS.libExt
    compile(dest, Array(source), config.headerDir, Array[String]())
  }

  // emit Makefile and call make
  def compile(destination: String, sources: Array[String], includes: Array[String], libs: Array[String]) {
    val destDir = Path(destination).parent
    destDir.createDirectory()

    // generate Makefile
    val makefile = destDir + File.separator + "Makefile"
    if(!Config.noRegenerate) {
      val writer = new FileWriter(makefile)
      writer.write(makefileString(destination, sources, includes, libs))
      writer.close()
    }
    // TODO: Generate QMake file
    /*val qmakefile = destDir + File.separator + "QMakefile.pro"
    if(!Config.noRegenerate && Config.numCuda>0)
    {
      val writer = new FileWriter(qmakefile)
      println("CUDA Device SM::"+compileFlags(6))
      writer.write(qmakefileString(destination, sources, includes, libs,compileFlags(6)))
      writer.close()
    }
		val qmkf = destDir+"/QMakefile.pro"
		//
    val cpargs = Array("/bin/sh", "-c","cp "+qmkf+"  $PWD/QMakefile.pro")
    var cpprocess = Runtime.getRuntime.exec(cpargs)
    cpprocess.waitFor
    checkError(cpprocess,cpargs)
		//
		//TODO: Running QMake.
		//
    val qmargs = Array("/bin/sh", "-c","cd "+destDir+"; qmake "+qmkf)
    var qmprocess = Runtime.getRuntime.exec(qmargs)
    qmprocess.waitFor
    checkError(qmprocess,qmargs)*/
    //
    if (config.compiler == "")
      throw new RuntimeException("compiler path is not set. Please specify in $DELITE_HOME/config/delite/" + configFile + " (<compiler> </compiler>)")
    if (config.make == "")
      throw new RuntimeException("make command path is not set. Please specify in $DELITE_HOME/config/delite/" + configFile + " (<make> </make>)")
    if (config.headerDir.length == 0)
      throw new RuntimeException("JNI header paths are not set. Please specify in $DELITE_HOME/config/delite/" + configFile + " (<headers> </headers>)")
		//
    //TODO: How many parallel jobs? For now, the number of processors.
    val args = Array(config.make, "-s", "-j", Runtime.getRuntime.availableProcessors.toString, "-f", makefile, "all")
    val process = Runtime.getRuntime.exec(args)
    process.waitFor
    checkError(process, args)
  }

  protected def checkError(process: Process, args: Array[String]) {
    val errorStream = process.getErrorStream
    val inputStream = process.getInputStream
    val out = new StringBuilder

    var err = errorStream.read()
    if (err != -1) {
      out.append("--" + target + " compile args: " + args.mkString(","))
      while (err != -1) {
        out.append(err.asInstanceOf[Char])
        err = errorStream.read()
      }
      out.append('\n')
    }

    var in = inputStream.read()
    if (in != -1) {
      while (in != -1) {
        out.append(in.asInstanceOf[Char])
        in = inputStream.read()
      }
      out.append('\n')
    }

    if (process.exitValue != 0) {
      sourceBuffer.clear()
      headerBuffer.clear()
      kernelBuffer.clear()
      if(Config.clusterMode == 2) // Send the error message to the master node
        ppl.delite.runtime.DeliteMesosExecutor.sendDebugMessage(out.toString)
      else 
        println(out.toString)
      sys.error(target + " compilation failed with exit value " + process.exitValue)
    }
  }
	// String for QMakefile.pro
	def qmakefileString(destination: String, sources: Array[String], includes: Array[String], libs: Array[String],sm: String) = """
## Qt Project File: Generated by Delite/OptiSDR Runtime ##
CC = %1$s
DELITE_HOME = %2$s
SOURCECACHE_HOME = %3$s
BINCACHE_HOME = %4$s
INCLUDES = %5$s
INCLUDEPATH += %5$s
#CFLAGS = %6$s
#LDFLAGS = %7$s
SOURCES = %8$s
#OBJECTS = $(SOURCES:.%9$s=.o)
OUTPUT = %10$s
DESTDIR     = $$BINCACHE_HOME
############
QT       += core
QT       += gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport
#
TARGET    = %10$s
CONFIG   += console
CONFIG   -= app_bundle
#
TEMPLATE  = app
#
# C++ source code
SOURCES += $$DELITE_HOME/runtime/src/static/cuda/qcustomplot.cpp
#
HEADERS  += $$DELITE_HOME/runtime/src/static/cuda/qcustomplot.h
#
FORMS    += $$DELITE_HOME/runtime/src/static/cuda/mainwindow.ui
#
# Cuda sources
CUDA_SOURCES += $$SOURCES
CUDA_HEARDERS += $$INCLUDES
#
# project build directories
DESTDIR     = $$BINCACHE_HOME
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-O3
#
# Path to cuda toolkit install
CUDA_DIR      = /usr
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
INCLUDEPATH  += $$INCLUDES
QMAKE_LIBDIR += $$CUDA_DIR/lib64    
# Note I am using a 64 bits Operating system
# libs used in your code
LIBS += -lcudart -lcuda -lcufft
# GPU architecture - TODO Remember to make this variable - sm_20
CUDA_ARCH     = %13$s
#
# NVCC Flags
NVCCFLAGS = -shared -Xcompiler '-fPIC' --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
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
#
QMAKE_CXXFLAGS		+= -lncurses
QMAKE_CXXFLAGS		+= -luhd
QMAKE_CXXFLAGS		+= -lboost_program_options
QMAKE_CXXFLAGS		+= -lboost_system
QMAKE_CXXFLAGS		+= -lboost_thread
QMAKE_CXXFLAGS		+= -lboost_date_time
QMAKE_CXXFLAGS		+= -lboost_filesystem
QMAKE_CXXFLAGS		+= -lboost_regex
QMAKE_CXXFLAGS		+= -lboost_serialization
#
QMAKE_LFLAGS   +=  -fopenmp
#
QMAKE_LFLAGS		+= -lncurses
QMAKE_LFLAGS		+= -luhd
QMAKE_LFLAGS		+= -lboost_program_options
QMAKE_LFLAGS		+= -lboost_system
QMAKE_LFLAGS		+= -lboost_thread
QMAKE_LFLAGS		+= -lboost_date_time
QMAKE_LFLAGS		+= -lboost_filesystem
QMAKE_LFLAGS		+= -lboost_regex
QMAKE_LFLAGS		+= -lboost_serialization
#
LIBS += -fopenmp
#
LIBS		+= -lncurses
LIBS		+= -luhd
LIBS		+= -lboost_program_options
LIBS		+= -lboost_system
LIBS		+= -lboost_thread
LIBS		+= -lboost_date_time
LIBS		+= -lboost_filesystem
LIBS		+= -lboost_regex
LIBS		+= -lboost_serialization
#
""".format(config.compiler,Config.deliteHome,sourceCacheHome,binCacheHome,includes.mkString(" "),
           compileFlags.mkString(" "),(linkFlags++libs).mkString(" "),sources.mkString(" "),ext,destination,
           Config.numCpp,Config.cppMemMgr.toUpperCase,sm)
  // string for the Makefile
  def makefileString(destination: String, sources: Array[String], includes: Array[String], libs: Array[String]) = """
## Makefile: Generated by Delite Runtime ##
CC = %1$s
DELITE_HOME = %2$s
SOURCECACHE_HOME = %3$s
BINCACHE_HOME = %4$s
INCLUDES = %5$s

CFLAGS = %6$s
LDFLAGS = %7$s
SOURCES = %8$s
OBJECTS = $(SOURCES:.%9$s=.o)
OUTPUT = %10$s

#COMPILER_FLAGS	=  --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -D_FORCE_INLINES

all: $(OUTPUT)

# The order of objects and libraries matter because of the dependencies
$(OUTPUT): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(OUTPUT)

%%.o: %%.%9$s
	$(CC) -c -DDELITE_CPP=%11$s $(INCLUDES) $(CFLAGS) -D_FORCE_INLINES $< -o $@

clean:
	rm -f $(OBJECTS) $(OUTPUT)

.PHONY: all clean
""".format(config.compiler,Config.deliteHome,sourceCacheHome,binCacheHome,includes.mkString(" "),
           compileFlags.mkString(" "),(linkFlags++libs).mkString(" "),sources.mkString(" "),ext,destination,
           Config.numCpp,Config.cppMemMgr.toUpperCase)
}