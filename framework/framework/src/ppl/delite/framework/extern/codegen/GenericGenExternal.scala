package ppl.delite.framework.extern.codegen

import _root_.scala.virtualization.lms.common._
import _root_.scala.virtualization.lms.internal._
import collection.mutable.{ListBuffer, HashSet, HashMap}
import java.io.{FileWriter, BufferedWriter, File, PrintWriter}

import ppl.delite.framework.{Config, DeliteApplication}
import ppl.delite.framework.extern.lib._
import ppl.delite.framework.ops._
import ppl.delite.framework.codegen.delite._

trait GenericGenExternal extends GenericNestedCodegen {
  val IR: DeliteOpsExp
  import IR._

  val generatedOps = HashSet[String]()
    
  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case e:DeliteOpExternal[_] =>
      // generate library once only
      if (!generatedOps.contains(e.funcName)) {
        try {
          emitExternalLib(e)
          generatedOps += e.funcName
        }
        catch {
          case g: GenerationFailedException =>
        }        
      }      
      // generate external call for this node
      emitExternalNode(sym, e)
      
    case _ => super.emitNode(sym, rhs) 
  }
  
  // external generators implement emitExternalNode instead of emitNode
  def emitExternalNode(sym: Sym[Any], rhs: Def[Any]): Unit = {
    throw new GenerationFailedException("don't know how to generate external call for " + rhs)
  }
  
  def emitExternalLib(rhs: Def[Any]): Unit = {
    throw new GenerationFailedException("don't know how to generate external lib for " + rhs)
  }
        
  ///////////////////
  // generator config
  
  /* location for generated interface objects */
  lazy val headerDir = new File(Config.buildDir + "/" + this.toString + "/kernels/")
  
  /* location for generated native method wrappers */
  lazy val nativeDir = new File(Config.buildDir + "/native/" + this.toString)
  
  /* location for compiled .so shared libraries */
  lazy val libDir = new File(Config.buildDir + "/libraries/" + this.toString)
  
  def libInterfaceHdr(lib: ExternalLibrary) = ""
  def libInterfaceFtr(lib: ExternalLibrary) = ""
  def hdrName(lib: ExternalLibrary) = lib.name  // default header file name is library name
  val hdrExt: String // generator specific interface file / header file extension
  
  override def initializeGenerator(buildDir:String, args: Array[String]): Unit = {
    headerDir.mkdirs()
    nativeDir.mkdirs() 
    
    libDir.mkdirs()

    super.initializeGenerator(buildDir, args)
  }
  
  override def finalizeGenerator() {
    interfaceStreams foreach { v => val s = v._2; s.println(libInterfaceFtr(v._1)); s.close() }
    nativeStreams foreach { v => val s = v._2; s.close() }
    
    // compile native code into .so
    libraries foreach { v => val lib = v._1; lib.compile(new File(nativeDir, "/" + lib.name + "." + lib.ext) toString, libDir.toString) }
    
    super.finalizeGenerator()    
  }
  
  //////////////////
  // generator state
  
  val libraries = HashMap[ExternalLibrary, Set[String]]() // from library to a set of method names
  val interfaceStreams = HashMap[ExternalLibrary, PrintWriter]() // from library to interface printstreams  
  val nativeStreams = HashMap[ExternalLibrary, PrintWriter]() // from library to native printstreams
    
  
  /////////////////
  // implementation
    
  def emitInterfaceAndMethod(lib: ExternalLibrary, funcName: String, funcSignature: String, nativeFunc: String) {    
    if (!libraries.contains(lib)){
      emitHeader(lib)
    }

    val methodSet = libraries.getOrElse(lib, Set[String]())
    if (!methodSet.contains(funcName)){
      emitMethod(lib, funcSignature)
      emitNativeWrapper(lib, nativeFunc)        
    }

    libraries.put(lib, methodSet + funcName)
  }
  
  // one per library
  def emitHeader(lib: ExternalLibrary) {
    assert(!interfaceStreams.contains(lib))
    assert(!nativeStreams.contains(lib))

    // interface file header
    val i = new PrintWriter(new File(headerDir, "/" + hdrName(lib) + "." + hdrExt))
    i.println(libInterfaceHdr(lib))
    interfaceStreams.put(lib, i)

    // native file header
    val n = new PrintWriter(new File(nativeDir, "/" + lib.name + "." + lib.ext))
    n.println(lib.header)
    nativeStreams.put(lib, n)    
  }
  
  // one per method into shared file <libName>.{hdrExt}
  def emitMethod(lib: ExternalLibrary, method: String) {
    val s = interfaceStreams.getOrElse(lib, throw new IllegalArgumentException("tried to emit external method header without an open interface file"))
    s.println(method)
    s.println("")
  }
  

  // one per method into shared file <libName>.{lib.ext}
  def emitNativeWrapper(lib: ExternalLibrary, nativeFunc: String) {
    val s = nativeStreams.getOrElse(lib, throw new IllegalArgumentException("tried to emit native wrapper without an open native file"))
    s.println(nativeFunc)
    s.println("")    
  }
}
