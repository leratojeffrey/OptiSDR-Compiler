package ppl.delite.framework.extern.codegen.cpp

import scala.virtualization.lms.common._
import scala.virtualization.lms.internal._
import collection.mutable.{ListBuffer}
import collection.mutable.HashMap
import java.io.{FileWriter, BufferedWriter, File, PrintWriter}

import ppl.delite.framework.{Config, DeliteApplication}
import ppl.delite.framework.extern.lib._
import ppl.delite.framework.extern.codegen.GenericGenExternal
import ppl.delite.framework.ops._
import ppl.delite.framework.codegen.delite._

trait CGenExternalBase extends GenericGenExternal with CGenFat with CGenDeliteOps {
  val IR: DeliteOpsExp
  import IR._

  val hdrExt = "h"
  lazy val globalInterfaceStream = new PrintWriter(new File(headerDir + "/" + "library" + "." + hdrExt)) // runtime conventions with GPUExecutableGenerator

  override def finalizeGenerator() {
    globalInterfaceStream.close()
    super.finalizeGenerator()
  }

  /////////////////
  // implementation
  
  override def emitHeader(lib: ExternalLibrary) {
    // global interface file for all cuda libraries
    globalInterfaceStream.println("#include \"" + hdrName(lib) + "." + hdrExt + "\"")
    super.emitHeader(lib)
  }

  def emitMethodCall(sym: Sym[Any], e: DeliteOpExternal[_], lib: ExternalLibrary, args: List[String]) = {
    emitBlock(e.allocVal) 
    emitValDef(sym, quote(getBlockResult(e.allocVal)))          
    stream.println(e.funcName + "(" + (args mkString ",") + ");")     
  }
  
  def emitInterfaceAndMethod(lib: ExternalLibrary, funcName: String, args: List[String], global: String, body: String) = {
    val funcSignature = "void " + funcName + "(" + (args mkString ",") + ")"
    super.emitInterfaceAndMethod(lib, funcName,
      funcSignature + ";",
      global + "\n" + funcSignature + body
    )
  }

  override def getDataStructureHeaders(): String = {
    val out = new StringBuilder
    out.append("#include \"library.h\"\n")
    super.getDataStructureHeaders() + out.toString
  }
     
}
