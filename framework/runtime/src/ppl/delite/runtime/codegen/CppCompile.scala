package ppl.delite.runtime.codegen

import xml.XML
import ppl.delite.runtime.Config
import ppl.delite.runtime.graph.targets.Targets
import tools.nsc.io._

object CppCompile extends CCompile {

  def target = Targets.Cpp
  override def ext = "cpp"

  protected def configFile = "CPP.xml"
  protected def compileFlags = if (Config.cppMemMgr == "refcnt") Array("-w", "-O3", "-fPIC", "-std=c++0x")
                               else Array("-w", "-O3", "-fPIC")
  protected def linkFlags = Array("-shared", "-fPIC")
  protected def outputSwitch = "-o"
  
  private val dsFiles = Directory(Path(sourceCacheHome + "datastructures")).files.toList
  override protected def auxSourceList = dsFiles.filter(_.extension == ext).map(_.toAbsolute.toString) :+ (sourceCacheHome + "kernels" + sep + target + "helperFuncs." + ext) :+ (staticResources + "DeliteCpp." + ext) :+ (staticResources + "DeliteCppProfiler." + ext)
}
