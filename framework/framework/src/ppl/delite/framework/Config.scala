package ppl.delite.framework

object Config {

  private def getProperty(prop: String, default: String) = {
    val p1 = System.getProperty(prop)
    val p2 = System.getProperty(prop.substring(1))
    if (p1 != null && p2 != null) {
      assert(p1 == p2, "ERROR: conflicting properties")
      p1
    }
    else if (p1 != null) p1 else if (p2 != null) p2 else default
  }

  //var degFilename = System.getProperty("delite.deg.filename", "")
  var degFilename = getProperty("delite.deg.filename", "out.deg")
  var restageFile = getProperty("delite.restage.filename", "restage-scopes.scala")
  var opfusionEnabled = getProperty("delite.enable.fusion", "true") != "false"
  var soaEnabled = getProperty("delite.enable.soa", "true") != "false"
  var generateCUDA = getProperty("delite.generate.cuda", "false") != "false"
  var generateCpp = getProperty("delite.generate.cpp", "false") != "false"
  var generateOpenCL = getProperty("delite.generate.opencl", "false") != "false"
  var generateSerializable = getProperty("delite.generate.serializable", "false") != "false"
  var homeDir = getProperty("delite.home.dir", sys.env.getOrElse("DELITE_HOME",System.getProperty("user.dir")))
  var buildDir = getProperty("delite.build.dir", "generated")
  var useBlas = getProperty("delite.extern.blas", "false") != "false"  
  var debug = getProperty("delite.debug","false") != "false"
  var cacheSyms = getProperty("delite.cache.syms","false") != "false"
  var collectStencil = System.getProperty("deliszt.stencil.enabled", "false") == "true"
  var printGlobals = System.getProperty("delite.print_globals.enabled", "false") == "true"
  val optimize = getProperty("delite.optimize", "0").toInt
  val enableGPUTransform = getProperty("delite.enable.gputransform","false") != "false"
  val enableGPUObjReduce = getProperty("delite.enable.gpu.objreduce","true") != "false"
  val enableGPUMultiDim = getProperty("delite.enable.gpu.multidim","false") != "false"

  //Print generationFailedException info
  val dumpException: Boolean = getProperty("delite.dump.exception", "false") != "false"
  var enableProfiler = System.getProperty("delite.enable.profiler", "false") != "false"
}
