package ppl.dsl.optiml

import java.io._
import scala.reflect.SourceContext
import scala.virtualization.lms.common._
import scala.virtualization.lms.internal.{Expressions, GenericFatCodegen, GenericCodegen}
import ppl.delite.framework.{Config, DeliteApplication, DeliteInteractive, DeliteInteractiveRunner, DeliteRestageOps, DeliteRestageOpsExp, DeliteRestageRunner}
import ppl.delite.framework.codegen.Target
import ppl.delite.framework.codegen.scala.TargetScala
import ppl.delite.framework.codegen.cuda.TargetCuda
import ppl.delite.framework.codegen.cpp.TargetCpp
import ppl.delite.framework.codegen.opencl.TargetOpenCL
import ppl.delite.framework.codegen.restage.{DeliteCodeGenRestage,LMSCodeGenRestage,TargetRestage}
import ppl.delite.framework.codegen.delite.overrides.{DeliteCudaGenAllOverrides, DeliteOpenCLGenAllOverrides, DeliteCGenAllOverrides, DeliteScalaGenAllOverrides, DeliteAllOverridesExp}
import ppl.delite.framework.ops._
import ppl.delite.framework.datastructures._

import ppl.dsl.optila.{OptiLAApplication}
import ppl.dsl.optila.{OptiLAScalaOpsPkg, OptiLAScalaOpsPkgExp, OptiLA, OptiLAExp, OptiLACompiler, OptiLALift, OptiLAUtilities}
import ppl.dsl.optila.{OptiLAScalaCodeGenPkg, OptiLACudaCodeGenPkg, OptiLAOpenCLCodeGenPkg, OptiLACCodeGenPkg, OptiLACodeGenBase, OptiLACodeGenScala, OptiLACodeGenCuda, OptiLACodeGenOpenCL, OptiLACodeGenC}
import ppl.dsl.optila.capabilities.ScalaGenArithOps

import ppl.dsl.optiml.io._
import ppl.dsl.optiml.vector._
import ppl.dsl.optiml.matrix._
import ppl.dsl.optiml.graph._
import ppl.dsl.optiml.stream._
import ppl.dsl.optiml.library.cluster._
import ppl.dsl.optiml.library.regression._
import ppl.dsl.optiml.application._
import ppl.dsl.optiml.capabilities._
import ppl.dsl.optiml.transform._

/**
 * Microbenchmark experiments: OptiMLApplicationRunners with optimizations disabled
 */

trait OptiMLNoCSE extends Expressions {
  override def findDefinition[T](d: Def[T]) = None
}


/**
 * These separate OptiML applications from the Exp world.
 */

trait OptiMLApplicationRunner extends OptiMLApplicationRunnerBase with OptiMLExpOpt

// ex. object GDARunner extends OptiMLApplicationRunner with GDA
trait OptiMLApplicationRunnerBase extends OptiMLApplication with DeliteApplication 
// ex. trait GDA extends OptiMLApplication
trait OptiMLApplication extends OptiLAApplication with OptiML with OptiMLLift with OptiMLLibrary {
  var args: Rep[Array[String]]
  def main(): Unit  
}

trait OptiMLLibrary extends OptiMLKmeans with OptiMLLinReg {
  this: OptiMLApplication with OptiMLLift =>
}


trait OptiMLInteractive extends OptiMLApplication with DeliteInteractive

trait OptiMLInteractiveRunner[R] extends OptiMLApplicationRunner with DeliteInteractiveRunner[R]

// executes scope immediately
object OptiML {
  def apply[R](b: => R) = new Scope[OptiMLInteractive, OptiMLInteractiveRunner[R], R](b)
}

// stages scope and generates re-stageable code
trait OptiMLLower extends OptiMLApplication with DeliteRestageOps
trait OptiMLLowerRunner[R] extends OptiMLApplicationRunner with DeliteRestageRunner[R]

object OptiML_ {
  def apply[R](b: => R) = new Scope[OptiMLLower, OptiMLLowerRunner[R], R](b)
}

/**
 * These are the portions of Scala imported into OptiML's scope.
 */
trait OptiMLLift extends OptiLALift {
  this: OptiML =>
}

trait OptiMLScalaOpsPkg extends OptiLAScalaOpsPkg

trait OptiMLScalaOpsPkgExp extends OptiLAScalaOpsPkgExp with OptiMLScalaOpsPkg

trait OptiMLScalaCodeGenPkg extends OptiLAScalaCodeGenPkg 
  { val IR: OptiMLScalaOpsPkgExp  }

trait OptiMLCudaCodeGenPkg extends OptiLACudaCodeGenPkg
  { val IR: OptiMLScalaOpsPkgExp  }

trait OptiMLOpenCLCodeGenPkg extends OptiLAOpenCLCodeGenPkg
  { val IR: OptiMLScalaOpsPkgExp  }

trait OptiMLCCodeGenPkg extends OptiLACCodeGenPkg
  { val IR: OptiMLScalaOpsPkgExp  }

/**
 * This is the trait that every OptiML application must extend.
 */
trait OptiML extends OptiLA with OptiMLTypes with OptiMLScalaOpsPkg 
  with LanguageOps with ApplicationOps 
  with MLInputReaderOps with MLOutputWriterOps
  with CanSumOps
  with VectorOps with OptiMLDenseVectorOps with OptiMLDenseVectorViewOps with OptiMLSparseVectorOps with OptiMLSparseVectorViewOps with OptiMLRangeVectorOps
  with MatrixOps with OptiMLDenseMatrixOps with OptiMLSparseMatrixOps 
  with IndexVectorOps with IndexVectorDenseOps with IndexVectorRangeOps with IndexVector2Ops with IndexVectorTriangularOps
  with StreamOps with StreamRowOps
  with GraphOps with EdgeOps with VertexOps with VSetOps
  with TrainingSetOps with ImageOps with GrayscaleImageOps with GrayscaleImageOpsExtension {

  this: OptiMLApplication =>
}

// these ops are only available to the compiler (they are restricted from application use)
trait OptiMLCompiler extends OptiLACompiler with OptiML with OptiMLUtilities with GraphCompilerOps with DeliteCollectionOps 
  with LanguageImplOpsStandard with VectorImplOpsStandard with IndexVectorImplOpsStandard with MatrixImplOpsStandard
  with GrayscaleImageImplOpsStandard with MLInputReaderImplOpsStandard with MLOutputWriterImplOpsStandard with StreamImplOpsStandard
  with GraphImplOpsStandard with EdgeImplOpsStandard with VertexImplOpsStandard with VerticesImplOpsStandard {

  this: OptiMLApplication with OptiMLExp =>
}


/**
 * These are the corresponding IR nodes for OptiML.
 */
trait OptiMLExp extends OptiLAExp with OptiMLCompiler with OptiMLScalaOpsPkgExp
  with LanguageOpsExpOpt with ApplicationOpsExp
  with MLInputReaderOpsExp with MLOutputWriterOpsExp
  with VectorOpsExpOpt with MatrixOpsExpOpt with DenseMatrixOpsExpOpt 
  with IndexVectorOpsExp with IndexVectorDenseOpsExpOpt with IndexVectorRangeOpsExp with IndexVector2OpsExp with IndexVectorTriangularOpsExp
  with StreamOpsExpOpt with StreamRowOpsExpOpt
  with TrainingSetOpsExp with ImageOpsExp with GrayscaleImageOpsExp
  with GraphOpsExp with EdgeOpsExp with VertexOpsExp with VSetOpsExp
  with MultiloopTransformExp
  with DeliteRestageOpsExp
  with DeliteAllOverridesExp {

  // this: OptiMLApplicationRunner => why doesn't this work?
  this: DeliteApplication with OptiMLApplication with OptiMLExp => // can't be OptiMLApplication right now because code generators depend on stuff inside DeliteApplication (via IR)

  override def getCodeGenPkg(t: Target{val IR: OptiMLExp.this.type}) : GenericFatCodegen{val IR: OptiMLExp.this.type} = {
    t match {
      case _:TargetScala => new OptiMLCodeGenScala{val IR: OptiMLExp.this.type = OptiMLExp.this}
      case _:TargetCuda => new OptiMLCodeGenCuda{val IR: OptiMLExp.this.type = OptiMLExp.this}
      case _:TargetOpenCL => new OptiMLCodeGenOpenCL{val IR: OptiMLExp.this.type = OptiMLExp.this}
      case _:TargetCpp => new OptiMLCodeGenC{val IR: OptiMLExp.this.type = OptiMLExp.this}
      case _:TargetRestage => new OptiMLCodeGenRestage{val IR: OptiMLExp.this.type = OptiMLExp.this}
      case _ => err("optiml does not support this target")
    }
  }
}

// add rewritings
trait OptiMLExpOpt extends OptiMLExp
  with VectorOpsExpOpt with MatrixOpsExpOpt with StreamOpsExpOpt with StreamRowOpsExpOpt {
    
  this: DeliteApplication with OptiMLApplication with OptiMLExp =>
}

trait OptiMLUtilities extends OptiLAUtilities {
  override def err(s: String)(implicit ctx: SourceContext) = {
    println("[optiml error]: " + s)
    println("  at " + (ctx.fileName.split("/").last + ":" + ctx.line).mkString("//").mkString(";"))
    exit(1)
  }
  override def warn(s: String) = println("[optiml warning]: " + s)
}


/**
 * OptiML code generators
 */
trait OptiMLCodeGenBase extends OptiLACodeGenBase {

  val IR: DeliteApplication with OptiMLExp
  override def initialDefs = IR.deliteGenerator.availableDefs

  val mlspecialize = Set[String]()
  val mlspecialize2 = Set[String]()  
  def genSpec2(f: File, outPath: String) = {}

    
  override def emitDataStructures(path: String) {
    super.emitDataStructures(path) // get optila data structures
    
    val s = File.separator
    val dsRoot = Config.homeDir + s+"dsls"+s+"optiml"+s+"src"+s+"ppl"+s+"dsl"+s+"optiml"+s+"datastruct"+s + this.toString

    val dsDir = new File(dsRoot)
    if (!dsDir.exists) return
    val outDir = new File(path)
    outDir.mkdirs()

    val files = getFiles(dsDir)
    for (f <- files) {
      if (f.isDirectory){
        emitDataStructures(f.getPath())
      }
      else {
        if (mlspecialize contains (f.getName.substring(0, f.getName.indexOf(".")))) {
          genSpec(f, path)
        }
        if (mlspecialize2 contains (f.getName.substring(0, f.getName.indexOf(".")))) {
          genSpec2(f, path)
        }
        val outFile = path + s + f.getName
        val out = new BufferedWriter(new FileWriter(outFile))
        for (line <- scala.io.Source.fromFile(f).getLines) {
          out.write(dsmap(line) + "\n")
        }
        out.close()
      }
    }
  }
}

// strategy is to inherit all of the base Scala generators and override what we need
trait OptiMLCodeGenRestage extends OptiMLScalaCodeGenPkg with ScalaGenArithOps with DeliteCodeGenRestage with LMSCodeGenRestage { 
  val IR: DeliteApplication with OptiMLExp
  import IR._
  
  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case ArithAbs(x) => emitValDef(sym, "Math.abs(" + quote(x) + ")")
    case ArithExp(x) => emitValDef(sym, "Math.exp(" + quote(x) + ")")
    case _ => super.emitNode(sym, rhs)
  }
}

trait OptiMLCodeGenScala extends OptiLACodeGenScala with OptiMLCodeGenBase with OptiMLScalaCodeGenPkg
  with ScalaGenApplicationOps with ScalaGenLanguageOps 
  with ScalaGenVectorOps with ScalaGenMatrixOps with ScalaGenIndexVectorOps with ScalaGenIndexVectorDenseOps with ScalaGenIndexVector2Ops 
  with ScalaGenStreamOps with ScalaGenStreamRowOps
  with ScalaGenGraphOps with ScalaGenEdgeOps with ScalaGenVertexOps with ScalaGenVSetOps
  with ScalaGenTrainingSetOps with ScalaGenDeliteCollectionOps
  with ScalaGenImageOps with ScalaGenGrayscaleImageOps
  with DeliteScalaGenAllOverrides { //with ScalaGenMLInputReaderOps {
  
  val IR: DeliteApplication with OptiMLExp

  override val mlspecialize = Set(/*"LabelsImpl", "Image",*/ "UnsupervisedTrainingSet", "Stream", "StreamRow")
  override val mlspecialize2 = Set("SupervisedTrainingSet")

  override def genSpec2(f: File, dsOut: String) {
    for (s1 <- List("Double","Int","Float","Long","Boolean")) {
      for (s2 <- List("Double","Int","Float","Long","Boolean")) {
        val outFile = dsOut + s1 + s2 + f.getName
        val out = new BufferedWriter(new FileWriter(outFile))
        for (line <- scala.io.Source.fromFile(f).getLines) {
          out.write(specmap2(line, s1, s2) + "\n")
        }
        out.close()
    }
    }
  }

  override def remap(s: String) = parmap(s)
  
  override def specmap(line: String, t: String) : String = {
    var res = line.replaceAll("import ppl.dsl.optila.datastruct.scala._", "") // ends up in the same package in generated code    
    super.specmap(res, t)
  }
    
  def specmap2(line: String, t1: String, t2: String) : String = {
    var res = line.replaceAll("object ", "object " + t1 + t2)
    res = res.replaceAll("import ppl.dsl.optila.datastruct.scala._", "") 
    res = res.replaceAll("import ", "import " + t1 + t2)
    res = res.replaceAll("@specialized T: Manifest", t1)
    res = res.replaceAll("@specialized L: Manifest", t2)
    res = res.replaceAll("T:Manifest", t1)
    res = res.replaceAll("L:Manifest", t2)
    res = res.replaceAll("\\bT\\b", t1)
    res = res.replaceAll("\\bL\\b", t2)
    dsmap(res)
  }
  
  override def parmap(line: String): String = {
    var res = line
    for(tpe1 <- List("Int","Long","Double","Float","Boolean")) {
      for (s <- mlspecialize) {
        res = res.replaceAll(s+"\\["+tpe1+"\\]", tpe1+s)
      }
      for(tpe2 <- List("Int","Long","Double","Float","Boolean")) {
        for (s <- mlspecialize2) {
          // should probably parse and trim whitespace, this is fragile
          res = res.replaceAll(s+"\\["+tpe1+","+tpe2+"\\]", tpe1+tpe2+s)
          res = res.replaceAll(s+"\\["+tpe1+", "+tpe2+"\\]", tpe1+tpe2+s)
        }
      }
    }
    super.parmap(res)
  }

  override def dsmap(line: String) : String = {
    var res = line.replaceAll("ppl.dsl.optiml.datastruct", "generated")
    res = res.replaceAll("import ppl.dsl.optila.datastruct.scala._", "")     
    res = res.replaceAll("ppl.delite.framework.datastruct", "generated")
    res = res.replaceAll(".*\\$", "generated.scala.")
    super.dsmap(res) // calls parmap
  }
}

trait OptiMLCodeGenCuda extends OptiLACodeGenCuda with OptiMLCodeGenBase with OptiMLCudaCodeGenPkg 
  with CudaGenVectorOps with CudaGenMatrixOps with CudaGenTrainingSetOps
  with DeliteCudaGenAllOverrides with OptiMLCppHostTransfer with OptiMLCudaDeviceTransfer 
{
  val IR: DeliteApplication with OptiMLExp
  import IR._
}

trait OptiMLCodeGenOpenCL extends OptiLACodeGenOpenCL with OptiMLCodeGenBase with OptiMLOpenCLCodeGenPkg
  with OpenCLGenVectorOps with OpenCLGenMatrixOps with OpenCLGenTrainingSetOps
  with DeliteOpenCLGenAllOverrides
{
  val IR: DeliteApplication with OptiMLExp
  import IR._
}

trait OptiMLCodeGenC extends OptiLACodeGenC with OptiMLCodeGenBase with OptiMLCCodeGenPkg with CGenIndexVectorOps 
  with CGenVectorOps with CGenMatrixOps with CGenTrainingSetOps with DeliteCGenAllOverrides 
  with OptiMLCppHostTransfer
{
  val IR: DeliteApplication with OptiMLExp
  import IR._
}