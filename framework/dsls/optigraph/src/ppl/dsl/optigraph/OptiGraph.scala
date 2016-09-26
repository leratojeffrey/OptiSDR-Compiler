package ppl.dsl.optigraph

import java.io._
import scala.virtualization.lms.common._
import scala.virtualization.lms.internal.{Expressions,GenericFatCodegen, GenericCodegen}
import ppl.delite.framework.{Config, DeliteApplication}
import ppl.delite.framework.datastructures._
import ppl.delite.framework.codegen.Target
import ppl.delite.framework.codegen.scala.TargetScala
import ppl.delite.framework.codegen.delite.overrides.{DeliteScalaGenAllOverrides, DeliteAllOverridesExp}
import ppl.delite.framework.codegen.delite.DeliteCodeGenPkg
import ppl.delite.framework.transform.ForeachReduceTransformExp
import ppl.delite.framework.ops._
import ppl.delite.framework.{DeliteInteractive, DeliteInteractiveRunner, DeliteRestageOps, DeliteRestageOpsExp, DeliteRestageRunner}
import ppl.delite.framework.codegen.restage.{DeliteCodeGenRestage,LMSCodeGenRestage,TargetRestage}

//import ppl.dsl.optigraph.io._
import ppl.dsl.optigraph.ops._
import ppl.dsl.optigraph.datastruct.scala._

/**
 * OptiGraph application packages
 */
//trait OptiGraphApplicationRunner extends OptiGraphApplicationRunnerBase with OptiGraphExp
//trait OptiGraphApplicationRunnerBase extends OptiGraphApplication with DeliteApplication

trait OptiGraphApplicationRunner extends OptiGraphApplication with OptiGraphExp
trait OptiGraphApplication extends DeliteApplication with OptiGraph with OptiGraphLift

trait OptiGraphLibrary {
  this: OptiGraphApplication =>
}

/**
 * Portions of Scala imported into OptiGraph
 */
trait OptiGraphLift extends LiftVariables with LiftEquals with LiftString with LiftBoolean with LiftNumeric {
  this: OptiGraph =>
}

trait OptiGraphScalaOpsPkg extends Base
  with Equal with IfThenElse with Variables with While with Functions
  with ImplicitOps with OrderingOps with StringOps
  with BooleanOps with PrimitiveOps with MiscOps with TupleOps with StructOps with NumericOps
  with MathOps with CastingOps with ObjectOps with IOOps with HashMapOps
  with ArrayOps with ExceptionOps

trait OptiGraphScalaOpsPkgExp extends OptiGraphScalaOpsPkg with DSLOpsExp
  with EqualExp with IfThenElseExp with VariablesExp with WhileExp with FunctionsExp
  with ImplicitOpsExp with OrderingOpsExp with StringOpsExp with RangeOpsExp with IOOpsExp
  with ArrayOpsExp with BooleanOpsExp with PrimitiveOpsExp with MiscOpsExp with TupleOpsExp with StructExp with StructFatExpOptCommon
  with ListOpsExp with SeqOpsExp with MathOpsExp with CastingOpsExp with SetOpsExp with ObjectOpsExp
  with SynchronizedArrayBufferOpsExp with HashMapOpsExp with IterableOpsExp with ExceptionOpsExp
  with NumericOpsExp 

trait OptiGraphScalaCodeGenPkg extends ScalaGenDSLOps
  with ScalaGenEqual with ScalaGenIfThenElse with ScalaGenVariables with ScalaGenWhile with ScalaGenFunctions
  with ScalaGenImplicitOps with ScalaGenOrderingOps with ScalaGenStringOps with ScalaGenRangeOps with ScalaGenIOOps
  with ScalaGenArrayOps with ScalaGenBooleanOps with ScalaGenPrimitiveOps with ScalaGenMiscOps with ScalaGenTupleOps
  with ScalaGenListOps with ScalaGenSeqOps with ScalaGenMathOps with ScalaGenCastingOps with ScalaGenSetOps with ScalaGenObjectOps
  with ScalaGenSynchronizedArrayBufferOps with ScalaGenHashMapOps with ScalaGenIterableOps with ScalaGenExceptionOps
  with ScalaGenNumericOps   
  { val IR: OptiGraphScalaOpsPkgExp  }

/**
 * Operations available to the compiler only (cannot be used by applications)
 */
trait OptiGraphCompiler extends OptiGraph
  with DeliteArrayCompilerOps
  with DeliteArrayBufferCompilerOps
  with DeliteCollectionOps
  with GIterableCompilerOps
  with RangeOps with IOOps with SeqOps with SetOps
  with ListOps with HashMapOps with IterableOps
  with ExceptionOps
  // -- kernel implementations
  with LanguageImplOpsStandard
  with GraphImplOpsStandard
  with GIterableImplOpsStandard
  with NodeImplOpsStandard {

  this: OptiGraphApplication with OptiGraphExp =>
}

trait OptiGraph extends OptiGraphScalaOpsPkg with DeliteCollectionOps with DeliteArrayOps with StructOps with DeliteArrayBufferOps
  with LanguageOps
  with GraphOps with NodeOps with EdgeOps
  with NodePropertyOps with EdgePropertyOps
  with GIterableOps with GSetOps with GOrderOps with GSeqOps
  with ReduceableOps with DeferrableOps {

  this: OptiGraphApplication =>
}

trait OptiGraphLower extends OptiGraphApplication with DeliteRestageOps
trait OptiGraphLowerRunner[R] extends OptiGraphApplicationRunner with DeliteRestageRunner[R]

object OptiGraph_ {
  def apply[R](b: => R) = new Scope[OptiGraphLower, OptiGraphLowerRunner[R], R](b)
}

/**
 * OptiGraph IR
 */

trait OptiGraphExp extends OptiGraphCompiler with OptiGraphScalaOpsPkgExp with DeliteOpsExp with DeliteArrayFatExp with DeliteArrayBufferOpsExp with DeliteMapOpsExp with StructExp 
  with LanguageOpsExp
  with GraphOpsExp with NodeOpsExp with EdgeOpsExp
  with NodePropertyOpsExp with EdgePropertyOpsExp
  with ExceptionOps
  with GIterableOpsExp with GSetOpsExp with GOrderOpsExp with GSeqOpsExp
  with ReduceableOpsExp with DeferrableOpsExp
  with ForeachReduceTransformExp
  with DeliteRestageOpsExp with DeliteAllOverridesExp {

  this: DeliteApplication with OptiGraphApplication with OptiGraphExp =>

  def getCodeGenPkg(t: Target{val IR: OptiGraphExp.this.type}) : GenericFatCodegen{val IR: OptiGraphExp.this.type} = {
    t match {
      case _:TargetScala => new OptiGraphCodeGenScala{val IR: OptiGraphExp.this.type = OptiGraphExp.this}
      case _:TargetRestage => new OptiGraphCodeGenRestage{val IR: OptiGraphExp.this.type = OptiGraphExp.this}
      case _ => throw new RuntimeException("unsupported target")
    }
  }
}

/**
 * OptiGraph code generators
 */
trait OptiGraphCodeGenBase extends GenericFatCodegen {
  val IR: DeliteApplication with OptiGraphExp
  override def initialDefs = IR.deliteGenerator.availableDefs

  def dsmap(line: String) = line
  def parmap(line: String) = line
  val specialize = Set[String]()
  val specialize2 = Set[String]()
  val specialize3 = Set[String]()
  def genSpec(f: File, outPath: String) = {}
  def genSpec2(f: File, outPath: String) = {}
  def genSpec3(f: File, outPath: String) = {}

  def getFiles(d: File): Array[File] = {
    d.listFiles flatMap { f => if (f.isDirectory()) getFiles(f) else Array(f) }
  }

  override def remap[A](m: Manifest[A]): String = m.erasure.getSimpleName match {
    case "NodeProperty" => IR.structName(m)
    case "Graph" => IR.structName(m)
    case "GIterable" => IR.structName(m)
    case "Node" => "Int" //IR.structName(m)
    case _ => super.remap(m)
  }
  
  override def emitDataStructures(path: String) {
    val s = File.separator
    val dsDir = new File(Config.homeDir + s+"dsls"+s+"optigraph"+s+"src"+s+"ppl"+s+"dsl"+s+"optigraph"+s+"datastruct"+s + this.toString)
    if (!dsDir.exists) return
    val outDir = new File(path)
    outDir.mkdirs()

    val files = getFiles(dsDir)
    for (f <- files) {
      if (f.isDirectory){
        emitDataStructures(f.getPath())
      }
      else {
        if (specialize contains (f.getName.substring(0, f.getName.indexOf(".")))) {
          genSpec(f, path)
        }
        if (specialize2 contains (f.getName.substring(0, f.getName.indexOf(".")))) {
          genSpec2(f, path)
        }
        if (specialize3 contains (f.getName.substring(0, f.getName.indexOf(".")))) {
          genSpec3(f, path)
        }
        val outFile = path + s + f.getName
        val out = new BufferedWriter(new FileWriter(outFile))
        for (line <- scala.io.Source.fromFile(f).getLines) {
          var l = dsmap(line) //+ "\n"
          out.write(parmap(l) + "\n")
        }
        out.close()
      }
    }
  }
}

trait OptiGraphCodeGenRestage extends OptiGraphScalaCodeGenPkg with DeliteCodeGenRestage with LMSCodeGenRestage { 
  val IR: DeliteApplication with OptiGraphExp  
  import IR._

  // we shouldn't need this if we have a proper lowering stage (i.e. transformation)
  override def remap[A](m: Manifest[A]): String = m.erasure.getSimpleName match {
    // the next two cases would happen automatically (in DeliteCodeGenRestage) if NodeProperty, GIterable and Graph <: Record
    case "NodeProperty" | "GIterable"  => "DeliteCollection[" + remap(m.typeArguments(0)) + "]" 
    case "Graph" => "Record"
    case "Node" => "Int" //IR.structName(m)
    case _ => super.remap(m)
  }  
}

trait OptiGraphCodeGenScala extends OptiGraphCodeGenBase with OptiGraphScalaCodeGenPkg with ScalaGenDeliteOps
  with ScalaGenDeliteCollectionOps with ScalaGenDeliteStruct with ScalaGenDeliteArrayOps with ScalaGenLanguageOps with ScalaGenDeliteArrayBufferOps
  with ScalaGenReduceableOps with ScalaGenDeferrableOps
  with ScalaGenGraphOps with ScalaGenNodeOps with ScalaGenEdgeOps
  with ScalaGenExceptionOps
  with ScalaGenNodePropertyOps with ScalaGenEdgePropertyOps
  with ScalaGenGIterableOps with ScalaGenGSetOps with ScalaGenGOrderOps with ScalaGenGSeqOps
  with DeliteScalaGenAllOverrides {

  val IR: DeliteApplication with OptiGraphExp

  override val specialize = Set[String]("Reduceable", "Deferrable")
  override val specialize2 = Set[String]()
  override val specialize3 = Set[String]("GOrder", "GSet", "GSeq")

  override def genSpec(f: File, dsOut: String) {
    for (s <- List("Double","Int","Float","Long","Boolean")) {
      val outFile = dsOut + s + f.getName
      val out = new BufferedWriter(new FileWriter(outFile))
      for (line <- scala.io.Source.fromFile(f).getLines) {
        out.write(specmap(line, s) + "\n")
      }
      out.close()
    }
  }

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

  override def genSpec3(f: File, dsOut: String) {
    for (s <- List("Node","Edge")) {
      val outFile = dsOut + s + f.getName
      val out = new BufferedWriter(new FileWriter(outFile))
      for (line <- scala.io.Source.fromFile(f).getLines) {
        out.write(specmap(line, s) + "\n")
      }
      out.close()
    }
  }

  def specmap(line: String, t: String) : String = {
    var res = line.replaceAll("object ", "object " + t)
    //res = res.replaceAll("import ", "import " + t)
    res = res.replaceAll("@specialized T: ClassManifest", t)
    res = res.replaceAll("T:Manifest", t)
    res = res.replaceAll("\\bT\\b", t)
    parmap(res)
  }

  def specmap2(line: String, t1: String, t2: String) : String = {
    var res = line.replaceAll("object ", "object " + t1 + t2)
    res = res.replaceAll("import ", "import " + t1 + t2)
    res = res.replaceAll("@specialized T: ClassManifest", t1)
    res = res.replaceAll("@specialized L: ClassManifest", t2)
    res = res.replaceAll("T:Manifest", t1)
    res = res.replaceAll("L:Manifest", t2)
    res = res.replaceAll("\\bT\\b", t1)
    res = res.replaceAll("\\bL\\b", t2)
    parmap(res)
  }

  override def remap[A](m: Manifest[A]): String = {
    //val mE = manifest[EdgeProperty[]]
    //m match {
      //case `mE` => {
        //parmap(manifest[Property].toString)
      //}
      //case _ => parmap(super.remap(m))
    //}
    parmap(super.remap(m))
  }

  override def parmap(line: String): String = {
    //printf(line)
    var res = line

    for(tpe1 <- List("Int","Long","Double","Float","Boolean")) {
      for (s <- specialize) {
        res = res.replaceAll(s+"\\["+tpe1+"\\]", tpe1+s)
      }
      for(tpe2 <- List("Int","Long","Double","Float","Boolean")) {
        for (s <- specialize2) {
          // should probably parse and trim whitespace, this is fragile
          res = res.replaceAll(s+"\\["+tpe1+","+tpe2+"\\]", tpe1+tpe2+s)
          res = res.replaceAll(s+"\\["+tpe1+", "+tpe2+"\\]", tpe1+tpe2+s)
        }
      }
    }

    for(tpe1 <- List("Node","Edge")) {
      for (s <- specialize3) {
        res = res.replaceAll(s+"\\["+tpe1+"\\]", tpe1+s)
      }
    }
    for (s <- specialize3) {
      res = res.replaceAll(s+"\\[generated.scala.Node\\]", "Node"+s)
      res = res.replaceAll(s+"\\[generated.scala.Edge\\]", "Edge"+s)
      res = res.replaceAll(s+"\\[ppl.dsl.optigraph.Node\\]", "Node"+s)
      res = res.replaceAll(s+"\\[ppl.dsl.optigraph.Edge\\]", "Edge"+s)
    }

    dsmap(res)
  }

  override def dsmap(line: String) : String = {
    var res = line.replaceAll("ppl.dsl.optigraph.datastruct", "generated")
    res = res.replaceAll("ppl.delite.framework.datastruct", "generated")
    res = res.replaceAll("ppl.dsl.optigraph", "generated.scala")
    res
  }
}
