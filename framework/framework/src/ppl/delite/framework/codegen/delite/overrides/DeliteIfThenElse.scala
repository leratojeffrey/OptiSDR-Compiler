package ppl.delite.framework.codegen.delite.overrides

import scala.virtualization.lms.common._
import ppl.delite.framework.ops.DeliteOpsExp
import java.io.PrintWriter
import scala.virtualization.lms.internal.{GenericNestedCodegen,GenerationFailedException}
import scala.reflect.SourceContext

trait DeliteIfThenElseExp extends IfThenElseExp with BooleanOpsExp with EqualExpBridge with DeliteOpsExp {

  this: DeliteOpsExp =>

  // there is a lot of code duplication between DeliteIfThenElse and IfThenElse in lms -- do we really need a separate DeliteIfThenElse?

  case class DeliteIfThenElse[T:Manifest](cond: Exp[Boolean], thenp: Block[T], elsep: Block[T], flat: Boolean) extends DeliteOpCondition[T]{
    val m = manifest[T]
  }

  override def __ifThenElse[T:Manifest](cond: Rep[Boolean], thenp: => Rep[T], elsep: => Rep[T])(implicit ctx: SourceContext) = delite_ifThenElse(cond, thenp, elsep, false, true)

  // a 'flat' if is treated like any other statement in code motion, i.e. code will not be pushed explicitly into the branches
  def flatIf[T:Manifest](cond: Rep[Boolean])(thenp: => Rep[T])(elsep: => Rep[T]) = delite_ifThenElse(cond, thenp, elsep, true, true)

  def delite_ifThenElse[T:Manifest](cond: Rep[Boolean], thenp: => Rep[T], elsep: => Rep[T], flat: Boolean, controlFlag: Boolean)(implicit ctx: SourceContext): Rep[T] = cond match {
      // TODO: need to handle vars differently, this could be unsound  <--- don't understand ...
    case Const(true) => thenp
    case Const(false) => elsep
    case Def(BooleanNegate(a)) => delite_ifThenElse(a, elsep, thenp, flat, controlFlag)
    case Def(NotEqual(a,b)) => delite_ifThenElse(equals(a,b), elsep, thenp, flat, controlFlag)
    case _ =>
      val a = reifyEffectsHere[T](thenp, controlFlag)
      val b = reifyEffectsHere[T](elsep, controlFlag)
      val ae = summarizeEffects(a).withoutControl
      val be = summarizeEffects(b).withoutControl
      reflectEffectInternal(DeliteIfThenElse(cond,a,b,flat), ae orElse be)
  }

  override def mirrorDef[A:Manifest](e: Def[A], f: Transformer)(implicit pos: SourceContext): Def[A] = e match {
    case e@DeliteIfThenElse(c,a,b,h) => DeliteIfThenElse(f(c),f(a),f(b),h)(mtype(e.m))
    case _ => super.mirrorDef(e,f)
  }

  override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit ctx: SourceContext): Exp[A] = (e match {
    case Reflect(e@DeliteIfThenElse(c,a,b,h), u, es) =>
      if (f.hasContext)
        delite_ifThenElse(f(c),f.reflectBlock(a),f.reflectBlock(b),h,false)(mtype(e.m), ctx)
      else {
        reflectMirrored(Reflect(new { override val original = Some(f,e) } with DeliteIfThenElse(f(c),f(a),f(b),h)(mtype(e.m)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
      }
    case e@DeliteIfThenElse(c,a,b,h) =>
      if (f.hasContext)
        delite_ifThenElse(f(c),f.reflectBlock(a),f.reflectBlock(b),h,false)(mtype(e.m), ctx)
      else {
        reflectPure(DeliteIfThenElse(f(c),f(a),f(b),h)(mtype(e.m)))(mtype(manifest[A]), ctx) // FIXME: should apply pattern rewrites (ie call smart constructor)
      }
    case _ => super.mirror(e, f)
  }).asInstanceOf[Exp[A]] // why??


  override def syms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c, t, e, h) => syms(c):::syms(t):::syms(e)
    case _ => super.syms(e)
  }

  override def readSyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c, t, e, h) => readSyms(c):::readSyms(t):::readSyms(e)
    case _ => super.readSyms(e)
  }

  override def boundSyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c, t, e, h) => effectSyms(t):::effectSyms(e)
    case _ => super.boundSyms(e)
  }

  override def symsFreq(e: Any): List[(Sym[Any], Double)] = e match {
    case DeliteIfThenElse(c, t, e, true) => freqNormal(c):::freqNormal(t):::freqNormal(e)
    case DeliteIfThenElse(c, t, e, _) => freqNormal(c):::freqCold(t):::freqCold(e)
    case _ => super.symsFreq(e)
  }

  override def aliasSyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c,a,b,h) => syms(a):::syms(b)
    case _ => super.aliasSyms(e)
  }

  override def containSyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c,a,b,h) => Nil
    case _ => super.containSyms(e)
  }

  override def extractSyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c,a,b,h) => Nil
    case _ => super.extractSyms(e)
  }

  override def copySyms(e: Any): List[Sym[Any]] = e match {
    case DeliteIfThenElse(c,a,b,h) => Nil // could return a,b but implied by aliasSyms
    case _ => super.copySyms(e)
  }

}

trait DeliteBaseGenIfThenElse extends GenericNestedCodegen {
  val IR: DeliteIfThenElseExp
  import IR._

}

trait DeliteScalaGenIfThenElse extends ScalaGenEffect with ScalaGenBooleanOps with DeliteBaseGenIfThenElse {
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    /**
     * IfThenElse generates methods for each branch due to empirically discovered performance issues in the JVM
     * when generating long blocks of straight-line code in each branch.
     */
    case DeliteIfThenElse(c,a,b,h) =>
      //val save = deliteKernel
      //deliteKernel = false
      stream.println("val " + quote(sym) + " = {")
      (a.res,b.res) match {
        case (Const(()), Const(())) => stream.println("()")
        case (_, Const(())) => generateThenOnly(sym, c, a, !deliteKernel && !simpleCodegen)
        case (Const(()), _) => generateElseOnly(sym, c, b, !deliteKernel && !simpleCodegen)
        case _ => generateThenElse(sym, c, a, b, !deliteKernel && !simpleCodegen)
      }
      stream.println("}")
      //deliteKernel = save

    case _ => super.emitNode(sym, rhs)
  }

  def generateThenOnly(sym: Sym[Any], c: Exp[Any], thenb: Block[Any], wrap: Boolean) = wrap match {
    case true =>  wrapMethod(sym, thenb, "thenb")
                  stream.println("if (" + quote(c) + ") {")
                  stream.println(quote(sym) + "thenb()")
                  stream.println("}")

    case false => stream.println("if (" + quote(c) + ") {")
                  emitBlock(thenb)
                  stream.println("}")
  }

  def generateElseOnly(sym: Sym[Any], c: Exp[Any], elseb: Block[Any], wrap: Boolean) = wrap match {
    case true =>  wrapMethod(sym, elseb, "elseb")
                  stream.println("if (" + quote(c) + ") {}")
                  stream.println("else {")
                  stream.println(quote(sym) + "elseb()")
                  stream.println("}")

    case false => stream.println("if (" + quote(c) + ") {}")
                  stream.println("else {")
                  emitBlock(elseb)
                  stream.println("}")
  }

  def generateThenElse(sym: Sym[Any], c: Exp[Any], thenb: Block[Any], elseb: Block[Any], wrap: Boolean) = wrap match {
    case true =>  wrapMethod(sym, thenb, "thenb")
                  wrapMethod(sym, elseb, "elseb")
                  stream.println("if (" + quote(c) + ") {")
                  stream.println(quote(sym) + "thenb()")
                  stream.println("} else { ")
                  stream.println(quote(sym) + "elseb()")
                  stream.println("}")

    case false => stream.println("if (" + quote(c) + ") {")
                  emitBlock(thenb)
                  stream.println(quote(getBlockResult(thenb)))
                  stream.println("} else {")
                  emitBlock(elseb)
                  stream.println(quote(getBlockResult(elseb)))
                  stream.println("}")
  }

  def wrapMethod(sym: Sym[Any], block: Block[Any], postfix: String) = {
    stream.println("def " + quote(sym) + postfix + "(): " + remap(getBlockResult(block).tp) + " = {")
    emitBlock(block)
    stream.println(quote(getBlockResult(block)))
    stream.println("}")
  }

}


trait DeliteGPUGenIfThenElse extends GPUGenEffect with DeliteBaseGenIfThenElse {
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
    rhs match {
      case DeliteIfThenElse(c,a,b,h) =>
        remap(sym.tp) match {
          case "void" =>
            stream.println(addTab() + "if (" + quote(c) + ") {")
            tabWidth += 1
            emitBlock(a)
            tabWidth -= 1
            stream.println(addTab() + "} else {")
            tabWidth += 1
            emitBlock(b)
            tabWidth -= 1
            stream.println(addTab()+"}")
          case _ =>
            stream.println(addTab() + "%s %s;".format(remap(sym.tp),quote(sym)))
            stream.println(addTab() + "if (" + quote(c) + ") {")
            tabWidth += 1
            emitBlock(a)
            stream.println("%s = %s;".format(quote(sym),quote(getBlockResult(a))))
            tabWidth -= 1
            stream.println(addTab() + "} else {")
            tabWidth += 1
            emitBlock(b)
            stream.println("%s = %s;".format(quote(sym),quote(getBlockResult(b))))
            tabWidth -= 1
            stream.println(addTab()+"}")
          }
        case _ => super.emitNode(sym, rhs)
      }
  }
}

trait DeliteCudaGenIfThenElse extends CudaGenEffect with CudaGenBooleanOps with DeliteGPUGenIfThenElse
trait DeliteOpenCLGenIfThenElse extends OpenCLGenEffect with OpenCLGenBooleanOps with DeliteGPUGenIfThenElse

trait DeliteCGenIfThenElse extends CGenEffect with CGenBooleanOps with DeliteBaseGenIfThenElse {
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
      rhs match {
        case DeliteIfThenElse(c,a,b,h) =>
          remap(sym.tp) match {
            case "void" =>
              stream.println("if (" + quote(c) + ") {")
              emitBlock(a)
              stream.println("} else {")
              emitBlock(b)
              stream.println("}")
            case _ =>
              stream.println("%s %s;".format(remapWithRef(sym.tp),quote(sym)))
              stream.println("if (" + quote(c) + ") {")
              emitBlock(a)
              stream.println("%s = %s;".format(quote(sym),quote(getBlockResult(a))))
              stream.println("} else {")
              emitBlock(b)
              stream.println("%s = %s;".format(quote(sym),quote(getBlockResult(b))))
              stream.println("}")
          }
        case _ => super.emitNode(sym, rhs)
      }
    }
}
