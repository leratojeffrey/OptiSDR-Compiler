package ppl.dsl.optiml.vector

import ppl.dsl.optiml.{OptiMLExp, OptiML}
import ppl.delite.framework.DeliteApplication
import ppl.delite.framework.datastructures.DeliteArray
import scala.virtualization.lms.common.{EffectExp, BaseExp, Base, ScalaGenBase, CGenBase}
import scala.virtualization.lms.util.OverloadHack
import scala.reflect.SourceContext
import java.io.PrintWriter

trait IndexVectorOps extends Base with OverloadHack { this: OptiML =>
  
  object IndexVector {
    def apply(len: Rep[Int], isRow: Rep[Boolean])(implicit ctx: SourceContext) = indexvector_obj_new(len, isRow)
    def apply(xs: Interface[Vector[Int]])(implicit o: Overloaded1, ctx: SourceContext) = indexvector_obj_fromvec(xs)
    def apply(xs: Rep[Int]*)(implicit o: Overloaded2, ctx: SourceContext) = {
      val out = indexvector_obj_new(unit(0), unit(true))
      // interpreted (not lifted)
      xs.foreach { out <<= _ }
      out.unsafeImmutable // return immutable object
    }    
  }

  trait IndexVecOpsCls extends OptiMLVecOpsOverrides[Int] with InterfaceOps[IndexVector] {
    //implicit def toOps(x: Rep[VA]): IndexVecOpsCls
    //implicit def toIntf(x: Rep[VA]): Interface[IndexVector]        
    type V[X] = DenseVector[X] // conversion operations on IndexVectors will return a DenseVector
    type M[X] = DenseMatrix[X]
    type I[X] = DenseMatrix[X]
    def wrap(x: Rep[Self]): Interface[IndexVector]    
    def vecToOps[B:Manifest](x: Rep[DenseVector[B]]) = repToDenseVecOps(x)
    def vecToIntf[B:Manifest](x: Rep[DenseVector[B]]): Interface[Vector[B]] = denseVecToInterface(x)
    def vecBuilder[B:Manifest](implicit ctx: SourceContext): VectorBuilder[B,V[B]] = denseVectorBuilder[B]        
    def matToIntf[B:Manifest](x: Rep[DenseMatrix[B]]): Interface[Matrix[B]] = denseMatToInterface(x)
    def matBuilder[B:Manifest](implicit ctx: SourceContext): MatrixBuilder[B,I[B],M[B]] = denseMatrixBuilder[B]
    def mV[B:Manifest] = manifest[DenseVector[B]]
    def mM[B:Manifest] = manifest[DenseMatrix[B]]
    def mA = manifest[Int]
            
    def apply[A:Manifest](block: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[V[A]] = indexvector_construct(wrap(x), block)    
    //def *(y: Rep[Matrix[Int]])(implicit a: Arith[Int], o: Overloaded2, ctx: SourceContext) = throw new UnsupportedOperationException("tbd")
  }
    
  class IVInterface(override val ops: IndexVecOpsCls) extends VInterface[Int](ops) with Interface[IndexVector]  
  implicit def interfaceToIndexVecOps(intf: Interface[IndexVector]): InterfaceIndexVecOpsCls = new InterfaceIndexVecOpsCls(intf.asInstanceOf[IVInterface])
  
  class InterfaceIndexVecOpsCls(override val intf: IVInterface) extends InterfaceVecOpsCls[Int](intf) {
    def apply[A:Manifest](block: Rep[Int] => Rep[A])(implicit ctx: SourceContext) = intf.ops.apply(block)
  }
  
  // impl defs
  def indexvector_hashreduce[V:Manifest:Arith](x: Interface[IndexVector], keyFunc: Rep[Int] => Rep[Int], mapFunc: Rep[Int] => Rep[V], reduceFunc: (Rep[V],Rep[V]) => Rep[V]): Rep[DenseVector[V]]
  def indexvector_range(start: Rep[Int], end: Rep[Int])(implicit ctx: SourceContext): Rep[IndexVectorRange]
  def indexvector_obj_new(len: Rep[Int], isRow: Rep[Boolean])(implicit ctx: SourceContext): Rep[IndexVectorDense]
  def indexvector_obj_fromvec(xs: Interface[Vector[Int]])(implicit ctx: SourceContext): Rep[IndexVectorDense]

  // class defs
  def indexvector_construct[A:Manifest](x: Interface[IndexVector], block: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]]
}

trait IndexVectorOpsExp extends IndexVectorOps with EffectExp { this: OptiMLExp with IndexVectorImplOps =>

  ///////////////////////////////////////////////////
  // implemented via method on real data structure

  case class IndexVectorRangeNew(start: Exp[Int], end: Exp[Int]) extends Def[IndexVectorRange]
  
  case class IndexVectorDenseNew(len: Exp[Int], isRow: Exp[Boolean]) extends DeliteStruct[IndexVectorDense] {
    val elems = copyTransformedElems(collection.Seq("_data" -> var_new(DeliteArray[Int](len)).e, "_length" -> var_new(len).e, "_isRow" -> var_new(isRow).e))
  }

  ////////////////////////////////
  // implemented via delite ops

  case class IndexVectorObjectFromVec(xs: Interface[Vector[Int]]) extends DeliteOpSingleTask[IndexVectorDense](reifyEffectsHere(index_vector_obj_fromvec_impl(xs)))

  // Note: Construction from a discrete index vector set will currently return a contiguous (non-sparse) vector.
  // Is this what we want?
  case class IndexVectorConstruct[B:Manifest](intf: Interface[IndexVector], func: Exp[Int] => Exp[B])
    extends DeliteOpMap[Int,B,DenseVector[B]] {

    val in = intf.ops.elem.asInstanceOf[Exp[Vector[Int]]]
    val size = copyTransformedOrElse(_.size)(intf.length)    
    override def alloc = DenseVector[B](intf.length, intf.isRow)
    def m = manifest[B]
  }

  case class IndexVectorHash[V:Manifest:Arith](intf: Interface[IndexVector], keyFunc: Exp[Int] => Exp[Int], valFunc: Exp[Int] => Exp[V], reduceFunc: (Exp[V],Exp[V]) => Exp[V]) 
    extends DeliteOpMappedGroupByReduce[Int, Int, V, DeliteArray[V]] {
    
    val in = intf.ops.elem.asInstanceOf[Exp[Vector[Int]]]
    val size = copyTransformedOrElse(_.size)(intf.length)
    def zero = aV.empty
    def alloc(i: Exp[Int]) = DeliteArray(i)
    def mV = manifest[V]
    def aV = implicitly[Arith[V]]
  }
  
  // impl defs
  def indexvector_range(start: Exp[Int], end: Exp[Int])(implicit ctx: SourceContext) = reflectPure(IndexVectorRangeNew(start, end))
  def indexvector_obj_new(len: Exp[Int], isRow: Exp[Boolean])(implicit ctx: SourceContext) = reflectMutable(IndexVectorDenseNew(len, isRow))
  def indexvector_obj_fromvec(xs: Interface[Vector[Int]])(implicit ctx: SourceContext) = reflectPure(IndexVectorObjectFromVec(xs))
  def indexvector_hashreduce[V:Manifest:Arith](x: Interface[IndexVector], keyFunc: Exp[Int] => Exp[Int], mapFunc: Exp[Int] => Exp[V], reduceFunc: (Exp[V],Exp[V]) => Exp[V]) = {
    val data = reflectPure(IndexVectorHash(x, keyFunc, mapFunc, reduceFunc))
    DenseVectorNewImm(data, data.length, false)
  }

  // class defs
  def indexvector_construct[A:Manifest](x: Interface[IndexVector], block: Exp[Int] => Exp[A])(implicit ctx: SourceContext): Exp[DenseVector[A]] = {
    reflectPure(IndexVectorConstruct(x, block))
    // HACK -- better scheduling performance in our apps, forces some expensive dependencies to be hoisted
    // TR TODO: use effect summary of func
    //reflectEffect(IndexVectorConstruct(x, block))
  }

  //////////////
  // mirroring

  override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit ctx: SourceContext): Exp[A] = (e match {
    case e@IndexVectorObjectFromVec(x) => reflectPure(new { override val original = Some(f,e) } with IndexVectorObjectFromVec(f.intf(x)))(mtype(manifest[A]), implicitly[SourceContext])
    case e@IndexVectorConstruct(in,b) => reflectPure(new { override val original = Some(f,e) } with IndexVectorConstruct(f.intf(in),f(b))(e.m))(mtype(manifest[A]), implicitly[SourceContext])
    case IndexVectorRangeNew(start,end) => indexvector_range(f(start),f(end))
    case e@IndexVectorHash(x,kf,mf,rf) => reflectPure(new { override val original = Some(f,e) } with IndexVectorHash(f.intf(x),f(kf),f(mf),f(rf))(e.mV,e.aV))(mtype(manifest[A]), implicitly[SourceContext])
    case Reflect(e@IndexVectorObjectFromVec(x), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with IndexVectorObjectFromVec(f.intf(x)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@IndexVectorDenseNew(l,r), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with IndexVectorDenseNew(f(l),f(r)), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@IndexVectorConstruct(in,b), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with IndexVectorConstruct(f.intf(in),f(b))(e.m), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@IndexVectorHash(x,kf,mf,rf), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with IndexVectorHash(f.intf(x),f(kf),f(mf),f(rf))(e.mV,e.aV), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case _ => super.mirror(e, f)
  }).asInstanceOf[Exp[A]]
}

trait ScalaGenIndexVectorOps extends ScalaGenBase {
  val IR: IndexVectorOpsExp
}

trait CGenIndexVectorOps extends CGenBase {
  val IR: IndexVectorOpsExp
}
