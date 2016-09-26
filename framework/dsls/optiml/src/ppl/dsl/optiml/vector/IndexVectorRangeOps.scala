package ppl.dsl.optiml.vector

import java.io.PrintWriter
import scala.reflect.SourceContext
import scala.virtualization.lms.common.{EffectExp, BaseExp, Base, ScalaGenBase}
import scala.virtualization.lms.util.OverloadHack
import ppl.delite.framework.DeliteApplication
import ppl.delite.framework.ops.{DeliteCollectionOpsExp}
import ppl.delite.framework.ops.DeliteCollection
import ppl.dsl.optiml._

trait IndexVectorRangeOps extends Base with OverloadHack { this: OptiML =>

  implicit def repToIndexVecRangeOps(x: Rep[IndexVectorRange]) = new IndexVecRangeOpsCls(x)
  implicit def varToIndexVecRangeOps(x: Var[IndexVectorRange]) = new IndexVecRangeOpsCls(readVar(x))
  implicit def indexVecRangeToInterface(lhs: Rep[IndexVectorRange]) = new IVInterface(new IndexVecRangeOpsCls(lhs))
  
  // hack to allow index range vector builders to be found without explicit type parameters
  //implicit def indexRangeVecBuilderGeneric[A <% Int]: VectorBuilder[A,IndexVectorRange] = indexRangeVecBuilder.asInstanceOf[VectorBuilder[A,IndexVectorRange]]
  // manifest[A] match {
  //     case Manifest.Int => indexRangeVecBuilder
  //   }  
  // implicit def indexRangeVecBuilder = new VectorBuilder[Int,IndexVectorRange] {
  //   def alloc(length: Rep[Int], isRow: Rep[Boolean]) = (unit(0)::length)
  //   def toIntf(x: Rep[IndexVectorRange]): Interface[IndexVector] = indexVecRangeToInterface(x)
  // }  
  
  class IndexVecRangeOpsCls(val elem: Rep[IndexVectorRange]) extends IndexVecOpsCls {
    type Self = IndexVectorRange
    type VA = IndexVectorDense
    def wrap(x: Rep[IndexVectorRange]) = indexVecRangeToInterface(x)
    def vaToOps(x: Rep[VA]) = repToIndexVecDenseOps(x)
    def vaToIntf(x: Rep[VA]) = indexVecDenseToInterface(x)
    def vaBuilder(implicit ctx: SourceContext) = indexVecDenseBuilder
    def mVA = manifest[VA]
    
    // VectorOps
    def length(implicit ctx: SourceContext) = indexvectorrange_length(x)
    def isRow(implicit ctx: SourceContext) = unit(true)
    def apply(n: Rep[Int])(implicit ctx: SourceContext) = indexvectorrange_apply(x,n)
    def sort(implicit o: Ordering[Int], ctx: SourceContext) = x.Clone        
    def t(implicit ctx: SourceContext) = IndexVector(unit(0), !elem.isRow) << elem
    
    def mt()(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")    
    def update(n: Rep[Int], y: Rep[Int])(implicit ctx: SourceContext): Rep[Unit] = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def copyFrom(pos: Rep[Int], y: Rep[IndexVectorDense])(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def insert(pos: Rep[Int], y: Rep[Int])(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def insertAll(pos: Rep[Int], y: Rep[IndexVectorDense])(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def removeAll(pos: Rep[Int], len: Rep[Int])(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def trim()(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")
    def clear()(implicit ctx: SourceContext) = throw new UnsupportedOperationException("RangeVectors cannot be updated")        
  } 
  
  def indexvectorrange_length(x: Rep[IndexVectorRange])(implicit ctx: SourceContext): Rep[Int]
  def indexvectorrange_apply(x: Rep[IndexVectorRange], n: Rep[Int])(implicit ctx: SourceContext): Rep[Int]
  // def indexvectorrange_times_matrix(x: Rep[IndexVectorRange], y: Rep[Matrix[Int]]): Rep[DenseVector[Int]]
  // def indexvectorrange_flatmap[B:Manifest](x: Rep[IndexVectorRange], f: Rep[A] => Rep[DenseVector[B]]): Rep[DenseVector[B]]
}

trait IndexVectorRangeOpsExp extends IndexVectorRangeOps with DeliteCollectionOpsExp { this: OptiMLExp =>
    
  def indexvectorrange_length(x: Rep[IndexVectorRange])(implicit ctx: SourceContext) = x match {
    case Def(IndexVectorRangeNew(Const(0),e)) => e
    case Def(v@Reflect(IndexVectorRangeNew(Const(0),e), u, es)) => e
    case Def(IndexVectorRangeNew(s,e)) => e - s
    case Def(v@Reflect(IndexVectorRangeNew(s,e), u, es)) /*if context.contains(v)*/ => e - s
  }
  
  def indexvectorrange_apply(x: Rep[IndexVectorRange], n: Rep[Int])(implicit ctx: SourceContext) = x match {
    case Def(IndexVectorRangeNew(Const(0),e)) => n
    case Def(v@Reflect(IndexVectorRangeNew(Const(0),e), u, es)) => n
    case Def(IndexVectorRangeNew(s,e)) => s + n
    case Def(v@Reflect(IndexVectorRangeNew(s,e), u, es)) /*if context.contains(v)*/ => s + n
  }  
    
  /////////////////////
  // delite collection
  
  def isIndexRange[A](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = x.tp.erasure == classOf[IndexVectorRange]
  def asIndexRange[A](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = x.asInstanceOf[Exp[IndexVectorRange]]
    
  override def dc_size[A:Manifest](x: Exp[DeliteCollection[A]])(implicit ctx: SourceContext) = { 
    if (isIndexRange(x)) asIndexRange(x).length
    else super.dc_size(x)
  }
  
  override def dc_apply[A:Manifest](x: Exp[DeliteCollection[A]], n: Exp[Int])(implicit ctx: SourceContext) = {
    if (isIndexRange(x)) (asIndexRange(x).apply(n)).asInstanceOf[Exp[A]]
    else super.dc_apply(x,n)    
  }
  
  override def dc_update[A:Manifest](x: Exp[DeliteCollection[A]], n: Exp[Int], y: Exp[A])(implicit ctx: SourceContext) = {
    if (isIndexRange(x)) asIndexRange(x).update(n,y.asInstanceOf[Exp[Int]])
    else super.dc_update(x,n,y)        
  }
  
}
  
