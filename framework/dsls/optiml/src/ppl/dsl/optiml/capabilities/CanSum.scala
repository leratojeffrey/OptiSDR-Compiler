package ppl.dsl.optiml.capabilities

import scala.virtualization.lms.common.{Variables, Base}
import ppl.dsl.optiml.{OptiMLExp, OptiML}
import scala.reflect.SourceContext

// trait CanSumInternal[Rep[X],R,A] {
//   def +=(acc: Rep[R], y: Rep[A]): Rep[R] // should be unit! (but we have an issue using the same type class for primitives...)
//   def mutable(lhs: Rep[A]): Rep[R]         
// }

trait CanSumOps extends Variables {
  this: OptiML =>

  //type CanSum[R,A] = CanSumInternal[Rep,R,A]
  
  trait CanSum[R,A] {
    def +=(acc: Rep[R], y: Rep[A])(implicit ctx: SourceContext): Rep[R] // should be unit! (but we have an issue using the same type class for primitives...)
    def mutable(lhs: Rep[A])(implicit ctx: SourceContext): Rep[R]      
  }

  // scalac: erroneous or inaccessible type. wtf?
  implicit def canSumView[A:Manifest:Arith] = new CanSum[DenseVector[A],DenseVectorView[A]] {
    def +=(acc: Rep[DenseVector[A]], y: Rep[DenseVectorView[A]])(implicit ctx: SourceContext) = /*acc.+=(denseViewToInterface[A](y)) //*/ acc += y   
    def mutable(lhs: Rep[DenseVectorView[A]])(implicit ctx: SourceContext) = /*denseViewToInterface(lhs).mutable //*/ lhs.mutable
  }

  implicit def canSumArith[A:Manifest:Arith:Cloneable] = new CanSum[A,A] {
    def +=(acc: Rep[A], y: Rep[A])(implicit ctx: SourceContext) = acc += y
    def mutable(lhs: Rep[A])(implicit ctx: SourceContext) = lhs.mutable
  } 
  
  /*
  implicit object CanSumDenseView extends CanSum[DenseVector[Double],DenseVectorView[Double]] {
    def +=(acc: Rep[DenseVector[Double]], y: Rep[DenseVectorView[Double]]) = acc += y    
    def mutable(lhs: Rep[DenseVectorView[Double]]) = lhs.mutable
  }
      
  implicit object CanSumDense extends CanSum[DenseVector[Double],DenseVector[Double]] {
    def +=(acc: Rep[DenseVector[Double]], y: Rep[DenseVector[Double]]) = acc += y
    def mutable(lhs: Rep[DenseVector[Double]]) = lhs.mutable
  }
  
  implicit object CanSumDouble extends CanPlusEquals[Double,Double] {
    def +=(acc: Rep[Double], y: Rep[Double]) = acc + y
    def mutable(lhs: Rep[Double]) = lhs
  }
  */  
}

