package ppl.dsl.optila.capabilities

import scala.virtualization.lms.util.OverloadHack
import scala.virtualization.lms.common._
import java.io.PrintWriter
import scala.virtualization.lms.internal.{CLikeCodegen}
import ppl.dsl.optila.{OptiLAExp, OptiLA}
import scala.reflect.SourceContext

/*
 * Arith definitions for OptiLA supported types.
 *
 * author:  Arvind Sujeeth (asujeeth@stanford.edu)
 * created: Dec 2, 2010
 *
 * Pervasive Parallelism Laboratory (PPL)
 * Stanford University
 *
 */

/* Type class for basic math, but less restrictive than Numeric. */

trait ArithInternal[Rep[X],T] {
  def +=(a: Rep[T], b: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def +(a: Rep[T], b: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def -(a: Rep[T], b: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def *(a: Rep[T], b: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def /(a: Rep[T], b: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def abs(a: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def exp(a: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def empty(implicit ctx: SourceContext): Rep[T]
  def zero(a: Rep[T])(implicit ctx: SourceContext): Rep[T]
  /*
  def unary_-(a: Rep[T]) : Rep[T]
  */
}

trait ArithOps extends Variables with OverloadHack {
  this: OptiLA =>
  
  type Arith[X] = ArithInternal[Rep,X]
  
  // hack: need to pass explicit type class parameters during mirroring, similar to mtype
  def atype[A,B](a: Arith[A]): Arith[B] = a.asInstanceOf[Arith[B]]

  /**
   * Interface, enables using Ariths with operator notation. Note that the inclusion of these
   * causes the NumericOps and FractionalOps implicit conversions to be ambiguous, so OptiLA
   * programs cannot include them.
   */
  implicit def arithToArithOps[T:Arith:Manifest](n: T) = new ArithOpsCls(unit(n))
  implicit def repArithToArithOps[T:Arith:Manifest](n: Rep[T]) = new ArithOpsCls(n)
  implicit def varArithToArithOps[T:Arith:Manifest](n: Var[T]) = new ArithOpsCls(readVar(n))

  class ArithOpsCls[T](lhs: Rep[T])(implicit mT: Manifest[T], arith: Arith[T]){
    def +=(rhs: Rep[T])(implicit ctx: SourceContext): Rep[T] = arith.+=(lhs,rhs)
    def +=[B](rhs: B)(implicit c: B => Rep[T], ctx: SourceContext): Rep[T] = arith.+=(lhs,c(rhs))    
    def +(rhs: Rep[T])(implicit ctx: SourceContext): Rep[T] = arith.+(lhs,rhs)
    def -(rhs: Rep[T])(implicit ctx: SourceContext): Rep[T] = arith.-(lhs,rhs)
    def *(rhs: Rep[T])(implicit ctx: SourceContext): Rep[T] = arith.*(lhs,rhs)
    def /(rhs: Rep[T])(implicit ctx: SourceContext): Rep[T] = arith./(lhs,rhs)

    def abs(implicit ctx: SourceContext): Rep[T] = arith.abs(lhs)
    def exp(implicit ctx: SourceContext): Rep[T] = arith.exp(lhs)
    def empty(implicit ctx: SourceContext): Rep[T] = arith.empty
    def zero(implicit ctx: SourceContext): Rep[T] = arith.zero(lhs)
  }
         
         
  /**
   * Vector
   */

  implicit def denseVectorArith[T:Arith:Manifest]: Arith[DenseVector[T]] = new Arith[DenseVector[T]] {
    // these are used in sum; dynamic checks are required due to conditionals
    // def +=(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]]) = if (!b.IsInstanceOf[ZeroVector[T]]) a += b else a
    // def +(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]]) = if (a.IsInstanceOf[ZeroVector[T]]) b
    //                                               else if (b.IsInstanceOf[ZeroVector[T]]) a
    //                                               else a+b

    def +=(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).+=(b) 
    def +(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).+(b)
    def -(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).-(b)
    def *(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).*(b)
    def /(a: Rep[DenseVector[T]], b: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a)./(b)
    def abs(a: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).abs
    def exp(a: Rep[DenseVector[T]])(implicit ctx: SourceContext) = repToDenseVecOps(a).exp
    
    /**
     * zero for Vector[T] is a little tricky. It is used in nested Vector/Matrix operations, e.g.
     * a reduction on a Vector[Vector[T]]. For a variable dimension nested vector, the empty vector is the only
     * right answer. For a fixed dimension nested Vector, such as [[1,2,3],[4,5,6]], you'd ideally want the 
     * k-dimension zero vector, e.g. [0,0,0] in this example. However, this is the dimension
     * of v(0).dim, not v.dim, and cannot be statically enforced with our types, and furthermore would need to
     * correctly handled multiple levels of nesting. This situation is resolved by the DeliteOpReduce contract to
     * never use zero except in the case of the empty collection.
     *  
     * For non-nested cases, i.e. conditional maps or reduces, we want the zero-valued k-dimensional value,
     * but we don't always know k before running the function... (see sumIf in kmeans)
     */
    def empty(implicit ctx: SourceContext) = EmptyVector[T]
    def zero(a: Rep[DenseVector[T]])(implicit ctx: SourceContext) = ZeroVector[T](a.length)
  }


  /**
   * Matrix
   */

  implicit def denseMatrixArith[T:Arith:Manifest]: Arith[DenseMatrix[T]] = new Arith[DenseMatrix[T]] {
    def +=(a: Rep[DenseMatrix[T]], b: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).+=(b)
    def +(a: Rep[DenseMatrix[T]], b: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).+(b)
    def -(a: Rep[DenseMatrix[T]], b: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).-(b)
    def *(a: Rep[DenseMatrix[T]], b: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).*(b)
    def /(a: Rep[DenseMatrix[T]], b: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a)./(b)
    def abs(a: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).abs
    def exp(a: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = repToDenseMatOps(a).exp
    def empty(implicit ctx: SourceContext) = DenseMatrix[T](unit(0),unit(0)) // EmptyDenseMatrix? 
    def zero(a: Rep[DenseMatrix[T]])(implicit ctx: SourceContext) = DenseMatrix[T](a.numRows, a.numCols)
    /*
    def unary_-(a: Rep[DenseMatrix[T]]) = -a
    */
  }


  /**
   *  Tuple
   */
  
  implicit def tuple2Arith[A:Manifest:Arith,B:Manifest:Arith] : Arith[Tuple2[A,B]] =
    new Arith[Tuple2[A,B]] {
      def +=(a: Rep[Tuple2[A,B]], b: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1 += b._1, a._2 += b._2)

      def +(a: Rep[Tuple2[A,B]], b: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1+b._1, a._2+b._2)

      def -(a: Rep[Tuple2[A,B]], b: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1-b._1, a._2-b._2)

      def *(a: Rep[Tuple2[A,B]], b: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1*b._1, a._2*b._2)

      def /(a: Rep[Tuple2[A,B]], b: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1/b._1, a._2/b._2)

      def abs(a: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1.abs, a._2.abs)

      def exp(a: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1.exp, a._2.exp)
      
      def empty(implicit ctx: SourceContext) =
        Tuple2(implicitly[Arith[A]].empty, implicitly[Arith[B]].empty)
      
      def zero(a: Rep[Tuple2[A,B]])(implicit ctx: SourceContext) =
        Tuple2(a._1.zero, a._2.zero)
    }
  
  implicit def tuple3Arith[A:Manifest:Arith,B:Manifest:Arith,C:Manifest:Arith] : Arith[Tuple3[A,B,C]] =
    new Arith[Tuple3[A,B,C]] {
      def +=(a: Rep[Tuple3[A,B,C]], b: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1 += b._1, a._2 += b._2, a._3 += b._3)

      def +(a: Rep[Tuple3[A,B,C]], b: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1+b._1, a._2+b._2, a._3+b._3)

      def -(a: Rep[Tuple3[A,B,C]], b: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1-b._1, a._2-b._2, a._3-b._3)

      def *(a: Rep[Tuple3[A,B,C]], b: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1*b._1, a._2*b._2, a._3*b._3)

      def /(a: Rep[Tuple3[A,B,C]], b: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1/b._1, a._2/b._2, a._3/b._3)

      def abs(a: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1.abs, a._2.abs, a._3.abs)

      def exp(a: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1.exp, a._2.exp, a._3.exp)
      
      def empty(implicit ctx: SourceContext) =
        Tuple3(implicitly[Arith[A]].empty, implicitly[Arith[B]].empty, implicitly[Arith[C]].empty)

      def zero(a: Rep[Tuple3[A,B,C]])(implicit ctx: SourceContext) =
        Tuple3(a._1.zero, a._2.zero, a._3.zero)
    }

  //implicit def tuple4Arith[A,B,C,D](implicit rA: A => Rep[A], rB: B => Rep[B], rC: C => Rep[C], rD: D => Rep[D], opsA: Arith[A], mA: Manifest[A], opsB: Arith[B], mB: Manifest[B],
  implicit def tuple4Arith[A:Manifest:Arith,B:Manifest:Arith,C:Manifest:Arith,D:Manifest:Arith] : Arith[Tuple4[A,B,C,D]] =
    new Arith[Tuple4[A,B,C,D]] {
      def +=(a: Rep[Tuple4[A,B,C,D]], b: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1 += b._1, a._2 += b._2, a._3 += b._3, a._4 += b._4)

      def +(a: Rep[Tuple4[A,B,C,D]], b: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1+b._1, a._2+b._2, a._3+b._3, a._4+b._4)

      def -(a: Rep[Tuple4[A,B,C,D]], b: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1-b._1, a._2-b._2, a._3-b._3, a._4-b._4)

      def *(a: Rep[Tuple4[A,B,C,D]], b: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1*b._1, a._2*b._2, a._3*b._3, a._4*b._4)

      def /(a: Rep[Tuple4[A,B,C,D]], b: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1/b._1, a._2/b._2, a._3/b._3, a._4/b._4)

      def abs(a: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1.abs, a._2.abs, a._3.abs, a._4.abs)

      def exp(a: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1.exp, a._2.exp, a._3.exp, a._4.exp)
      
      def empty(implicit ctx: SourceContext) =
        Tuple4(implicitly[Arith[A]].empty, implicitly[Arith[B]].empty, implicitly[Arith[C]].empty, implicitly[Arith[D]].empty)

      def zero(a: Rep[Tuple4[A,B,C,D]])(implicit ctx: SourceContext) =
        Tuple4(a._1.zero, a._2.zero, a._3.zero, a._4.zero)    
    }

  /**
   * Tuple-scalar math
   * 
   * This unfortunately large number of declarations is used to resolve overload ambiguities with previous arith declarations.
   */     
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded13): Rep[Tuple2[A,A]] = ((a._1+b,a._2))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple2[A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded14): Rep[Tuple2[A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded13): Rep[Tuple2[A,A]] = ((a._1-b,a._2-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded13): Rep[Tuple2[A,A]] = ((a._1*b,a._2*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple2[A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded14): Rep[Tuple2[A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded13): Rep[Tuple2[A,A]] = ((a._1/b,a._2/b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded15): Rep[Tuple2[A,A]] = ((a._1+b,a._2))
   def infix_+[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple2[A,A]])(implicit conv: B => Rep[A], o: Overloaded16): Rep[Tuple2[A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded15): Rep[Tuple2[A,A]] = ((a._1-b,a._2-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded15): Rep[Tuple2[A,A]] = ((a._1*b,a._2*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple2[A,A]])(implicit conv: B => Rep[A], o: Overloaded16): Rep[Tuple2[A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple2[A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded15): Rep[Tuple2[A,A]] = ((a._1/b,a._2/b))   
   
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded17): Rep[Tuple3[A,A,A]] = ((a._1+b,a._2+b,a._3+b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple3[A,A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded18): Rep[Tuple3[A,A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded17): Rep[Tuple3[A,A,A]] = ((a._1-b,a._2-b,a._3-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded17): Rep[Tuple3[A,A,A]] = ((a._1*b,a._2*b,a._3*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple3[A,A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded18): Rep[Tuple3[A,A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded17): Rep[Tuple3[A,A,A]] = ((a._1/b,a._2/b,a._3/b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded19): Rep[Tuple3[A,A,A]] = ((a._1+b,a._2+b,a._3+b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple3[A,A,A]])(implicit conv: B => Rep[A], o: Overloaded20): Rep[Tuple3[A,A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded19): Rep[Tuple3[A,A,A]] = ((a._1-b,a._2-b,a._3-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded19): Rep[Tuple3[A,A,A]] = ((a._1*b,a._2*b,a._3*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple3[A,A,A]])(implicit conv: B => Rep[A], o: Overloaded20): Rep[Tuple3[A,A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple3[A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded19): Rep[Tuple3[A,A,A]] = ((a._1/b,a._2/b,a._3/b))
      
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded21): Rep[Tuple4[A,A,A,A]] = ((a._1+b,a._2+b,a._3+b,a._4+b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple4[A,A,A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded22): Rep[Tuple4[A,A,A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded21): Rep[Tuple4[A,A,A,A]] = ((a._1-b,a._2-b,a._3-b,a._4-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded21): Rep[Tuple4[A,A,A,A]] = ((a._1*b,a._2*b,a._3*b,a._4*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[B], b: Rep[Tuple4[A,A,A,A]])(implicit conv: Rep[B] => Rep[A], o: Overloaded22): Rep[Tuple4[A,A,A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: Rep[B])(implicit conv: Rep[B] => Rep[A], o: Overloaded21): Rep[Tuple4[A,A,A,A]] = ((a._1/b,a._2/b,a._3/b,a._4/b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded23): Rep[Tuple4[A,A,A,A]] = ((a._1+b,a._2+b,a._3+b,a._4+b))
   def infix_+[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple4[A,A,A,A]])(implicit conv: B => Rep[A], o: Overloaded24): Rep[Tuple4[A,A,A,A]] = infix_+(b,a)
   def infix_-[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded23): Rep[Tuple4[A,A,A,A]] = ((a._1-b,a._2-b,a._3-b,a._4-b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded23): Rep[Tuple4[A,A,A,A]] = ((a._1*b,a._2*b,a._3*b,a._4*b))
   def infix_*[A:Manifest:Arith,B:Manifest](a: B, b: Rep[Tuple4[A,A,A,A]])(implicit conv: B => Rep[A], o: Overloaded24): Rep[Tuple4[A,A,A,A]] = infix_*(b,a)
   def infix_/[A:Manifest:Arith,B:Manifest](a: Rep[Tuple4[A,A,A,A]], b: B)(implicit conv: B => Rep[A], o: Overloaded23): Rep[Tuple4[A,A,A,A]] = ((a._1/b,a._2/b,a._3/b,a._4/b))   

  /**
   * Primitives
   *
   * unfortunately, to use ArithOps, we have to redefine all of the operations we want to
   * to support from NumericOps and FractionalOps, since their implicits are ambiguous with ours.
   */

  implicit val doubleArith : Arith[Double] = new Arith[Double] {
    def +=(a: Rep[Double], b: Rep[Double])(implicit ctx: SourceContext) = arith_plus(a,b)
    def +(a: Rep[Double], b: Rep[Double])(implicit ctx: SourceContext) = arith_plus(a,b)
    def -(a: Rep[Double], b: Rep[Double])(implicit ctx: SourceContext) = arith_minus(a,b)
    def *(a: Rep[Double], b: Rep[Double])(implicit ctx: SourceContext) = arith_times(a,b)
    def /(a: Rep[Double], b: Rep[Double])(implicit ctx: SourceContext) = arith_fractional_divide(a,b)
    def abs(a: Rep[Double])(implicit ctx: SourceContext) = arith_abs(a)
    def exp(a: Rep[Double])(implicit ctx: SourceContext) = arith_exp(a)
    def empty(implicit ctx: SourceContext) = unit(0.0)
    def zero(a: Rep[Double])(implicit ctx: SourceContext) = empty
    //def unary_-(a: Rep[Double]) = -a
  }

  implicit val floatArith : Arith[Float] = new Arith[Float] {
    def +=(a: Rep[Float], b: Rep[Float])(implicit ctx: SourceContext) = arith_plus(a,b)
    def +(a: Rep[Float], b: Rep[Float])(implicit ctx: SourceContext) = arith_plus(a,b)
    def -(a: Rep[Float], b: Rep[Float])(implicit ctx: SourceContext) = arith_minus(a,b)
    def *(a: Rep[Float], b: Rep[Float])(implicit ctx: SourceContext) = arith_times(a,b)
    def /(a: Rep[Float], b: Rep[Float])(implicit ctx: SourceContext) = arith_fractional_divide(a,b)
    def abs(a: Rep[Float])(implicit ctx: SourceContext) = arith_abs(a)
    def exp(a: Rep[Float])(implicit ctx: SourceContext) = arith_exp(a).AsInstanceOf[Float]
    def empty(implicit ctx: SourceContext) = unit(0f)
    def zero(a: Rep[Float])(implicit ctx: SourceContext) = empty
    //def unary_-(a: Rep[Float]) = -a
  }

  implicit val intArith : Arith[Int] = new Arith[Int] {
    def +=(a: Rep[Int], b: Rep[Int])(implicit ctx: SourceContext) = arith_plus(a,b)
    def +(a: Rep[Int], b: Rep[Int])(implicit ctx: SourceContext) = arith_plus(a,b)
    def -(a: Rep[Int], b: Rep[Int])(implicit ctx: SourceContext) = arith_minus(a,b)
    def *(a: Rep[Int], b: Rep[Int])(implicit ctx: SourceContext) = arith_times(a,b)
    def /(a: Rep[Int], b: Rep[Int])(implicit ctx: SourceContext) = int_divide(a,b)
    def abs(a: Rep[Int])(implicit ctx: SourceContext) = arith_abs(a)
    def exp(a: Rep[Int])(implicit ctx: SourceContext) = arith_exp(a).AsInstanceOf[Int]
    def empty(implicit ctx: SourceContext) = unit(0)
    def zero(a: Rep[Int])(implicit ctx: SourceContext) = empty
    //def unary_-(a: Rep[Int]) = -a
  }
  
  implicit val longArith : Arith[Long] = new Arith[Long] {
    def +=(a: Rep[Long], b: Rep[Long])(implicit ctx: SourceContext) = arith_plus(a,b)
    def +(a: Rep[Long], b: Rep[Long])(implicit ctx: SourceContext) = arith_plus(a,b)
    def -(a: Rep[Long], b: Rep[Long])(implicit ctx: SourceContext) = arith_minus(a,b)
    def *(a: Rep[Long], b: Rep[Long])(implicit ctx: SourceContext) = arith_times(a,b)
    def /(a: Rep[Long], b: Rep[Long])(implicit ctx: SourceContext) = throw new UnsupportedOperationException("tbd")
    def abs(a: Rep[Long])(implicit ctx: SourceContext) = arith_abs(a)
    def exp(a: Rep[Long])(implicit ctx: SourceContext) = arith_exp(a).AsInstanceOf[Long]
    def empty(implicit ctx: SourceContext) = unit(0L)
    def zero(a: Rep[Long])(implicit ctx: SourceContext) = empty
    //def unary_-(a: Rep[Long]) = -a
  }
  
  
  def arith_plus[T:Manifest:Numeric](lhs: Rep[T], rhs: Rep[T])(implicit ctx: SourceContext): Rep[T]
  def arith_minus[T:Manifest:Numeric](lhs: Rep[T], rhs: Rep[T])(implicit ctx: SourceContext): Rep[T]
  def arith_times[T:Manifest:Numeric](lhs: Rep[T], rhs: Rep[T])(implicit ctx: SourceContext): Rep[T]
  def arith_fractional_divide[T:Manifest:Fractional](lhs: Rep[T], rhs: Rep[T])(implicit ctx: SourceContext) : Rep[T]
  def arith_abs[T:Manifest:Numeric](lhs: Rep[T])(implicit ctx: SourceContext): Rep[T]
  def arith_exp[T:Manifest:Numeric](lhs: Rep[T])(implicit ctx: SourceContext): Rep[Double]
}

trait ArithOpsExp extends ArithOps with VariablesExp {
  this: OptiLAExp =>

  abstract class NumericDef[A:Manifest:Numeric,R:Manifest] extends DefWithManifest[A,R] {
    val n = implicitly[Numeric[A]]
  }
  case class ArithPlus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T]) extends NumericDef[T,T]
  case class ArithPlusEquals[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T]) extends NumericDef[T,Unit]
  case class ArithMinus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T]) extends NumericDef[T,T]
  case class ArithTimes[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T]) extends NumericDef[T,T]
  case class ArithFractionalDivide[T:Manifest:Fractional](lhs: Exp[T], rhs: Exp[T]) extends DefWithManifest[T,T]{
    val f = implicitly[Fractional[T]]
  }
  case class ArithAbs[T:Manifest:Numeric](lhs: Exp[T]) extends NumericDef[T,T]
  case class ArithExp[T:Manifest:Numeric](lhs: Exp[T]) extends NumericDef[T,Double]

  def arith_plus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext): Exp[T] = reflectPure(ArithPlus(lhs, rhs))
  def arith_minus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext): Exp[T] = reflectPure(ArithMinus(lhs, rhs))
  def arith_times[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext): Exp[T] = reflectPure(ArithTimes(lhs, rhs))
  def arith_fractional_divide[T:Manifest:Fractional](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext): Exp[T] = reflectPure(ArithFractionalDivide(lhs, rhs))
  def arith_abs[T:Manifest:Numeric](lhs: Exp[T])(implicit ctx: SourceContext) = reflectPure(ArithAbs(lhs))
  def arith_exp[T:Manifest:Numeric](lhs: Exp[T])(implicit ctx: SourceContext) = reflectPure(ArithExp(lhs))

  override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit ctx: SourceContext): Exp[A] = (e match {
    case e@ArithPlus(lhs,rhs) => reflectPure(ArithPlus(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]))(mtype(manifest[A]),implicitly[SourceContext])
    case e@ArithMinus(lhs,rhs) => reflectPure(ArithMinus(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]))(mtype(manifest[A]),implicitly[SourceContext])
    case e@ArithTimes(lhs,rhs) => reflectPure(ArithTimes(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]))(mtype(manifest[A]),implicitly[SourceContext])
    case e@ArithFractionalDivide(lhs,rhs) => reflectPure(ArithFractionalDivide(f(lhs),f(rhs))(mtype(e.mA),e.f.asInstanceOf[Fractional[A]]))(mtype(manifest[A]),implicitly[SourceContext])
    case e@ArithAbs(lhs) => reflectPure(ArithAbs(f(lhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]))(mtype(manifest[A]),implicitly[SourceContext])
    case e@ArithExp(lhs) => reflectPure(ArithExp(f(lhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[Any]]))(mtype(manifest[A]),implicitly[SourceContext])

    case Reflect(e@ArithPlus(lhs,rhs), u, es) => reflectMirrored(Reflect(ArithPlus(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@ArithMinus(lhs,rhs), u, es) => reflectMirrored(Reflect(ArithMinus(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@ArithTimes(lhs,rhs), u, es) => reflectMirrored(Reflect(ArithTimes(f(lhs),f(rhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@ArithFractionalDivide(lhs,rhs), u, es) => reflectMirrored(Reflect(ArithFractionalDivide(f(lhs),f(rhs))(mtype(e.mA),e.f.asInstanceOf[Fractional[A]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@ArithAbs(lhs), u, es) => reflectMirrored(Reflect(ArithAbs(f(lhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[A]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@ArithExp(lhs), u, es) => reflectMirrored(Reflect(ArithExp(f(lhs))(mtype(e.mA),e.n.asInstanceOf[Numeric[Any]]), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case _ => super.mirror(e,f)
  }).asInstanceOf[Exp[A]] 
  
}

trait ArithOpsExpOpt extends ArithOpsExp {
  this: OptiLAExp =>
  
  def unbox[T:Manifest](n: java.lang.Number): T = {
    val mD = manifest[Double]
    val mF = manifest[Float]
    val mI = manifest[Int]
    val mL = manifest[Long]
    manifest[T] match {
      case `mD` => n.doubleValue().asInstanceOf[T]
      case `mF` => n.floatValue().asInstanceOf[T]
      case `mI` => n.intValue().asInstanceOf[T]
      case `mL` => n.longValue().asInstanceOf[T]
    }
  }

  override def arith_abs[T:Manifest:Numeric](lhs: Exp[T])(implicit ctx: SourceContext): Exp[T] = lhs match {
    case Const(x) => 
      // weird java class cast exception inside numeric while unboxing java.lang.Integer              
      val a = if (x.isInstanceOf[java.lang.Number]) unbox(x.asInstanceOf[java.lang.Number]) else x
      unit(implicitly[Numeric[T]].abs(a))
    case _ => super.arith_abs(lhs)
  }
  override def arith_plus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext) : Exp[T] = (lhs,rhs) match {
    case (Const(x), Const(y)) => 
      val a = if (x.isInstanceOf[java.lang.Number]) unbox(x.asInstanceOf[java.lang.Number]) else x
      val b = if (y.isInstanceOf[java.lang.Number]) unbox(y.asInstanceOf[java.lang.Number]) else y    
      unit(implicitly[Numeric[T]].plus(a,b))
    case (Const(0 | 0.0 | 0.0f | -0.0 | -0.0f), y) => y
    case (y, Const(0 | 0.0 | 0.0f | -0.0 | -0.0f)) => y
    case _ => super.arith_plus(lhs, rhs)
  }
  
  override def arith_minus[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext) : Exp[T] = (lhs,rhs) match {
    case (Const(x), Const(y)) => 
      val a = if (x.isInstanceOf[java.lang.Number]) unbox(x.asInstanceOf[java.lang.Number]) else x
      val b = if (y.isInstanceOf[java.lang.Number]) unbox(y.asInstanceOf[java.lang.Number]) else y    
      unit(implicitly[Numeric[T]].minus(a,b))
    // case (Const(0 | 0.0 | 0.0f | -0.0 | -0.0f), y) => unit(-1.asInstanceOf[T])*y
    case (y, Const(0 | 0.0 | 0.0f | -0.0 | -0.0f)) => y
    case _ => super.arith_minus(lhs, rhs)
  }
  
  override def arith_times[T:Manifest:Numeric](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext) : Exp[T] = (lhs,rhs) match {
    case (Const(x), Const(y)) => 
      val a = if (x.isInstanceOf[java.lang.Number]) unbox(x.asInstanceOf[java.lang.Number]) else x
      val b = if (y.isInstanceOf[java.lang.Number]) unbox(y.asInstanceOf[java.lang.Number]) else y
      unit(implicitly[Numeric[T]].times(a,b))
    case (Const(0 | 0.0 | 0.0f | -0.0 | -0.0f), _) => lhs
    case (_, Const(0 | 0.0 | 0.0f | -0.0 | -0.0f)) => rhs
//    case (Const(1 | 1.0 | 1.0f), y) => y //TODO: careful about type promotion!
//    case (y, Const(1 | 1.0 | 1.0f)) => y
    case _ => super.arith_times(lhs, rhs)
  }
  
  override def arith_fractional_divide[T:Manifest:Fractional](lhs: Exp[T], rhs: Exp[T])(implicit ctx: SourceContext) : Exp[T] = (lhs,rhs) match {
    case (Const(0 | 0.0 | 0.0f | -0.0 | -0.0f), _) => lhs  // TODO: shouldn't match on 0 / 0 ?
    case _ => super.arith_fractional_divide(lhs, rhs)
  }  
}



trait ScalaGenArithOps extends ScalaGenBase {
  val IR: ArithOpsExp
  import IR._
  
  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = rhs match {
    case ArithPlus(a,b) => emitValDef(sym, quote(a) + " + " + quote(b))
    case ArithMinus(a,b) => emitValDef(sym, quote(a) + " - " + quote(b))
    case ArithTimes(a,b) => emitValDef(sym, quote(a) + " * " + quote(b))
    case ArithFractionalDivide(a,b) => emitValDef(sym, quote(a) + " / " + quote(b))
    case ArithAbs(x) => emitValDef(sym, "java.lang.Math.abs(" + quote(x) + ")")
    //case a@ArithAbs(x) => a.m.asInstanceOf[Manifest[_]] match {
    //  case Manifest.Double => emitValDef(sym, "java.lang.Double.longBitsToDouble((java.lang.Double.doubleToRawLongBits(" + quote(x) + ")<<1)>>>1)")
    //  case _ => emitValDef(sym, "Math.abs(" + quote(x) + ")")
    //}
    case ArithExp(a) => emitValDef(sym, "java.lang.Math.exp(" + quote(a) + ")")
    case _ => super.emitNode(sym, rhs)
  }
}

trait CLikeGenArithOps extends CLikeCodegen {
  val IR: ArithOpsExp
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
      rhs match {
        case ArithPlus(a,b) => emitValDef(sym, quote(a) + " + " + quote(b))
        case ArithMinus(a,b) => emitValDef(sym, quote(a) + " - " + quote(b))
        case ArithTimes(a,b) => emitValDef(sym, quote(a) + " * " + quote(b))
        case ArithFractionalDivide(a,b) => emitValDef(sym, quote(a) + " / " + quote(b))
        case ArithAbs(x) if(remap(x.tp)=="float") => emitValDef(sym, "fabsf(" + quote(x) + ")")
        case ArithAbs(x) if(remap(x.tp)=="double") => emitValDef(sym, "fabs(" + quote(x) + ")")
        case ArithAbs(x) => emitValDef(sym, "abs(" + quote(x) + ")")
        case ArithExp(a) if(remap(a.tp)=="float") => emitValDef(sym, "expf(" + quote(a) + ")")
        case ArithExp(a) if(remap(a.tp)=="double") => emitValDef(sym, "exp(" + quote(a) + ")")
        case ArithExp(a) => emitValDef(sym, "exp(" + quote(a) + ")")
        case _ => super.emitNode(sym, rhs)
      }
    }
}

trait CudaGenArithOps extends CudaGenBase with CLikeGenArithOps
trait OpenCLGenArithOps extends OpenCLGenBase with CLikeGenArithOps
trait CGenArithOps extends CGenBase with CLikeGenArithOps

