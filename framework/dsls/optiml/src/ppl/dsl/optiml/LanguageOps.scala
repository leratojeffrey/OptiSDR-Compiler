package ppl.dsl.optiml

import ppl.delite.framework.ops.{DeliteOpsExp, DeliteCollection}
import java.io.PrintWriter
import reflect.{Manifest, SourceContext}
import scala.virtualization.lms.internal.GenericFatCodegen
import scala.virtualization.lms.common._

/* Machinery provided by OptiML itself (language features and control structures).
 *
 * author: Arvind Sujeeth (asujeeth@stanford.edu)
 * created: Nov 29, 2010
 *
 * Pervasive Parallelism Laboratory (PPL)
 * Stanford University
 *
 */

// subclassing LanguageOps prioritizes the implicits in our LanguageOps over OptiLAs for OptiML programs
trait LanguageOps extends ppl.dsl.optila.LanguageOps { this: OptiML => 
  
  /**
   * shapes
   * 
   * returns index vectors representing the indices of the particular shaped matrix
   */
      
  // def square(n: Rep[Int]) = ((unit(0)::n),(unit(0)::n))
  
  def utriangle(n: Rep[Int], includeDiagonal: Rep[Boolean] = unit(true)) = IndexVectorTriangular(n,includeDiagonal)
  
  //def triangular(n: Rep[Int], includeDiagonal: Rep[Boolean] = unit(false)) = optiml_triangular(n,includeDiagonal)  
  // def optiml_triangular[A:Manifest](n: Rep[Int], includeDiagonal: Rep[Boolean])(implicit ctx: SourceContext): Interface[IndexVector2]
    
  /**
   * aggregate
   */
  // 1D aggregate is just a Vector constructor!
  // def aggregate[A:Manifest](start: Rep[Int], end: Rep[Int])(block: Rep[Int] => Rep[A]): Rep[Vector[A]] = {
  //     optiml_aggregate(start, end, block)
  //   }
  
  def aggregateIf[A:Manifest](start: Rep[Int], end: Rep[Int])(cond: Rep[Int] => Rep[Boolean])(block: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]] = {
    optiml_aggregateif(start, end, cond, block)
  }  

  // 2D aggregate
  def aggregate[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector])
                           (block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]] = {
    optiml_aggregate2d(rows, cols, block)
  }
    
  def aggregateIf[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector])
                             (cond: (Rep[Int], Rep[Int]) => Rep[Boolean])(block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext) = {
    optiml_aggregate2dif(rows, cols, cond, block)
  }
  
  def aggregate[A:Manifest](idx: Interface[IndexVector2])
                           (block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]] = {
    optiml_aggregate2d_flat(idx, block)
  }
  

  def optiml_aggregateif[A:Manifest](start: Rep[Int], end: Rep[Int], cond: Rep[Int] => Rep[Boolean], block: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]]
  def optiml_aggregate2d[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                     block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]]
  def optiml_aggregate2dif[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                       cond: (Rep[Int], Rep[Int]) => Rep[Boolean], block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]]
  def optiml_aggregate2d_flat[A:Manifest](idx: Interface[IndexVector2],
                                          block: (Rep[Int], Rep[Int]) => Rep[A])(implicit ctx: SourceContext): Rep[DenseVector[A]]                                       

  // TODO: for some reason, the implicit ops conversions aren't kicking in for sum/min/max
  /**
   * sum
   */
  def sum[A:Manifest:Arith:Cloneable](start: Rep[Int], end: Rep[Int])(block: Rep[Int] => Rep[A])(implicit ctx: SourceContext) = optiml_sum(start, end, block)
  // sumRows currently just re-uses sumIf implementation; check if the condition always being true is actually slower than a specialized implementation with no conditional at all
  def sumRows[A:Manifest:Arith:Cloneable](start: Rep[Int], end: Rep[Int])(block: Rep[Int] => Rep[DenseVectorView[A]])(implicit ctx: SourceContext) = optiml_sumif[DenseVector[A],DenseVectorView[A]](start,end,i => unit(true),block)
  // def sum[R:Manifest:Arith:Cloneable,A:Manifest](start: Rep[Int], end: Rep[Int])(block: Rep[Int] => Rep[A])(implicit cs: CanSum[R,A], ctx: SourceContext) = optiml_sum[R,A](start, end, block)  
  def sumIf[A:Manifest:Arith:Cloneable](start: Rep[Int], end: Rep[Int])(cond: Rep[Int] => Rep[Boolean])(block: Rep[Int] => Rep[A])(implicit ctx: SourceContext) = optiml_sumif[A,A](start,end,cond,block)
  def sumRowsIf[A:Manifest:Arith:Cloneable](start: Rep[Int], end: Rep[Int])(cond: Rep[Int] => Rep[Boolean])(block: Rep[Int] => Rep[DenseVectorView[A]])(implicit ctx: SourceContext) = optiml_sumif[DenseVector[A],DenseVectorView[A]](start,end,cond,block)
  // def sumIf[R:Manifest:Arith:Cloneable,A:Manifest](start: Rep[Int], end: Rep[Int])(cond: Rep[Int] => Rep[Boolean])(block: Rep[Int] => Rep[A])(implicit cs: CanSum[R,A], ctx: SourceContext) = optiml_sumif[R,A](start,end,cond,block)
  
  def optiml_sum[A:Manifest:Arith:Cloneable](start: Rep[Int], end: Rep[Int], block: Rep[Int] => Rep[A])(implicit ctx: SourceContext): Rep[A]
  def optiml_sumif[R:Manifest:Arith:Cloneable,A:Manifest](start: Rep[Int], end: Rep[Int], cond: Rep[Int] => Rep[Boolean], block: Rep[Int] => Rep[A])(implicit cs: CanSum[R,A], ctx: SourceContext): Rep[R]

  /**
   *  IndexVector construction
   *  
   *  Overrides OptiLA's (x::n) to create an IndexVector instead of a RangeVector.
   */
  implicit def intToIndexOp(i: Int): IndexOp = new IndexOp(unit(i))
  implicit def repIntToIndexOp(i: Rep[Int]): IndexOp = new IndexOp(i)

  class IndexOp(val _end: Rep[Int]) {
    def ::(_start: Rep[Int]) = indexvector_range(_start, _end)
  }

  /**
   * IndexVector2 construction (Matrix construction)
   * ex.  (0::n, *) { { rowIdx => ... }
   * ex2. (*, 0::n) { { colIdx => ... }
   * ex3. (0::n,0::m) { (i,j) => .. }
   */
  class IndexWildcard
  val * = new IndexWildcard

  implicit def tuple2ToIndexVector1(tup: (Interface[IndexVector], IndexWildcard))(implicit overloaded1 : Overloaded1) = indexvector2_new_wc(tup._1, tup._2)
// currently not allowed
//  implicit def tuple2ToIndexVector2(tup: (IndexWildcard, Interface[IndexVector]))(implicit overloaded2 : Overloaded2) = indexvector2_new(indexvector2_wildcard(), tup._2)
  implicit def tuple2ToIndexVector3(tup: (Interface[IndexVector], Interface[IndexVector]))(implicit overloaded3 : Overloaded3) = indexvector2_new(tup._1, tup._2)


  /**
   * untilconverged
   */
  def untilconverged[A](x: Rep[A],
                        thresh: Rep[Double],
                        max_iter: Rep[Int] = unit(1000),
                        clone_prev_val: Rep[Boolean] = unit(false))
                        (block: Rep[A] => Rep[A])
                        (implicit diff: (Rep[A],Rep[A]) => Rep[Double], mA: Manifest[A], c: Cloneable[A], ctx: SourceContext): Rep[A]
    = optiml_untilconverged(x, (a: Rep[A]) => thresh, max_iter, clone_prev_val, block, diff)

  def untilconverged[A](x: Rep[A],
                        thresh: Rep[A] => Rep[Double],
                        max_iter: Rep[Int],
                        clone_prev_val: Rep[Boolean])
                        (block: Rep[A] => Rep[A])
                        (implicit diff: (Rep[A],Rep[A]) => Rep[Double], mA: Manifest[A], c: Cloneable[A], ctx: SourceContext): Rep[A]
    = optiml_untilconverged(x, thresh, max_iter, clone_prev_val, block, diff)
    
  def optiml_untilconverged[A:Manifest:Cloneable](x: Rep[A], thresh: Rep[A] => Rep[Double], max_iter: Rep[Int], clone_prev_val: Rep[Boolean],
                                                  block: Rep[A] => Rep[A], diff: (Rep[A],Rep[A]) => Rep[Double])(implicit ctx: SourceContext): Rep[A]


  def untilconverged[VD:Manifest,ED:Manifest](g: Rep[Graph[VD,ED]])
                    (block: Rep[Vertex[VD,ED]] => Rep[Unit])
                    (implicit ctx: SourceContext): Rep[Unit]
    = optiml_untilconverged(g, block)

  def optiml_untilconverged[VD:Manifest,ED:Manifest](g: Rep[Graph[VD,ED]], block: Rep[Vertex[VD,ED]] => Rep[Unit])(implicit ctx: SourceContext) : Rep[Unit]


  /**
   * gradient descent
   */
  
  def gradient(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double] = unit(.001), tol: Rep[Double] = unit(.0001),
               maxIter: Rep[Int] = unit(10000))(hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]
    = optiml_gradient(x, alpha, tol, maxIter, hyp)

  // stochastic: block() updates every jth parameter for every ith training sample
  //    while not converged{
  //      for i from 0 until m
  //        for j from 0 until n
  //            updatej(i,j)
  // hypothesis function maps a training example to a prediction

  // stochastic can only be parallelized across features, which is generally << samples
  def stochastic(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double] = unit(.001), tol: Rep[Double] = unit(.0001),
                 maxIter: Rep[Int] = unit(10000))(hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]
    = optiml_stochastic(x, alpha, tol, maxIter, hyp)

  // batch: block() updates each jth parameter from the sum of all ith training samples
  //    while not converged{
  //      for j from 0 until n
  //        j_update = sum((y(i) - h(x(i))*x(j,i)
  //        updatej(j_update)

  // in batch, the sum(...) loops over the entire training set independently, which is where the parallelism comes from
  // batch can be parallized across samples
  def batch(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double] = unit(.001), tol: Rep[Double] = unit(.0001),
               maxIter: Rep[Int] = unit(10000))(hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]
    = optiml_batch(x, alpha, tol, maxIter, hyp)


  def optiml_gradient(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], thresh: Rep[Double],
                      maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]

  def optiml_stochastic(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], thresh: Rep[Double],
                        maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]

  def optiml_batch(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], thresh: Rep[Double],
                   maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]]

  // coordinate ascent: analogous to stochastic gradient descent, but updates m parameters (alphas(0)...alphas(m-1))
  // at each update, all but alpha(i) must be held constant, so there are dependencies between every iteration

  //Loop until convergence {
  // For i = 1, . . . ,m, {
  //   alphas(i) := arg max alpha^(i) W(alphas(0), . . . , alphas(i-1), alphas^(i), alphas(i+1), . . . , alphas(m-1))
  // }
  //}

  /**
   * Nearest neighbor
   */
  // returns the index of the nearest neighbor of row inside data
  def nearestNeighborIndex[A:Manifest:Arith:Ordering:HasMinMax](row: Rep[Int], data: Rep[DenseMatrix[A]], allowSame: Rep[Boolean] = unit(true))(implicit ctx: SourceContext): Rep[Int]
    = optiml_nearest_neighbor_index(row, data, allowSame)

  def optiml_nearest_neighbor_index[A:Manifest:Arith:Ordering:HasMinMax](row: Rep[Int], data: Rep[DenseMatrix[A]], allowSame: Rep[Boolean])(implicit ctx: SourceContext): Rep[Int]

}

trait LanguageOpsExp extends LanguageOps with BaseFatExp with EffectExp {
  this: OptiMLExp with LanguageImplOps =>

  /**
   * Shapes
   */
  
  // case class Triangular[A:Manifest](n: Exp[Int], d: Exp[Boolean])
  //   extends DeliteOpSingleTask[(IndexVectorDense,IndexVectorDense)](reifyEffects(optiml_triangular_impl(n,d)))
  // 
  // def optiml_triangular[A:Manifest](n: Rep[Int], d: Rep[Boolean])(implicit ctx: SourceContext) = {
  //   val out = t2(Triangular(n,d))
  //   IndexVector2Tup(out._1,out._2)
  // }
  
  /**
   * Aggregate
   */
     
  case class AggregateIf[A:Manifest](start: Exp[Int], end: Exp[Int], cond: Exp[Int] => Exp[Boolean], func: Exp[Int] => Exp[A])
    extends DeliteOpFilter[Int,A,DenseVector[A]] {
  
    override def alloc = DenseVector[A](unit(0), unit(true))      
    val in = copyTransformedOrElse(_.in)(unit(0)::end-start)
    val size = copyTransformedOrElse(_.size)(end-start)
    
    def m = manifest[A]
  }
  
  def optiml_aggregateif[A:Manifest](start: Exp[Int], end: Exp[Int], cond: Exp[Int] => Exp[Boolean], block: Exp[Int] => Exp[A])(implicit ctx: SourceContext) = {
    reflectPure(AggregateIf(start,end,cond,block))
  }
  
  case class Aggregate2d[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                     func2: (Exp[Int], Exp[Int]) => Exp[A])
    extends DeliteOpMap[Int,A,DenseVector[A]] {
  
    val flatSize = rows.length*cols.length        
    override def alloc = DenseVector[A](flatSize, unit(true))      
    def func = i => func2(rows(i/cols.length), cols(i%cols.length)) // i/cols.length + rows(unit(0)), i%cols.length + cols(unit(0)))    
    val in = copyTransformedOrElse(_.in)(unit(0)::flatSize)
    val size = copyTransformedOrElse(_.size)(flatSize)
    
    def m = manifest[A]
  }
  
  def optiml_aggregate2d[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                     block: (Exp[Int], Exp[Int]) => Exp[A])(implicit ctx: SourceContext) = {
    
    reflectPure(Aggregate2d(rows, cols, block))
  }

  case class Aggregate2dIf[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                       cond2: (Exp[Int],Exp[Int]) => Exp[Boolean], func2: (Exp[Int], Exp[Int]) => Exp[A])
    extends DeliteOpFilter[Int,A,DenseVector[A]] {
  
    val flatSize = rows.length*cols.length    
    override def alloc(size: Exp[Int]) = DenseVector[A](size, unit(true))      
    def cond = i => cond2(rows(i/cols.length), cols(i%cols.length))  //cond2(i/cols.length + rows(unit(0)), i%cols.length + cols(unit(0)))
    def func = i => func2(rows(i/cols.length), cols(i%cols.length))  // func2(i/cols.length + rows(unit(0)), i%cols.length + cols(unit(0)))    
    val in = copyTransformedOrElse(_.in)(unit(0)::flatSize)
    val size = copyTransformedOrElse(_.size)(flatSize)
    
    def m = manifest[A]
  }
      
  def optiml_aggregate2dif[A:Manifest](rows: Interface[IndexVector], cols: Interface[IndexVector],
                                       cond: (Exp[Int], Exp[Int]) => Exp[Boolean], block: (Exp[Int], Exp[Int]) => Exp[A])(implicit ctx: SourceContext) = {
    
    reflectPure(Aggregate2dIf(rows, cols, cond, block))
  }

  case class Aggregate2dFlat[A:Manifest](intf: Interface[IndexVector2], func2: (Exp[Int],Exp[Int]) => Exp[A])
    extends DeliteOpMap[(Int,Int),A,DenseVector[A]] {

    val in = intf.ops.elem.asInstanceOf[Exp[Vector[(Int,Int)]]]
    override def alloc = DenseVector[A](intf.length, unit(true))      
    def func = i => func2(i._1,i._2)
    val size = copyTransformedOrElse(_.size)(intf.length)

    def m = manifest[A]
  }
 
  def optiml_aggregate2d_flat[A:Manifest](in: Interface[IndexVector2],
                                   block: (Exp[Int], Exp[Int]) => Exp[A])(implicit ctx: SourceContext) = {

    reflectPure(Aggregate2dFlat(in, block))
  }
                                       
  /**
   * Sum
   */

  case class Sum[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], map: Exp[Int] => Exp[A], init: Exp[A])(implicit ctx: SourceContext)
    extends DeliteOpMapReduce[Int,A] {

    override val mutable = true // can we do this automatically?
    
    val in = copyTransformedOrElse(_.in)(start::end)
    val size = copyTransformedOrElse(_.size)(end - start)
    //val zero = copyTransformedOrElse(_.zero)(reifyEffects(a.zero(init).mutable).res) 
    //val zero = copyTransformedBlockOrElse(_.zero)(reifyEffects(a.zero(init).mutable)) // FIXME: zero can be a fresh matrix, mutable calls cloneL
    //def zero = a.zero(init).mutable
    def zero = a.zero(init)
    override def accInit = a.zero(init).mutable
    def reduce = (a,b) => a += b
    
    def m = manifest[A]
    def a = implicitly[Arith[A]]    
    def c = implicitly[Cloneable[A]]
    def sc = implicitly[SourceContext]
  }

/*
  case class SumIf[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], cond: Exp[Int] => Exp[Boolean], func: Exp[Int] => Exp[A], init: Exp[A])
    extends DeliteOpFilterReduceFold[Int,A] {

    override val mutable = true // can we do this automatically?
    
    val in = copyTransformedOrElse(_.in)((start::end))
    val size = copyTransformedOrElse(_.size)(end - start)
    val zero = copyTransformedOrElse(_.zero)(reifyEffects(a.zero(init).mutable))
    def reduce = (a,b) => a += b  
/*    
    def func = (v) => (v, cond(v))
    def reduce = (a,b) => (if (a._2 && b._2) a._1 += b._1 else if (!a._2 && b._2)) b._2.mutable else if (), a._2 || b._2)
    
*/    
    def m = manifest[A]
    def a = implicitly[Arith[A]]
    def c = implicitly[Cloneable[A]]
  }
*/

  case class SumIf[R:Manifest:Arith:Cloneable,A:Manifest](start: Exp[Int], end: Exp[Int], co: Exp[Int] => Exp[Boolean], fu: Exp[Int] => Exp[A])(implicit canSum: CanSum[R,A], ctx: SourceContext) // TODO aks: CS into Arith
    extends DeliteOpFilterReduceFold[R] {

    override val mutable = true // can we do this automatically?

    val in = copyTransformedOrElse(_.in)((start::end))
    val size = copyTransformedOrElse(_.size)(end - start)
    //val zero = (copyTransformedOrElse(_.zero._1)(reifyEffects(a.empty)),copyTransformedOrElse(_.zero._2)(unit(-1))) // (zero, -1)
    val zero = (copyTransformedBlockOrElse(_.zero._1)(reifyEffects(a.empty)),copyTransformedBlockOrElse(_.zero._2)(reifyEffects(unit(-1))))

/*
    def func = (v) => (zero._1, reifyEffects(if (co(v)) v else -1))
    def reduce = (a,b) => (reifyEffects(if (b._2 >= 0) { if (a._2 >= 0) a._1 += fu(b._2) else { val bb = fu(b._2); bb.mutable }} else a._1), 
                                        if (b._2 >= 0) b._2 else a._2 ) // FIXME: will not work in parallel!!!
*/
    
    // rV = ((mutable(R), mutable(Int)), (R, Int))
    // rVSeq = ((zero, zero_2), (elem.func._1, elem.func._2))
    //         ((zero, init), (zero, index))
    // rvPar = (__act, __act._2), (rhs, rhs._2) 
    //         ((zero, init), (zero, init))
    def func = (v) => (zero._1, reifyEffects(v)) // will copy block that is zero._1, not just reference result!
    
    // FOR REDUCE SEQ
    // a._1 = accumulator
    // a._2 = zero_2 = initialization check: -1 if uninitialized, >= 0 otherwise
    // b._1 = unused 
    // b._2 = loop index
    
    // FOR REDUCE PAR
    // a._1 = accumulator
    // a._2 = act zero_2 = initialization check: -1 if uninitialized, >= 0 otherwise
    // b._1 = next value to reduce
    // b._2 = rhs zero_2 = initialization check: -1 if uninitialized, >= 0 otherwise
    
    
    // this is the reduce used inside each chunk (R,A) => R
    def reduceSeq = (a,b) => (reifyEffects(if (co(b._2)) { if (a._2 >= unit(0)) canSum.+=(a._1, fu(b._2)) else canSum.mutable(fu(b._2)) } else a._1), 
                              reifyEffects(if (co(b._2)) b._2 else a._2 )) // would not work in parallel...  // returns the current index (v) if the condition is true, or a._2, which is defaulted to -1 (uninitialized)

    // this is the reduce used in the tree (R,R) => R
    def reducePar = (a,b) => (reifyEffects(if (b._2 >= unit(0)) { if (a._2 >= unit(0)) a._1 += b._1 else b._1.mutable } else a._1), 
                              reifyEffects(if (b._2 >= unit(0)) b._2 else a._2))

    /*
        def step = (a,v) => (reifyEffects(if (b._2 >= 0) { if (a._2 >= 0) a._1 += fu(b._2) else { val bb = fu(b._2); bb.mutable }} else a._1), 
                                            if (b._2 >= 0) b._2 else a._2 ) // FIXME: will not work in parallel!!!
    */
    
    def m = manifest[R]
    def a = implicitly[Arith[R]]
    def c = implicitly[Cloneable[R]]
    def mA = manifest[A]
    def cs = implicitly[CanSum[R,A]]
    def sc = implicitly[SourceContext]
  }
  
  // FIXME: peeling off the first iteration manually can prevent fusion because the LoopOp is 1 smaller than its cousins
  // it also causes problems with the stencil analysis, since we have to account for the +1 offset on accesses.
  // TODO: what is the deired behavior if the range is empty?
  def optiml_sum[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], block: Exp[Int] => Exp[A])(implicit ctx: SourceContext) = {
    // val firstBlock = block(start)
    // val out = reflectPure(Sum(start+unit(1), end, block, firstBlock))
    // out + firstBlock
     
    // don't add it back in, just re-compute it to avoid the peeled iteration problems
    val zero = block(start) 
    reflectPure(Sum(start,end,block,zero))    
  }

  def optiml_sumif[R:Manifest:Arith:Cloneable,A:Manifest](start: Exp[Int], end: Exp[Int], cond: Exp[Int] => Exp[Boolean], block: Exp[Int] => Exp[A])(implicit cs: CanSum[R,A], ctx: SourceContext) = {
    reflectPure(SumIf[R,A](start, end, cond, block))
    /*val firstCond = cond(start)
      val firstBlock = block(start)
      val out = reflectPure(SumIf(start+1, end, cond, block, firstBlock))
      flatIf (firstCond) {
        out + firstBlock
      } {
        out
      }
  */

  }
  
/*  
  def optiml_sum[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], block: Exp[Int] => Exp[A]) = {

    //Sum(start, end, block)
    // HACK -- better scheduling performance in our apps, forces some expensive dependencies to be hoisted
    //reflectEffect(Sum(start, end, block))
    val firstBlock = block(start)
    val out = reflectPure(Sum(start+1, end, block, firstBlock))
    // add the first value back in (exploit commutativity of +)
//    out += firstBlock      
//    out.unsafeImmutable
    out + firstBlock
  }
  
  def optiml_sumif[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], cond: Exp[Int] => Exp[Boolean], block: Exp[Int] => Exp[A]) = {
    val firstCond = cond(start)
    val firstBlock = block(start)
    val out = reflectPure(SumIf(start+1, end, cond, block, firstBlock))
    // add the first value back in (exploit commutativity of +)
    if (firstCond) {
//      out += firstBlock
//      ()
      out + firstBlock
    }
//    out.unsafeImmutable
    else {
      out
    }
  }
*/  
  
  

  /**
   * untilconverged
   */
  
//  case class UntilConverged[A:Manifest:Cloneable](x: Exp[A], thresh: Exp[Double], max_iter: Exp[Int], clone_prev_val: Exp[Boolean],
//                                                  func: Exp[A] => Exp[A], diff: (Exp[A],Exp[A]) => Exp[Double])
//    extends DeliteOpSingleTask[A](reifyEffects(optiml_untilconverged_impl(x,thresh,max_iter,clone_prev_val,func,diff)))
//

//  def optiml_untilconverged[A:Manifest:Cloneable](x: Exp[A], thresh: Exp[Double], max_iter: Exp[Int], clone_prev_val: Exp[Boolean],
//                                                  block: Exp[A] => Exp[A], diff: (Exp[A],Exp[A]) => Exp[Double]) =
//    reflectEffect(UntilConverged(x, thresh, max_iter, clone_prev_val, block, diff))
//

  // for now, just unroll the implementation
  // we need a concept of a composite op to do this without unrolling, so that we can have a different result type than the while
  def optiml_untilconverged[VD:Manifest,ED:Manifest](g: Rep[Graph[VD,ED]], block: Rep[Vertex[VD,ED]] => Rep[Unit])(implicit ctx: SourceContext) = {
    val vertices = g.vertices
    val tasks = vertices.mutable // not a deep clone - contains the same vertex references as g.vertices
    
    while(tasks.length > unit(0)) {
      tasks.foreach(block) // usually updates graph vertices
      tasks.clear()
      
      // check all graph vertices for new tasks (not just the previous task list)
      for(i <- unit(0) until vertices.length) {
        tasks <<= vertices(i).tasks.distinct
        vertices(i).clearTasks()
      }
      // println("tasks: " + tasks.length)
    }
  }

  def optiml_untilconverged[A:Manifest:Cloneable](x: Exp[A], thresh: Exp[A] => Exp[Double], max_iter: Exp[Int], clone_prev_val: Exp[Boolean],
                                                  block: Exp[A] => Exp[A], diff: (Exp[A],Exp[A]) => Exp[Double])(implicit ctx: SourceContext) = {

    implicit def rv[T:Manifest](v: Var[T]): Exp[T] = readVar(v) //TODO: why isn't readVar implicit working?

    var delta = var_new(unit(scala.Double.MaxValue))
    var cur = var_new(x)
    var iter = var_new(unit(0))

    while (abs(delta) > thresh(cur) && (iter < max_iter)){
      val prev = if (clone_prev_val)
        cur.Clone()
      else
        cur

//      try{
        val next = block(cur)
//      }
//      catch{
//        case e: Exception => throw new ConvergenceException("Converging block threw exception: " + e)
//      }
      iter += 1
      //prev.AsInstanceOfOfL[Matrix[Any]].pprint
      //next.AsInstanceOf[Matrix[Any]].pprint
      delta = diff(next,prev)
      cur = next
      //println("(" + delta + ")")
    }

    if (iter == max_iter){
      //throw new ConvergenceException("Maximum iterations exceeded")
      println(unit("Maximum iterations exceeded"))
      returnL()
    }

    cur
  }

  /**
   * gradient descent for exponential family objective functions
   */
  private val MIN_BATCH_PROCS = 4
  def optiml_gradient(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], tol: Rep[Double],
                      maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]] = {

    val y = x.labels
    val numProcs = 8 //Delite.threadNum // dynamically set
    if (numProcs < MIN_BATCH_PROCS){
      stochastic(x, alpha, tol, maxIter)(hyp)
    }
    else{
      batch(x, alpha, tol, maxIter)(hyp)
    }
  }

  def optiml_stochastic(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], tol: Rep[Double],
                        maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]] = {

    val y = x.labels
    val theta = Vector.zeros(x.numFeatures).mutable
    untilconverged(theta, tol, maxIter, unit(true)) { theta =>
      val mem = y - x.data.mapRowsToVector(row => hyp(theta, row))        
      for (i <- unit(0) until x.numSamples) {        
        for (j <- 0::x.numFeatures) {
          val tmp = x(i) // SourceContext hack
          theta(j) = theta(j) + alpha*mem(i)*tmp(j)          
        }
      }
      theta
    }
  }

  def optiml_batch(x: Rep[SupervisedTrainingSet[Double,Double]], alpha: Rep[Double], tol: Rep[Double],
                   maxIter: Rep[Int], hyp: (Interface[Vector[Double]],Interface[Vector[Double]]) => Rep[Double])(implicit ctx: SourceContext): Rep[DenseVector[Double]] = {

    val y = x.labels
    val theta = Vector.zeros(x.numFeatures)
    val iter = var_new(unit(0))        
    
    // naive
    // with m = 10k, n = 10, iter = 100, ~1s after warmup
    /*
    val v = untilconverged(theta, tol, maxIter, unit(true)) { theta =>
      iter += unit(1)
      (unit(0)::x.numFeatures) { j => 
        val acc = sum(unit(0), x.numSamples) { i =>
          val tmp = x(i) // SourceContext hack
          tmp(j)*(y(i) - hyp(theta,x(i))) // parallel work
        }
        theta(j) + alpha*acc
      }      
    }    
    */
            
    // memoized
    // with m = 10k, n = 10, iter = 100, ~.22s after warmup, same answer as above    
    /*
    val v = untilconverged(theta, tol, maxIter, unit(true)) { theta =>
      iter += unit(1)
      val mem = y - x.data.mapRowsToVector(row => hyp(theta, row))
      val z = (unit(0)::x.numFeatures) { j => 
        val acc = sum(unit(0), x.numSamples) { i =>
          val tmp = x(i) // SourceContext hack
          tmp(j)*mem(i)
        }
        // println("gradient(" + j + "): " + acc)
        theta(j) + alpha*acc
      }
      // println("next value (b): ")
      // z.pprint
      z            
    } 
    */
                                     
    // inverted loops    
    // with m = 10k, n = 10, iter = 100, ~.27s with sum after warmup
    val v = untilconverged(theta, tol, maxIter, unit(true)) { cur => 
      iter += unit(1)
      // println("current theta: ")
      // cur.pprint
      
      /* 
      val gradient = sum(unit(0),x.numSamples) { i =>
        // produces the correct answer (~.27s after warmup)
        (x(i)*(y(i) - hyp(cur,x(i)))).Clone
        
        // produces the wrong answer (only correct on the first iteration)
        // (x(i)*(y(i) - hyp(cur,x(i))))
      } 
      */      
      
      val gradient = ((0::x.numSamples) { i =>        
        // no violated error, correct answer
        (x(i)*(y(i) - hyp(cur,x(i)))).Clone
        
        // violated order of effects with soa, correct answer without soa (~.23s after warmup)
        //(x(i)*(y(i) - hyp(cur,x(i))))        
      }).sum
            
      val z = cur + alpha*gradient // note that each term in the gradient has opposite sign as in Spark
      // println("gradient: ")
      // gradient.pprint
      // println("next value (c): ")
      // z.pprint
      z
    }  
        
    println("(batch gradient descent) converged in " + iter + " iterations")
    v    
  }


  /**
   * Nearest neighbor
   */
  def optiml_nearest_neighbor_index[A:Manifest:Arith:Ordering:HasMinMax](row: Rep[Int], m: Rep[DenseMatrix[A]], allowSame: Rep[Boolean])(implicit ctx: SourceContext): Rep[Int] = {
    // unroll
    val dists = (unit(0)::m.numRows){ i =>
      val d = dist(m(row),m(i))
      if (d == implicitly[Arith[A]].empty && !allowSame) implicitly[HasMinMax[A]].maxValue else d
    }
    dists.minIndex
    /*
    if (allowSame) dists.minIndex
    else {
      val same = dist(m(row), m(row))
      val f = dists filter {  _ != same }
      val x = f.min
      val z = dists find (x)
      z(0)
    }
    */
  }

  /**
   * Mirroring
   */
  override def mirror[A:Manifest](e: Def[A], f: Transformer)(implicit ctx: SourceContext): Exp[A] = {
    val f1 = transformerToInterfaceTransformer(f) // implicit not kicking in??
    e match {
    case e@AggregateIf(st,en,c,b) => reflectPure(new { override val original = Some(f,e) } with AggregateIf(f(st),f(en),f(c),f(b))(e.m))(mtype(manifest[A]), implicitly[SourceContext])    
    case e@Aggregate2d(x,y,b) => reflectPure(new { override val original = Some(f,e) } with Aggregate2d(f1(x),f1(y),f(b))(e.m))(mtype(manifest[A]), implicitly[SourceContext])    
    case e@Aggregate2dIf(x,y,c,b) => reflectPure(new { override val original = Some(f,e) } with Aggregate2dIf(f1(x),f1(y),f(c),f(b))(e.m))(mtype(manifest[A]), implicitly[SourceContext])    
    case e@Sum(st,en,b,init) => reflectPure(new { override val original = Some(f,e) } with Sum(f(st),f(en),f(b),f(init))(mtype(e.m), atype(e.a), ctype(e.c), e.sc))(mtype(manifest[A]), implicitly[SourceContext])
    case e@SumIf(st,en,c,b) => reflectPure(new { override val original = Some(f,e) } with SumIf(f(st),f(en),f(c),f(b))(e.m, e.a, e.c,e.mA,e.cs,e.sc))(mtype(manifest[A]), implicitly[SourceContext])
//    case e@SumIf(st,en,c,b,init) => reflectPure(new { override val original = Some(f,e) } with SumIf(f(st),f(en),f(c),f(b),f(init))(e.m, e.a, e.c))(mtype(manifest[A]))
    case Reflect(e@Sum(st,en,b,init), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with Sum(f(st),f(en),f(b),f(init))(mtype(e.m), atype(e.a), ctype(e.c), e.sc), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@SumIf(st,en,c,b), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with SumIf(f(st),f(en),f(c),f(b))(e.m,e.a,e.c,e.mA,e.cs,e.sc), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
//    case Reflect(e@SumIf(st,en,c,b,init), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with SumIf(f(st),f(en),f(c),f(b),f(init))(e.m,e.a,e.c), mapOver(f,u), f(es)))(mtype(manifest[A]))
    case Reflect(e@AggregateIf(st,en,c,b), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with AggregateIf(f(st),f(en),f(c),f(b))(e.m), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@Aggregate2d(x,y,b), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with Aggregate2d(f1(x),f1(y),f(b))(e.m), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)
    case Reflect(e@Aggregate2dIf(x,y,c,b), u, es) => reflectMirrored(Reflect(new { override val original = Some(f,e) } with Aggregate2dIf(f1(x),f1(y),f(c),f(b))(e.m), mapOver(f,u), f(es)))(mtype(manifest[A]), ctx)    
    case _ => super.mirror(e, f)
  }}.asInstanceOf[Exp[A]] // why??
}

trait LanguageOpsExpOpt extends LanguageOpsExp {
  this: OptiMLExp with LanguageImplOps =>
  
  override def optiml_sum[A:Manifest:Arith:Cloneable](start: Exp[Int], end: Exp[Int], block: Exp[Int] => Exp[A])(implicit ctx: SourceContext) = {
    val test = fresh[Int]
    reifyEffects(block(test)).res match {
      case Const(x) if x == 0 => unit(0.asInstanceOf[A])
      case _ => super.optiml_sum(start,end,block)
    }
  }
}
  
trait BaseGenLanguageOps extends GenericFatCodegen {
  val IR: LanguageOpsExp
  import IR._

}

trait ScalaGenLanguageOps extends ScalaGenEffect with BaseGenLanguageOps {
  val IR: LanguageOpsExp
  import IR._

  // override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
  //   rhs match {
  //     case _ => super.emitNode(sym, rhs)
  //   }
  // }
}

/*
trait CudaGenLanguageOps extends CudaGenBase with BaseGenLanguageOps {
  val IR: LanguageOpsExp
  import IR._

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
      rhs match {
        case _ => super.emitNode(sym, rhs)
     }
  }
}
*/
