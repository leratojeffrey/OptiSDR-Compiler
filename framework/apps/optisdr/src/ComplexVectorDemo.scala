import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object ComplexVectorDemo extends OptiSDRApplicationRunner with complexvecdemo
trait complexvecdemo extends OptiSDRApplication { 
   def main() = 
   {  
   		val v = DenseVector.randf(8)
   		val v2 = DenseVector.randf(100)
   		
			val x = linspace(0,8,8)
			//val cxv1 = ComplexDenseVector[Double](10)
			//cxv1.rl = v
			//cxv1.im  = v
			//ddisp(cxv1)
			//
			val cxv2 = ComplexDenseVector(x,v)
			val cxv1 = ComplexDenseVector(v,x)
			//ddisp(cxv2)
			//cxv2 = ComplexDenseVector(v2,v2)

			//real(cxv2).pprint

			//val cxvmat1 = DenseVector[ComplexDenseVector[Double]](2,true)
			//cxvmat1(0) = cxv2
			//cxvmat1(1) = cxv2
			//val cxvmat2 = DenseMatrix[ComplexDenseVector[Double]](2,100)
			//cxvmat2(0) = cxv2
			//cxvmat2(1) = cxv2
			//
			val ftout = mult(cxv2,cxv1)
			ddisp(ftout)
   }
}