import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object SDRMatrixExampleRunner extends OptiSDRApplicationRunner with SDRMatrixExample
trait SDRMatrixExample extends OptiSDRApplication { 
   def main() = 
   {  
			val m1 = DenseMatrix[Float](3,3)
			//val invm1 = m1.inv
			val m2 = DenseVector.randf(3)

			for(i<-0 until 3)
			{
				m1(i) = m2
			}
			//val invm2 = m2.inv
   }
}