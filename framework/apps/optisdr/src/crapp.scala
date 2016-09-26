import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object CRAppRunner extends OptiSDRApplicationRunner with CRApp
trait CRApp extends OptiSDRApplication
{
	def main() = 
	{
		//
    val tc1 = tic
		//
		val preamble1 = preamble(10)
		//plot(preamble1.toFloat + 5.0f*DenseVector.randf(preamble1.length))
		
		toc(tc1)

	}
}