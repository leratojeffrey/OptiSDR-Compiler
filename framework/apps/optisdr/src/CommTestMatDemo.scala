import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object CommTestMapApp extends OptiSDRApplicationRunner with commsmatapp
trait commsmatapp extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		
			val inp = DenseMatrix.randf(4096,16072) //[33554432 - 268.435mb - 0.6sec] ....[67108864 - 536.870912mb - 0.6sec]
			//
			val tc1 = tic
			val outp = commtest(inp,true)

			val outr = outp(0)
			println(outr(0)+":::"+toc(tc1))
			//val outp1 = commtest(inp)			
			//val outp2 = commtest(inp)
			//
			//println()
			//println(outp(0)+":::"+toc(tc1))
			//println(outp1(0)+":::"+outp2(0)+":::"+toc(tc1))
			//println("Length = "+inp.length)
  	}
}