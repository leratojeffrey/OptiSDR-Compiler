import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object HilbertApp extends OptiSDRApplicationRunner with myhilb
trait myhilb extends OptiSDRApplication
{ 
  	def main() = 
  	{
			//val xdata =  load("/srv/rrsg/data/projects_general/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//xdata(0::10).pprint
		val T = 2
  		val x = linspace(0,T*1024,T*1024)
  		//x.pprint
 		// Parallel Hilbert Transform: Returns Complex Signal Vector of Type Double - Something like analytic signal
		//%
		//val hout  = hilbert(x)
		val htout = hilbert(x) //
		//println(htout.length)
		//val htout = hilbert(xdata(0::2048))
		//val htout2 = philbert(xdata(0::2048))
		//
		//val rl=htout.real
		//plot(htout.abs)
		//plot(imag(htout2))
		//
		//hout.ddisp
		//hout2.ddisp

  	}
}
