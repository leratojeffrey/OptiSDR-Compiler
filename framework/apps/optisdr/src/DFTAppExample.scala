import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object DFTAppExample extends OptiSDRApplicationRunner with mydft
trait mydft extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		//val x = linspace(0,8,8) 2097152
  		//val x1 = linspace(0,1000000,1000000)
  		val inp = SineWave(1,50,1, 0, 0.2f)
  		//
			//val dft1 = dft(inp)
			//
			//
			//plot(dft1,"Data","Number of Samples", "DFT data")
			//
  	}
}