import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object cufftdemo extends OptiSDRApplicationRunner with cufftapp
trait cufftapp extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		// 		
  		val L = 2048*8192
  		val fp = 2048
  		val imv = DenseVector[Float](L,true) //load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",L)
			//val tc1 = tic
  		val idata = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",L)
  		//
  		val inp = fft(ComplexDenseVector(idata,imv),fp)
  		//val inp = fft(idata)
  		
  		//println(toc(tc1))
  		//
  		//val inp2 = ifft(inp,fp)
  		//
  		val rlv = real(inp)
  		//val imvo = imag(inp)// Problem with Inheritance here, we need real([inp])
  		//
  		rlv(0::4).pprint  		
  		//imvo(0::4).pprint
  	}
}