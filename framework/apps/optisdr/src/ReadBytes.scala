import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object BytesApp extends OptiSDRApplicationRunner with bytesapp
trait bytesapp extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		// 		
  		val L = 2048*8192
  		val fp = 2048
  		val imv = DenseMatrix[Float](L/fp,fp)
			//val tc1 = tic
  		val idata = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",L)
  		//
  		val rlin = StreamIn(idata,fp) // Chunk Data
  		//
  		//val inp = fftmat(rlin,imv) // FFT - Change this to fft([x])
  		//val inp = qfft(rlin,imv)
  		//println(toc(tc1))
  		//
  		//val rlv = inp(0)// Problem with Inheritance here, we need real([inp])
  		//
  		//rlv(0::10).pprint
			//
  	}
}