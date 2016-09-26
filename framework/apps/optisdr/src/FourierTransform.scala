import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object FftApp extends OptiSDRApplicationRunner with myfft
trait myfft extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		//
			//
  		//val idata = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//val idata = loadDoubles("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//
  		/*val L = 2048*8192//
  		val fp = 2048
  		val imv = DenseMatrix[Double](L/fp,fp) //
  		val h_signal = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin",L)
  		//
  		val rlin = StreamIn(h_signal,fp) // Seq
  		//
  		val inp = fftmat(rlin,imv)
  		//
			val rlv = real(inp,1)
			//
			rlv(0::4).pprint
			//
		*/
			val t = DenseVector[Double](1,true)
			val L = 2048
			val v2 = DenseVector[Float](L,true)
			v2(1)  = 1.0f
			//
			val ftout = ifft(v2,DenseVector.zerosf(L),64)
			//
			val rlv = real(ftout)
			//println("Do we get here...!")
			//
			plot(rlv(0::64),"iFFT Plot")
			//
			//
  	}
}
