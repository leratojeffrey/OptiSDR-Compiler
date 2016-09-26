import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object QAMAppRunner extends OptiSDRApplicationRunner with QAMApp
trait QAMApp extends OptiSDRApplication
{
	def main() = 
	{
		//
		//
		//	
		val tst = randb(256*4096)
		//
    val tc1 = tic
		val encoded_signal = encode(tst,2)
		val qmsignal       = QAM(encoded_signal,pi/2.0f,1) // QPSK - Baseband 
		val ofdmsignal     = ifft(qmsignal,256) // Return ComplexDenseVector
		//
		val tPxx2 = fft(ofdmsignal)
		//
		toc(tc1)
		//
		//println(qmsignal.length+" <=Length=> "+ftout.length+" <=Value=> "+ftout(0)+" === "+ftout(ftout.length-1))
		//
		//
		//
		/*val k = 1.0 * tPxx.length
		var i = 0
		val Pxx = DenseVector[Float](tPxx.length,true)
		for(e<-tPxx)
		{
			Pxx(i) = 10*log10(pow(e,2.0/k))
			//println(Pxx(i)+":::::::"+e)
			i = i + 1
		}*/
		
		val preambofdm = ofdmsignal<<preamble(10,0.50f,0.55f);
		//val ftout = ComplexDenseVector(tPxx2,1)
		//val awgnsignal = preambofdm + 5.0*DenseVector.rand(preamble1.length)
		//
		plot(real(qmsignal),"QAM Signal")
		plot(real(preambofdm),"OFDM Real Signal With Preamble")
		//plot(real(ftout),"FFT of OFDM Imaginary Signal With Preamble")
		//
		val vt = DenseVector(1,1,0,1,1,0,1,1)
		val convcodes = convencode(vt)
		convcodes.pprint
		val convcodes2 = convencode(vt,2,2)
		convcodes2.pprint
		val M = DenseVector(1,0,1,0)
		val cyccodes = cyclicencode(M,7,4)
		cyccodes.pprint
	}
}