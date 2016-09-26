import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object EncodeAppRunner extends OptiSDRApplicationRunner with EncodeApp
trait EncodeApp extends OptiSDRApplication
{
	def main() = 
	{
		//
		val numSubCarriers = 64
		val Wc = 8.0e6f
		val vt = DenseVector.onesi(16*numSubCarriers) //randb(64*1024) // Generate Random Bits
		val preambofdm4 = DenseVector[Float](1,true)
		val t = 0.0f::>1.0f<::1024.0f*numSubCarriers
		val T = t*(1.0f/(100*Wc))
		val carrier = Cos(2.0f*pi*Wc*T)
		//
		vt.pprint
		val tc1 = tic
		val convcodes2 = convencode(vt,2,2) // TODO: Must run in GPU
		convcodes2.pprint
		println(convcodes2.length)
		val qmsignal   = QAM(convcodes2,pi/4.0f,1)// GPU - Parallel Indexed Loop
		println(qmsignal.length)
		//qmsignal.ddisp
		val ofdmsignal = ifft(qmsignal,numSubCarriers) // GPU -  Parallel Indexed Loops
		//
		//
		val rlvc = real(ofdmsignal)
		//
		/*for(i <- 0 until qmsignal.length/numSubCarriers)
		{
		//println(vt.length/64)
			val tst = rlvc(i*numSubCarriers::i*numSubCarriers+numSubCarriers)*carrier(i*numSubCarriers::i*numSubCarriers+numSubCarriers)
			preambofdm4.insertAll(i*numSubCarriers,tst) // GPU - Map
		}*/
		//
		//val preambofdm1 = preambofdm4(0::qmsignal.length) // GPU - Map
		//toc(tc1)
		//
		plot(rlvc,"Real Values of OFDM Signal")
		//plot(preambofdm1,"OFDM Signal with 8MHz Carrier Freq.")
		//periodogram(ofdmsignal,t,"OFDM Signal PSD")
	}
}