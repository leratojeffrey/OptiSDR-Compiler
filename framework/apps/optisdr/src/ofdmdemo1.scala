import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object ofdmdemo extends OptiSDRApplicationRunner with ofdmapp
trait ofdmapp extends OptiSDRApplication
{
	def main() = 
	{
		//
		val numSubCarriers = 64
		val Wc = 8.0e6f
		//val vt = DenseVector.onesi(numSubCarriers) //randb(64*100) // Generate Random Bits
		val preambofdm4 = DenseVector[Float](1,true)
		val t = 0.0f::>1.0f<::1.0f*numSubCarriers
		val T = t*(1.0f/(100*Wc))
		val carrier = Cos(2.0f*pi*Wc*T)
		//
		//
		//val tc1 = tic
		val qmsignal = DenseVector[Float](64,true) //DenseVector.zeros(64)
		qmsignal(1)  = 1.0f
		//qmsignal.pprint
		val ofdmsignal     = ifft(qmsignal,numSubCarriers) // GPU -  Parallel Indexed Loops
		//
		//
		val rlvc = real(ofdmsignal)
		val imvc = imag(ofdmsignal)
		//rlvc.pprint
		//
		/*println(rlvc.length+" === "+qmsignal.length+" == "+qmsignal.length/numSubCarriers)
		for(i <- 0 until qmsignal.length/numSubCarriers)
		{
		//println(vt.length/64)
			val tst = rlvc(i*numSubCarriers::i*numSubCarriers+numSubCarriers)*carrier(i*numSubCarriers::i*numSubCarriers+numSubCarriers)
			preambofdm4.insertAll(i*numSubCarriers,tst) // GPU - Map
		}
		//println(vt.length/64)
		val preambofdm1 = preambofdm4(0::qmsignal.length) // GPU - Map
		*/
		//toc(tc1)
		//
		//plot(carrier,"0. The Carrier")
		plot(rlvc,"Real Values of OFDM Signal")
		plot(imvc,"Imaginary Values of OFDM Signal")
		//plot(preambofdm1,"1. OFDM Signal with 8MHz Carrier Freq.")
		//plot(abs(preambofdm1),"2. FFT of OFDM Signal")
		//periodogram(preambofdm1,t,"Signal PSD")
	}
}