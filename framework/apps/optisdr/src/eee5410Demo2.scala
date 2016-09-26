//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object eee5410Demo2 extends OptiSDRApplicationRunner with eee54102
trait eee54102 extends OptiSDRApplication
{
	def main() = 
	{
		//
		val buff = DenseVector[Double](1, true)
		val len= 1000000
		val len2 = len*len
		val Fcenter = 2.5e9
		val Fs = 8e6
		val Gain = 64
		val Bw = 40e6
		val hlen = len/2
		//
		val uB = Fcenter+hlen
		val lB = Fcenter-hlen
		val T  = lB.toFloat::>1.0f<::uB.toFloat
		//
		val dev = usrp_init(Fcenter,Fs,Gain,Bw,len) // Config. USRP
		//
		val bwin = blackmanharris(len)
		val x = usrpstream(len)*bwin // Capture data at initial Center Freq. and apply blackman-harris window
		val X = fft(x,len)/len2 // Perform Normalized FFT
		val lout = 20*Log10(abs(X)) // Compute power*/
		/*buff <<= lout //(len) // Take some middle 1024 samples for plotting
		//
		var Fc = Fcenter // Initialize Fc to initial Center Frequency
		while(Fc < 2.5e9) // RF Sweep
		{
			Fc = Fc + 8e6 // Increment the Center frequency
			val xu = usrpstream(len,Fc) // Capture data at new Freq.
			val Xu = fft(xu,len)/len // FFT and normalize
			val tlout = 20*Log10(Xu.abs) // Compute power
			buff <<= tlout //(len) // Take some middle 1024 samples for plotting
		}
		//
		//plot(buff,"RF Sweep Plot")
		plot(lout,"RF Sweep Plot")*/
		//val hamm = blackmanharris(len)
		//
		//plot(hamm,"Blackman-Harris Windows Test")
		//
		plot(lout,T,"Power (dBc)","Freq (Hz)","RF Sweep Plot")
	}
}