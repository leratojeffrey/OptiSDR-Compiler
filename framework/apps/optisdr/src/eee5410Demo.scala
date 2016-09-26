//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object eee5410Demo extends OptiSDRApplicationRunner with eee5410
trait eee5410 extends OptiSDRApplication
{
	def main() = 
	{
		//
		val buff = DenseVector[Float](1, true)
		val ftlen = 1024
		val len=1000000
		val len2 = len*len
		val Fcenter = 2.48e9
		val Fs = 1e6
		val Gain = 45
		val Bw = 32e6
		val hlen = len/2
		//
		//val t = 0.0f::>1.0f<::1.0f*10
		//	
		val dev = usrp_init(Fcenter,Fs,Gain,Bw,len) //
		//
		val x = usrpstream(len)
		//
		val X = fft(x,len)
		val Pxx = (X*conj(X))/(len2) //
		val lout = Log10(Pxx(hlen-512,hlen+512))*20.0 //
		//
		buff <<= real(lout)
		//
		//
		var Fc = Fcenter
		while(Fc < 2.532e9)
		{
			Fc = Fc + 8e6 // Increment the Center frequency
			val xu = usrpstream(len,Fc) // Capture data at new Freq.
			val Xu = fft(xu,len) // FFT results and normalize
			val Puu = (Xu*conj(Xu))/(len2)
			val tlout = Log10(Puu(hlen-512,hlen+512))*20.0
			buff <<= real(tlout)
		}
		//
		plot(buff,"RF Sweep Plot")
	}
}
