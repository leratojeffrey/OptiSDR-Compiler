//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object USRPApp4 extends OptiSDRApplicationRunner with usrpapp4
trait usrpapp4 extends OptiSDRApplication
{
	def main() = 
	{
		//
		val ftlen = 1024
		val len=1000000
		val Fcenter = 2.6e9
		val Fs = 8e6
		val Gain = 45
		val Bw = 32e6
		val hlen = len/2
		//
		val t = (-1.0f*512.0f)::>1.0f<::1.0f*512.0f
		val freqv = (((t/512d)*(Fs))+Fcenter)*1e-9
		//freqv.pprint
		//	
		val dev = usrp_init(Fcenter,Fs,Gain,Bw,len) //
		//
		val x = usrpstream(len)
		//
		val X = fft(x,len)
		val Pxx = (X*conj(X))/(len*len)
		val lout = Log10(Pxx(hlen-512,hlen+512))*10.0
		//
		println(lout.length+":::::"+freqv.length)
		//
		val pl = rtplot(real(lout)-25,freqv,"Power vs Frequency Plot")
		//
		//
		/*var running = true
		while(running)
		{
			val xu = usrpstream(len)
			val Xu = fft(xu,len)
			val Puu = (Xu*conj(Xu))/(len*len)
			val tlout = Log10(Puu(hlen-512,hlen+512))*10.0
			pl.refresh(real(tlout))
		}*/
		//
	}
}