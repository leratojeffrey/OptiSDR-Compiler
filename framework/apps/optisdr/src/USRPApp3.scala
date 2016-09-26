//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object USRPApp3 extends OptiSDRApplicationRunner with usrpapp3
trait usrpapp3 extends OptiSDRApplication
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
		//val t = 0.0f::>1.0f<::1.0f*10
		//	
		val dev = usrp_init(Fcenter,Fs,Gain,Bw,len) //
		//
		val x = usrpstream(len)
		//
		val X = fft(x,len)
		val Pxx = (X*conj(X))/(len*len) // (X*conj(X)) = pow(real(X),2)+pow(imag(X),2)
		val lout = Log10(Pxx(hlen-512,hlen+512))*10.0 //
		//
		//writeVector(real(lout),"test.txt")
		//writeVector(imag(lout),"test2.txt")
		//writeVector(real(lout),"test3.txt")
		//
		val pl = rtplot(real(lout),"Power vs Frequency Plot") //
		//
		//
		var running = true
		while(running)
		{
			val xu = usrpstream(len)
			val Xu = fft(xu,len)
			val Puu = (Xu*conj(Xu))/(len*len)
			val tlout = Log10(Puu(hlen-512,hlen+512))
			pl.refresh(real(tlout))
		}
		//
	}
}
