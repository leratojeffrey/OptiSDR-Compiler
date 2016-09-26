//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object USRPApp2 extends OptiSDRApplicationRunner with usrpapp2
trait usrpapp2 extends OptiSDRApplication
{
	def main() = 
	{
		//
		val ftlen = 1024
		val len=8000000
		val Fcenter = 2.6e9
		val Fs = 8e6
		val Gain = 45
		val Bw = 32e6
		val hlen = len/2
		//
		val t = (-1.0f*512.0f)::>1.0f<::1.0f*512.0f
		val freqv = (((t/512d)*(Fs))+Fcenter)*1e-9
		//val cv = ComplexVector[Float](t,zeros(t.length))
		//pprintln(cv)
		//cv.pprint
		//	
		val dev = usrp_init(Fcenter,Fs,Gain,Bw,len) //
		//
		val test = usrpstream(len)//
		//
		// ddisp
		//
		//val Pxx = abs(fft(test))
		val Pxx = abs(fft(test,len)) //*conj(fout)/(len*len) // need to implement this tomorrow
		val lout = Log10(Pxx(hlen-512::hlen+512)) // And maybe Log10 of a complex signal
		val pl = rtplot(lout,freqv,"Real Time Power Plot") //
		//
		//writeVector(Pxx,"test.txt")
		//
		//var i = 0
		var running = true
		while(running)
		{
			val tmp = usrpstream(len)
			val pout = abs(fft(tmp,len))
			val tlout = Log10(pout(hlen-512::hlen+512))
			pl.refresh(tlout)
		}
		//
		//
		// println(test.length)
		//
		// val rl = real(test)
		//
		//Pxx(0::10).pprint
		//
		// println(test.length)
		//
		//val im = imag(test)
		//val rl = real(test)
		//writeVector(im,"test_im.csv")
		//writeVector(rl,"test_rl.csv")
		//
		// val fout = fft(test) //GPU
		//
	}
}