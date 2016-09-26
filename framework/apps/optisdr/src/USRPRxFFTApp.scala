import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object USRPRXFFTApp extends OptiSDRApplicationRunner with usrprxapp
trait usrprxapp extends OptiSDRApplication
{
	def main() = 
	{
		//
		val len=64*10000
		val hlen = len/2
		val dev = usrpinit(1)
		//
		val test = usrpstream(len)//
		val fout = abs(fft(test)) //GPU
		val lout = Log10(fout(hlen-512::hlen+512)) //GPU
		val pl = psplot(lout,"PSD")
		//
		var i = 0
		//var running = true
		while(i<100/*running*/)
		{
			val tmp = usrpstream(64*10000)
			val pout = abs(fft(tmp))
			val tlout = Log10(pout(hlen-512::hlen+512))
			pl.refresh(tlout)
			//pl.refresh(abs(fft(tmp(hlen-256,hlen+256))))
			//if(i==1000)
			//	running=false
			i+=1
		}
		/*usrpstop(1)*/
		//
	}
}

// TODO: Uncomments Below for the SAICSIT Paper
/*
import uct.rrsg.dsl.optisdr._ // use sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext
// The OptiSDR  OFDM Appication
object OFDMApp extends OptiSDRApplicationRunner
with mofdmapp
// Define mofdmapp
trait mofdmapp extends OptiSDRApplication
{
	def main() = {
		val numSubs = 64
		val len = 16*numSubs		
		val Wc = 8.0e6f
		val t = 0.0f::>1.0f<::1024.0f*numSubs
		val T = t*(1.0f/(100*Wc))
		val carr = Cos(2.0f*pi*Wc*T)
		val cpre = carr(len-(64/4)::len)
		val psdplot = periodogram(zeros(len),"OFDM Plot")
		//
		streamprocess{
			val vt = onesi(len)
			val encsig = convencode(vt,2,2)
			val qmsig   = QAM(encsig,pi/4.0f,1)
			val ofsig = ifft(qmsig,numSubs)
			psdplot.refresh(ofdmsig,"PSD")
			val dout = cycpref(ofsig,cpre)
			iodevice(dout*carr,out,usrp)
		}
		// TODO: Busy With this
		streamprocess {
			val recvdsig = iodevice(in,usrp)
			psdplot.refresh(recvdsig,"PSD")
		}
}*/