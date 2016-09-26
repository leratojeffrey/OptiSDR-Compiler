//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object USRPApp extends OptiSDRApplicationRunner with usrpapp
trait usrpapp extends OptiSDRApplication
{
	def main() = 
	{
		//
		/*val numSymbs = 16
		val numSubs = 64
		val len = numSymbs*numSubs
		val Wc = 1.0e6f
		//
		//val tc1 = tic
		val tm = DenseVector[Float](1,true)
		val t = 0.0f::>1.0f<::1.0f*len
		val T = t*(1.0f/(100.0f*Wc))
		val carr = Cos(2.0f*pi*Wc*T)
		//
		//val insig = onesi(len)
		val insig = randb(len) // Generate Random Bits
		val encsig = convencode(insig,2,2) //
		//
		val qmsig   = QAM(encsig,pi/4.0f,1)
		//
		val ofdmsig = ifft(qmsig,numSubs)
		//
		val tmp = ofdmsig*carr
		//
		val outsig = cycpref(tmp(0::len),numSubs);
		//
		val psdout = Log10(abs(fft(outsig)))
		//val absout = abs(ftout)
		//val logout = Log10(absout) // = 10*log10(...)
		//
		//plot(absout,"Power dB vs Frequency")
		//breezeplot(psdout,"test")
		val psdplot = psplot(psdout,"PSD")
		//psdplot(outsig,xd,"PSD")
		//val psdpl = periodogram(outsig,tm,"OFDM PSD 1")
		//
		//psdpl.refresh(outsig,"OFDM PSD 1")
		//val anpsd = Log10(abs(fft(outsig,len)))
		//
		//psdplot.refresh(10*Log10(abs(fft(outsig,len))))*/
		/*var i = 0.0
		var running = true
		while(running)
		{
			psdplot.refresh(Log10(abs(fft(outsig,len)))*i)
			i+=1.0
		}*/
		val len=64*10000
		val hlen = len/2
		val dev = usrpinit(1) //
		//
		val test = usrpstream(len)//
		//val fout = fft(test) //GPU
		val Pxx = abs(fft(test)) //*conj(fout)/(len*len) // need to implement this tomorrow
		val lout = Log10(Pxx(hlen-512::hlen+512)) // And maybe Log10 of a complex signal
		val pl = rtplot(lout,"Real Time Power Plot") //
		//writeVector(lout,"test.csv")
		//
		//var i = 0
		var running = true
		while(running/*i<100*/)
		{
			val tmp = usrpstream(64*10000)
			val pout = abs(fft(tmp))
			val tlout = Log10(pout(hlen-512::hlen+512))
			pl.refresh(tlout) 
			// TODO: Implement a way to set- running to false in the refresh method
			// This will have to be something like:
			// running = pl.refresh() // In rtplot(), make a method that returns if Window is closed
			//
			//i+=1
		}
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