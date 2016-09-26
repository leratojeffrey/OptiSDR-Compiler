import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object OFDMApp2 extends OptiSDRApplicationRunner with mofdmapp2
trait mofdmapp2 extends OptiSDRApplication
{
	def main() = 
	{
		//
		val numSymbs = 4096
		val numSubs = 64
		val len = numSymbs*numSubs //262144 samples = 200Ks
		val Wc = 131072f
		val Fs = len.toFloat
		//
		//val tc1 = tic
		val tm = DenseVector[Float](1,true)
		val t = 0.0f::>1.0f/Fs<::1.0f
		val carr = Cos(2.0f*pi*Wc*t)
		//
		val insig = onesi(len)
		//val insig = zerosi(len)
		//insig(0) = 1
		//val insig = randb(len) // Generate Random Bits
		val encsig = convencode(insig,2,2) //
		//
		val qmsig   = QAM(encsig,pi/4.0f,1)
		//
		val ofdmsig = ifft(qmsig,numSubs)
		//
		val tmp = ofdmsig*carr
		//
		val outsig = cycpref(real(tmp),numSubs);
		//
		val T = (-0.5f*outsig.length::>1.0f<::0.5f*outsig.length)/numSubs
		val psdout = Log10(fft(outsig))*10.0f
		//println(outsig.length+" == "+T.length+" == "+psdout.length)
		//plot(carr,t,"Amplitude","Time (s)","Carrier Signal Plot")
		//plot(tmp.real,T,"Amplitude","Time (s)","OFDM Real Signal Plot")
		plot(psdout.real,T,"Power (dB/Hz)","Frequency (Hz)","Spectral Density Plot")
		//val absout = abs(ftout)
		//val logout = Log10(absout) // = 10*log10(...)
		//
		//plot(absout,"Power dB vs Frequency")
		//breezeplot(psdout,"test")
		//val psdplot = psplot(psdout,"PSD")
		//psdplot(outsig,xd,"PSD")
		//val psdpl = periodogram(outsig,tm,"OFDM PSD 1")
		//
		//psdpl.refresh(outsig,"OFDM PSD 1")
		//val anpsd = Log10(abs(fft(outsig,len)))
		//
		//psdplot.refresh(10*Log10(abs(fft(outsig,len))))
		/*var i = 0.0
		var running = true
		while(running)
		{
			psdplot.refresh(Log10(abs(fft(outsig,len)))*i)
			i+=1.0
		}*/
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