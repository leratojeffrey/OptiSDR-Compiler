import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object ComplexSignalApp extends OptiSDRApplicationRunner with CXApp
trait CXApp extends OptiSDRApplication
{
	def main() = 
	{
		//
    //val csig1 = ComplexSignal[Float](10)
    //val csig2 = ComplexSignal[Float](10)
    
		//
    //val out = csig1*csig2;
    //
    // Testing Sinusiods
		//
		val Wc = 1.0e6f
		val Fs = 8192f//1000000f
		val t = 0.0f::>1.0f/Fs<::1.0f //0:1.0:2.0 change this to Matlab/Octave like
		val T = -Fs/2.0f::>1.0f<::Fs/2.0f
		//
		//println(T.length) 
		val x = Sin(2.0f*pi*t*500f)
		val X = fft(x)
		val Y = X*conj(X)/(Fs*Fs)//
		//val Magn = abs(Y)
		val Pxx = Log10(Y)*10.0f//*10.0f
		val Pww = abs(Pxx)
		//
		//Pxx(0,10).ddisp
		//
		plot(x,t,"Amplitude","Time (s)","Original Signal Plot")
		plot(Pww,T,"Power (dB/Hz)","Frequency (Hz)","Spectral Density Plot")
		//plot(Magn,T,"Magnitude (|Y(t)|)","Frequency (Hz)","Frequency Plot")
		//
	}
}