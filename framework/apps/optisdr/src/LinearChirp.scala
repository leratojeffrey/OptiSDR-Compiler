import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object LinearChirp extends OptiSDRApplicationRunner with chirpapp
trait chirpapp extends OptiSDRApplication
{
	def main() = 
	{
		//
		// chirp(t, f0, t1, f1, phase)
  		val t = 0.0f::>0.001f<::5.0f //args(0)
  		val f0 =  0f//args(1)
		val f1 = 1000f
		val t1 = 100f
		var phase = 0f//args(5)
		phase = 2f*pi*phase/360f
		//
		// chirp([0:0.001:5],0, 10,1000,0);
		val a = pi*(f1 - f0)/t1;
		val b = 2f*pi*f0;
		val y = Cos(a*Pow(t,2)+(b*t)+phase); //
		plot(y,"Amplitute","Time (s)");
		val ftout = abs(fft(y))
		plot(ftout,"Magnitude","Frequency (Hz)");
		//
	}
}
