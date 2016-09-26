//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%% How to Mount RRSG NetRAD Drive using sshfs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%% sshfs lmohapi@rrsg.uct.ac.za:/srv/data/projects_general/201x_NetRad /media/201x_NetRad %%%
//%%% Nomoro ea Lekunutu: Ke Spele kapo Lemao Laka
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object RFSweepApp extends OptiSDRApplicationRunner with rfsapp
trait rfsapp extends OptiSDRApplication
{
	def main() = 
	{
		//
		var Fcenter = 2.4e9f
		val num_samps = 1024f // Million Point FFT
		val Fs = 8e6f//1000000f
		val t = Fcenter-(Fs/2.0f)::>Fs/num_samps.toFloat<::Fcenter+(Fs/2.0f) //0:1.0:2.0 change this to Matlab/Octave like
		val Ti = Fcenter-(Fs/2.0f)::>Fs/(5.0f*num_samps.toFloat)<::Fcenter+(Fs/2.0f)
		//val T = 0f::>1.0f/Fs<::1f
		//println(t.length+" :: "+T.length)
		//
		val x = Sin(2.0f*pi*t*512f)
		val X = fft(x)
		val Y = X*conj(X)/(num_samps.toFloat*num_samps)//
		//
		val Pxx = Log10(Y)*10.0// We need to Implement Infix Operation for this
		val Pww = Pxx.real
		//
		val num_sweeps = 5*num_samps
		val T = Ti*1e-9
		val Pwr = Pww //<<zeros(num_sweeps.toInt-Pww.length)
		//println(init_p.length)
		val pl = rtplot(Pwr,T,"Power vs Frequency Plot")
		/*
		val Pww = abs(Pxx)
		//
		plot(x,t,"Amplitude","Time (s)","Original Signal Plot")
		plot(Pww,t,"Power (dB/Hz)","Frequency (Hz)","Spectral Density Plot")*/
		//
		var running = 0
		var F_s = Fs
		while(running<11)
		{
			// New Samples
			F_s = F_s + Fs // Create an Array of These Increments
			val Fc = Fcenter + F_s// Accesss here with i Mod 10 = i%10
			val tu = Fc-(F_s/2.0f)::>F_s/num_samps.toFloat<::Fc+(F_s/2.0f)
			val xu = Sin(2.0f*pi*tu*512f)
			val Xu = fft(xu)
			val Puu = (Xu*conj(Xu))/(num_samps.toFloat*num_samps)
			val tlout = Log10(Puu)*10.0
			//
			//t <<= tu
			//Pww <<= tlout.real
			val Pwru = tlout.real //Pww<<zeros(num_sweeps.toInt-Pww.length)
			//Pwr.insertAll(running*num_samps.toInt,tlout.real)
			pl.refresh(Pwru) // create an append/update function - it must appends until x-axis fixed length
			//
			println(F_s+"::"+Fc)
			running = running + 1
		}
		//		
	}
}