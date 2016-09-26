import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

// Import OptiSDR compiler
import uct.rrsg.dsl.optisdr._
// Also imports other lms libraries
object FFTNetRADApp extends OptiSDRApplicationRunner with netradapp
trait netradapp extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		// Chirp Reference Signal - 300 Samples @ 50MHz
  		val n3refsig = loadc("n3refsig.mat",2048)
  		// Read raw NedRAD data: 2048 Chunks into Matrix
  		val idata = loadMatrix("/srv/rrsg/data/projects_general/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//
  		/*val Matched = DenseVector[DenseVector[Double]](1024,true)
			//
			var i = 0
  		//
  		val ftrefsig = fft(ComplexVector(real(n3refsig),imag(n3refsig))) // FFT Ref. Signal
  		//plot(ftrefsig)
  		while(i < Matched.length) // 0 - 133 ?? 0 - 43333 - Matrix = 10000 x 4096,
  		//for(v <- idata)
  		{
  			// Hilbert Transform the input data
				val htout = khilbert(idata(i))
				// Take FFT of Hilbert Transformed data samples
				val ftout = fft(htout)
				// Mix FFT of HT data and FFT of Chirp
				val multout = ftout.mult(ftrefsig)
				// Take iFFT of multout
				val iftout = ifft(multout)
				//
				Matched(i) = abs(iftout)
				i += 1
  		}*/
  		//plot(imag(Matched(8191)))
  		//imagesc(Matched,"Number of Pulses","Range ?","Fast-Time vs Slow-Time NetRAD Plot")
  	}
}