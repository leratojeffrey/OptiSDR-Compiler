import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object PSPPNetRADProcessor extends OptiSDRApplicationRunner with psppnetrad
trait psppnetrad extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		// Parallel Stream Processing NetRAD Data
  		// load(...) Method: Reads single 2048 chunk and process
  		//val outp = StreamProcessor(load("/srv/rrsg/data/projects_general/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin"),hilbert, fft)
  		//outp.computeWith(StreamProcessor(load("/srv/rrsg/data/projects_general/201x_NetRad/chirpRefSignals/e10_10_02_1400_00_P1_1_1000_S0_1_2047_node2.bin"),fft),mult)
  		
  		//val ifftout = ifft(outp)
  		//plot(ifftout)
  		//val nPulses = 1024//130000;
			//val StartPulseNo = 0;
			//val n3refsig = ComplexDenseVector(DenseVector.rand(300),DenseVector.rand(300))
			val n3refsig = loadc("n3refsig.mat",2048) // Copied from NetRADARD Matlab Scripts - 300 Samples @ 50MHz
			//val n3refsig = zeropad(refSignal,2048-refSignal.l_size) // Zero Padding for 2048-point FFT
			//plot(refSignal)
			//Plot(ComplexVector(real(n3refsig),imag(n3refsig)))
			//println(filename.dropRight(4)+"_rl.mat")
  		// Read raw NedRAD data: 2048 Chunks into Matrix
  		//val idata = loadRange("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1731_23_P1_1_130000_S0_1_2047_node3.bin")
  		val idata2 = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//val idata3 = load("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//val idata = loadMatrix("/media/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//
  		//val dataout2 = hilbert(idata2).init;
  		//val dataout2 = fft(idata2(0::8388608));
  		//val v1 = dataout2.rl
  		//plot(v1(0::2048))
  		// Pipelined hilbert and fft
  		//val dataout = pipeliner(idata2(StartPulseNo::StartPulseNo+nPulses),hilbert);
  		//dataout.rl.pprint
  		//idata(0::2048).pprint
  		//println(idata.length)
  		//idata2(0::2048).pprint
  		//println(idata2.length)
  		//idata3(0::2048).pprint
  		//println(idata3.length)
  		// Read Reference Signal: 2048 Chunks into Matrix
  		//val chirpref =  loadMatrix("/srv/rrsg/data/projects_general/201x_NetRad/chirpRefSignals/e10_10_02_1400_00_P1_1_1000_S0_1_2047_node2.bin")
  		//
  		// Pipelined hilbert and fft
  		//val dataout3 = pipeliner(idata(0::4096),fft) // 4096 x 2048 Matrix 
  		//val dataout3 = pipeliner(idata2(0::8388608),fft)
  		//plot(dataout3(0)) // Must make a Matrix Plotter
  		// Perform parallel
  		//
  		//val dataout = pipeliner(pipeout1,pipeout2,*,ifft)
  		//
  		//val x = linspace(0,8,8)
  		//x.pprint
  		//val Matched = DenseVector[ComplexDenseVector[Float]](1024,true)
  		//
  		//val refftout = cxvfft(n3refsig); // New Complex Mult - Need to Rename this Appropriately
  		//ddisp(test) //16777216
  		val datftout = StreamProcessor(StreamAdaptor(idata2(0::4194304),2048,4096),hilbert)
  		val xcorrout = CStreamProcessor(datftout(0::500),n3refsig,xcorr) // Parallel Complex Correlator
  		//val xcorrout = CStreamProcessor(ftout,ComplexDenseVector(imag(n3refsig),real(n3refsig)),mult) // Parallel Complex Mult
			//val multout = mult(datftout(0),refftout)
			//val ifftout = CStreamProcessor(xcorrout,ifft) // No Need for these Actually ....
			/*println(ftout.length)
			var i = 0
  		while(i < 4096)
  		{
				Matched(i) = cxcorr(ftout(i),n3refsig)
				i = i +1
  		}*/
			//val iftout = CStreamProcessor(multiply(ftout,refFFT),ifft);
  		//val toplot = abs(dataout2(0))
  		//plot(abs(real(xcorrout(0)),imag(xcorrout(0))))
  		//Plot(ComplexVector[Float](real(datftout(0)),imag(datftout(0))))
  		//Plot(ComplexVector[Float](real(xcorrout(0)),imag(xcorrout(0))))
  		//val dataout2 = StreamProcessor(idata2,fft);
  		//dataout2.ddisp
  		//println(dataout2.length)
  		//imagesc(xcorrout)
  	}
}