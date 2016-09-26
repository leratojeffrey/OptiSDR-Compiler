import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object FFTNetRADProcessor extends OptiSDRApplicationRunner with fftnetrad
trait fftnetrad extends OptiSDRApplication
{ 
  	def main() = 
  	{
  		//
  		//val n3refsig = loadc("n3refsig.mat",4096) // Copied from NetRADARD Matlab Scripts - 300 Samples @ 50MHz
			//
  		//val idata2 = load("/home/swinberg/OptiSDR/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//
  		var arg0 = args(0)
  		//var arg1 = args(1)
  		val i = 2d
  		var t = i
  		t=7
  		println("*********************")
  		println("Testing Args: "+arg0)
  		//println("Testing Args: "+arg1)
  		println(t)
  		println("*********************")
  		//
  		val L = 1024*2048
  		//val idata = StreamAdaptor(DenseVector.randf(L),2048,4096)// 16777216
  		//
  		val x = DenseVector.randf(L)
  		x(0::10).pprint
  		//
  		/*val pspp = streamprocessor(x,fft)
  		//val pspp = psppfft(x)
  		//val datftout = StreamProcessor(idata,hilbert,fft) // Hilbert transform and FFT Input Data
  		//val pspp2 = streamprocessor(x,fft)
  		//
  		val rlv = real(pspp)
  		val ilv = imag(pspp)
  		//
  		rlv(0::10).pprint
  		x(0::10).pprint*/
  		//
  		// Testing Streamer
  		//
  		val pstrm = ParallelStreamer(Src(x),tasks(hilbert,fft),Argsv(argsv(2048),argsv(2048)))
  		val srcvec = pstrm.source
  		//srcvec.pprint
  		val parvec = pstrm.parameters
  		val funcvec = pstrm.functions
  		println("\n ************* Adding More Tasks **************** \n")
  		pstrm.addTask(ifft)
  		//pstrm.addTask(xcorr,Args(x,args(4096))) // xcorr previous task output with x argument in the stream
  		pstrm.addTask(xcorr,argsv(4096))
  		println("\n ************* Displaying Parameters **************** \n")
  		//
  		//parvec.pprint
  		//
  		pstrm.pprint
  		pstrm.execStreaming() // Execute in a stream based format, FIFO execution
  		//
  		//
			println("\n")
			//
			//
			//type UDT = DataType{var x: Int; var y: Int}
			/*val cv = Complex(120,43)
			val rl = cv.real
			println(rl+" + i"+cv.imag)
			println("\n")
			//
			val cvp = cv+Complex(23,100)
			//
			println(cvp.real+" + i"+cvp.imag)
			//
			println("\n")
			cv+=Complex(100,100)
			//
			println(cv.real+" + i"+cv.imag)
			//
			val rlv = DenseVector(1f,2f,3f,4f,5f,6f,7f,8f,9f,10f)
			val imv = DenseVector(10f,9f,8f,7f,6f,5f,4f,3f,2f,1f)
			//
			val cxv1 = ComplexSignal(rlv,imv)
			val cxv2 = ComplexSignal(imv,rlv)
			//
			val out = cxv1+cxv2;
			val out1 = cxv1*cxv2;
			val out2 = cxv1-cxv2;
			val out3 = cxv1/cxv2;
			//
			out.real.pprint
			out1.real.pprint
			out2.real.pprint
			out3.real.pprint
			//
			//val cxv3= DenseVector[Complex](100,true)
			*/
			//cxv3.pprint
			/*
			//
			// Check NetRAD Nan
			//
			val tx = DenseVector[Float](1,true)
			val idata2 = load("/home/optisdr/OptiSDR/201x_NetRad/ZA_Trials_2011_06/radarData/2011-06-04/e11_06_04_1740_34_P1_1_130000_S0_1_2047_node3.bin")
  		//
  		plot(idata2(0::2048),"Original Signal")
  		val X = fft(idata2(0::2048*10000))//ComplexDenseVector(idata2(0::2048*10000),zeros(2048*10000))  		
  		plot(X.imag(0,2048),"FFT Signal")
  		val Xa = abs(X)
  		//imagesc(Xa,2048)
  		plot(Xa(0::2048),"Abs FFT Signal") //
  		//
  		*/
  		/*var i = 0
			while(i<10)
			{
				//
				//val rlv=getreal(dvcval)
				val outr = tst(0::L)
				val outi = tst(L::2*L)
				val out = outr*outi
  			//val ivl=getimag(dvcval)
  			//dvcval(real)=DenseVector.randf(L)
  			//dvcval(imag)=DenseVector.randf(L)
  			i = i + 1
			}*/
  		//rlv(0::10).pprint
  		//ivl(0::10).pprint
  		//val outvec = DenseVector[Double](2*16777216,true)
  		//val dt = idata(0)
  		//dt(0::10).pprint
  		//idata2(0::10).pprint
  		/*println("HT Data...");
  		val t1=tic
  		val datftout = StreamProcessor(idata,hilbert,fft) // Hilbert transform and FFT Input Data
  		toc(t1)
  		//val datftout = fft(idata(0)) // Hilbert transform and FFT Input Data
  		//
  		println("FFT Reference Data...");
  		val refftout = cxvfft(n3refsig) // FFT Ref. Signal
  		println("FFT Data...");
  		val multout		 = CStreamProcessor(datftout,refftout,*)
  		println("Multiply Data with Ref. and iFFT...");
  		datftout		 = CStreamProcessor(multout,ifft)
  		*/
  		//val outp		 = CStreamProcessor(multout,ifft)
  		//
  		//outputd(0) = datftout(0)
			/*var i = 0
			var k = 0
  		while(i < outp.length)
  		{
  			val tst = abs(outp(i))//datftout.im
  			var j = 0
  			while(j<4096)
  			{
  				outvec(k) = tst(j)
  				k = k + 1
  				j = j + 1
  			}
  			i = i + 1
  		}
  		println(k)
  		writeVector(outvec,"optisdrnetradout.txt")
  		*/
  		//tst(0::10).pprint
  		//println("Plotting Data...");
  		//imagesc(datftout,"Number of Pulses","Range ?","Fast-Time vs Slow-Time NetRAD Plot")
  	}
}