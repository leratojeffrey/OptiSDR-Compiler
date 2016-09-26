import uct.rrsg.dsl.optisdr._ //sdr packages
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object FIRExampleRunner extends OptiSDRApplicationRunner with FIRExample
trait FIRExample extends OptiSDRApplication
{
	def main() = 
	{
		//val indata = DenseVector.rand(1024*50000) // Generate 1000 random data
		/*val tm2 = 0.0::>0.00001<::0.01
		val tm = 0.0::>1.0<::2048
		val l = tm.length*1.0
		val f = (-1.0*l/2.0)::>1.0<::(l/2.0)
    */
    //val incos = Cos(2*pi*1000*tm)
    
    //println(" "+tc1)
		val Fs = 10e4f;
		val Fs2 = 10e4f;
		val Fc1 = 1000f;
    val Fc2 = 2000f;
    val Fc3 = 3000f;
    val Fc4 = 4000f;
    val Fc5 = 5000f;
    val Fs3 = 12000f
    val Fs4 = 85000.0f;
    val fo = 3.5e3f; 
    // Carrier frequency in Hz
    //
    //
    val tc1 = tic
		val tm3 = 0.0f::>1.0f/Fs4<::(0.1f-1.0f/Fs4)
		val t = 0.0f::>1.0f/Fs<::(0.1f-1.0f/Fs)
		val t2 = 0.0f::>1.0f/Fs2<::(0.1f-1.0f/Fs2)
		val l = t.length*1.0f
		//val l2 = t.length*1.0
		val m = Sin(2.0f*pi*500.0f*t) + 0.5f*Sin(2.0f*pi*600.0f*t) + 2.0f*Sin(2.0f*pi*700f*t)
		
		val f = (-1.0f*l/2.0f)::>1.0f<::(l/2.0f)
    //
    //val insin = Sin(2*pi*1000*tm)
    //insin = 2*insin
    //println(insin(0))
		//val incos = Cos(2*pi*1000*tm2)
		//println(f.length)
		//incos.pprint
    //
		val dsbsig = m*Cos(2.0f*pi*fo*t);

		val m2 = Sin(2.0f*pi*800.0f*t2) + 0.5f*Sin(2.0f*pi*900.0f*t2) + 2.0f*Sin(2.0f*pi*1000.0f*t2)
		//
    //    
    val x1 = Cos(2.0f*pi*1000.0f*tm3)
    //
		val y1 = pmodulate(x1,Fc1,Fs3,AMDSB_SC)
		val y2 = pmodulate(x1,Fc2,Fs3,AMDSB_SC)
		val y3 = pmodulate(x1,Fc3,Fs3,AMDSSB) 
		val y4 = pmodulate(x1,Fc4,Fs3,AMDSSB) 
		val y5 = pmodulate(x1,Fc5,Fs3,AMDSB_SC)
		//val y6 = pmodulate(m,Fc1,Fs3,AMDSSB)

		val y = y1 + y2 + y3 + y4 + y5;
		//
    /*val perplot = periodogram(x1,f,"Sinusoid Signal")
		perplot.add(dsbsig,"DSB signal")
		perplot.add(m2,"Sinusoid Signal 2")
		perplot.add(y,"Sinusoid Signal 3")*/
		//perplot.add(y6,"Modulated Signal y")
		toc(tc1)
		//
		/*var working = true;
		var i = 1
		//val k = 0.0
    //val tm2 = 4.0::>1.0<::1024+4
    while(working)
    {
    	i = i + 1
    	
    	if(i%10 == 0)
    	{
    		val k = i*1
    		//println(k)		
    		//val tm2 = 0.0::>1.0<::1024
    		tm<<=(0.01::>0.00001<::0.02)
    		val inc = Sin(2*pi*1000*tm)
    		perplot.refresh(zeropad(inc(k::k+999),2048),"Sinusoid Signal")
    		//
	    	if(i == 2000)
	    	{
	    		val insin2 = 0.75*insin;
	    		perplot.add(insin2,"sin2")
	    	//	working = false;
	    	//	i=0
	    	}
    	}
    }*/
		//val inp = SineWave(1,50,1, 0, 0.2)
		//println("Sinusoid Length::"+inp.length)
		//plot(insin)
		//
		/*val coeffs = DenseVector(-0.0448093,0.0322875,0.0181163,0.0087615,0.0056797,
        0.0086685,0.0148049,0.0187190,0.0151019,0.0027594,-0.0132676,-0.0232561, -0.0187804,0.0006382,0.0250536,
   0.0387214,0.0299817,0.0002609,-0.0345546,-0.0525282,-0.0395620,0.0000246,0.0440998,0.0651867,0.0479110,
   0.0000135,-0.0508558,-0.0736313,-0.0529380,-0.0000709,0.0540186,0.0766746,0.0540186,-0.0000709,-0.0529380,
  -0.0736313,-0.0508558,0.0000135,0.0479110,0.0651867,0.0440998,0.0000246,-0.0395620,-0.0525282,-0.0345546,
   0.0002609,0.0299817,0.0387214,0.0250536,0.0006382,-0.0187804,-0.0232561,-0.0132676,0.0027594,0.0151019,
   0.0187190,0.0148049,0.0086685,0.0056797,0.0087615,0.0181163,0.0322875,-0.0448093)*/
   //
   //* val kcoeffs = kaiser(127)
   //*	val tst = kaisercoeffs(127,kcoeffs,0.2,0.5,HighPass)
   //*	val tst1 = kaisercoeffs(127,kcoeffs,0.2,0.5,LowPass)
	//*	val in = StreamIn(insin,1024) // create samples for parallel fir ops.
		//println("Length::"+in.numRows)
		//*val firout = fir(in,kcoeffs)
		/*val firout = DenseMatrix[Double](in.numRows,8192)
		
		for(i <- 0 until in.numRows)
		{
			val firout = fir(vec(in(i)),kcoeffs,127) // IR Defined in Listing 2
		}
		*/
		//*plot(tst)
		//*plot(tst1)
		//periodogram(tst1)
		//plot(tm,incos,"Signal","Time","")
		//plot(vec(firout(0)))
		//plot(tm,insin,"Signal","Time","Sinusoid Test")
		//odata.pprint // Print output data
	}
}