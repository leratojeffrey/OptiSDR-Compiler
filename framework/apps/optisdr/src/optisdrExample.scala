import uct.rrsg.dsl.optisdr._
import scala.virtualization.lms.common.Record
import scala.reflect.SourceContext

object SumVectorExampleRunner extends OptiSDRApplicationRunner with SumVectorExample
trait SumVectorExample extends OptiSDRApplication { 
   def main() = 
   {  
   	val v1 = DenseVector.randf(310)
   	//val v2 = Vector.rand(31)
   	val VCOEFFS = DenseVector(-0.0448093,0.0322875,0.0181163,0.0087615,0.0056797,
        0.0086685,0.0148049,0.0187190,0.0151019,0.0027594,-0.0132676,-0.0232561, -0.0187804,0.0006382,0.0250536,
   0.0387214,0.0299817,0.0002609,-0.0345546,-0.0525282,-0.0395620,0.0000246,0.0440998,0.0651867,0.0479110,
   0.0000135,-0.0508558,-0.0736313,-0.0529380,-0.0000709,0.0540186,0.0766746,0.0540186,-0.0000709,-0.0529380,
  -0.0736313,-0.0508558,0.0000135,0.0479110,0.0651867,0.0440998,0.0000246,-0.0395620,-0.0525282,-0.0345546,
   0.0002609,0.0299817,0.0387214,0.0250536,0.0006382,-0.0187804,-0.0232561,-0.0132676,0.0027594,0.0151019,
   0.0187190,0.0148049,0.0086685,0.0056797,0.0087615,0.0181163,0.0322875,-0.0448093)
   
    	val x = sum(0,31) {i => v1(i) * VCOEFFS(i).toFloat}
    	println(x)
   }
}

object ForLoopExampleRunner extends OptiSDRApplicationRunner with ForLoopExample
trait ForLoopExample extends OptiSDRApplication { 
   def main() = 
   {  
   	val v1 = DenseVector.zeros(1000).map(e => DenseVector.randf(32))
   	val VCOEFFS = DenseVector(-0.0448093,0.0322875,0.0181163,0.0087615,0.0056797,
        0.0086685,0.0148049,0.0187190,0.0151019,0.0027594,-0.0132676,-0.0232561, -0.0187804,0.0006382,0.0250536,
   0.0387214,0.0299817,0.0002609,-0.0345546,-0.0525282,-0.0395620,0.0000246,0.0440998,0.0651867,0.0479110,
   0.0000135,-0.0508558,-0.0736313,-0.0529380,-0.0000709,0.0540186,0.0766746,0.0540186,-0.0000709,-0.0529380,
  -0.0736313,-0.0508558,0.0000135,0.0479110,0.0651867,0.0440998,0.0000246,-0.0395620,-0.0525282,-0.0345546,
   0.0002609,0.0299817,0.0387214,0.0250536,0.0006382,-0.0187804,-0.0232561,-0.0132676,0.0027594,0.0151019,
   0.0187190,0.0148049,0.0086685,0.0056797,0.0087615,0.0181163,0.0322875,-0.0448093)

   	val x=DenseVector[Float](1000, true)
	var idx = 0
	var id = 0
   	for(v <- v1)
	{
    		x(idx) = sum(0,32){i => v(i) * VCOEFFS(i).toFloat}
		idx+=1		 
		//println(x(id)) // This line Makes it serially implemented, i.e. No CUDA Kernels loaded
    	}

	x(0::5).pprint //Parallel (i.e. Executed in CUDA) but limited data transfers, device to host. 1st values only displyed
   }
}

object WhileLoopExampleRunner extends OptiSDRApplicationRunner with WhileLoopExample
trait WhileLoopExample extends OptiSDRApplication
{ 
	def main() = 
	{
		val v  = DenseVector.randf(1000)
		val v1 = DenseVector[DenseVector[Float]](1000, true)
		// Parallel loop for partitioning input vector v
		var i = 0 
		val K = 1000 - 63 + 1
		while(i<K){
			v1(i) = v(i::63+i) // 0 -64, 1-65, 2-66, etc.
			i+=1
		}
		v1(0).pprint // Print values
		v1(1).pprint // Print values
		v1(2).pprint // Print values
	}
}