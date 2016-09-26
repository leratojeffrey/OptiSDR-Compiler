package ppl.delite.runtime.graph.ops

import ppl.delite.runtime.graph.DeliteTaskGraph
import ppl.delite.runtime.graph.targets.Targets

/**
 *
 */

class OP_While(val id: String,
               val predicateGraph: DeliteTaskGraph, val predicateValue: String,
               val bodyGraph: DeliteTaskGraph, val bodyValue: String,
               outputSymbol: String = null)
  extends OP_Control {

  def nestedGraphs = Seq(predicateGraph, bodyGraph)

  private[graph] var outputTypesMap = if (outputSymbol == null) Targets.unitTypes(id) else Targets.unitTypes(outputSymbol)

  /**
   * creates a While chunk for each requested resource and destroys the original
   */
  def makeChunks(indices: Seq[Int], graph: DeliteTaskGraph) = {
    val lastOps = if (bodyValue == "") for (r <- bodyGraph.schedule if(!r.isEmpty)) yield r.peekLast else null
    val chunks =
      for (idx <- indices) yield {
        val outputSym = if (idx == indices(0)) this.id else null
        val r = new OP_While(id+"_"+idx, predicateGraph, predicateValue, bodyGraph, bodyValue, outputSym)
        r.dependencies = dependencies
        r.inputList = inputList
        r.mutableInputs = mutableInputs
        r.filterMutableInputs(idx)
        r.consumers = consumers
        r.antiDeps = antiDeps
        r.filterAntiDeps(idx)
        
        for (tgt <- Targets.GPU) r.setGPUMetadata(tgt, getGPUMetadata(tgt))
        for (dep <- getDependencies) dep.addConsumer(r)
        for (c <- getConsumers) c.addDependency(r)

        //add special consumer ops
        if (predicateValue == "") predicateGraph.schedule(idx).add(new GetterOp(id+"p_"+idx, idx, Seq(predicateGraph.result._1), Seq(predicateGraph.result))) //get predicate result on all chunks
        if (bodyValue == "") bodyGraph.schedule(idx).add(new GetterOp(id+"b_"+idx, idx, lastOps, Seq())) //barrier end of body so predicate can be reevaluated
        r
      }

    graph.replaceOp(this, chunks(0))
    for (i <- 0 until indices.length)
      refineInputDeps(chunks(i), graph, indices(i))
    chunks
  }
}
