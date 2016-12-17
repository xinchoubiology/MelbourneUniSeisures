require(mxnet)

source('./modelBuilder.R')

# lstm cell symbol
lstm <- mxnet:::lstm

# unrolled lstm network
lstmThinNet <- function(nThinBlocks, nFatBlocks, depth, nUnits, dim,
                        num.lstm.layer, num.hidden, tail, dropout=0.2, name='thinLstm') {
  
  data <- mx.symbol.Variable("data")
  if(nThinBlocks > 0){
    for(i in 1:nThinBlocks){
      data <- resNetBlock(data, c(1,1), c(1,1), depth[i], stepUnit=TRUE, nUnits[i], pad=c(0,0), 
                          name = paste(name,"block",i,sep=""))
    }
  }
  
  shape <- c(dim[1],1,depth[nThinBlocks]*dim[2],-1)
  data <- mx.symbol.Reshape(data=data, shape=shape)
  
  if(nFatBlocks > 0){
    for(i in ((1:nFatBlocks)+nThinBlocks)){
      data <- resNetBlock(data, c(1,1), c(1,1), depth[i], stepUnit=TRUE, nUnits[i], pad=c(0,0), 
                          name = paste(name,"block",i,sep=""))
    }
  }
  param.cells <- lapply(1:num.lstm.layer, function(i) {
    cell <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                 i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                 h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                 h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
    return (cell)
  })
  last.states <- lapply(1:num.lstm.layer, function(i) {
    state <- list(c=mx.symbol.BlockGrad(mx.symbol.Variable(paste0("l", i, ".init.c"))),
                  h=mx.symbol.BlockGrad(mx.symbol.Variable(paste0("l", i, ".init.h"))))
    return (state)
  })
  
  data <- mx.symbol.Flatten(data)
  wordvec <- mx.symbol.SliceChannel(data=data, axis=1, num_outputs=dim[1], squeeze_axis=TRUE)
  
  last.hidden <- list()
  for (seqidx in 1:dim[1]) {
    hidden <- wordvec[[seqidx]]
    # stack lstm
    for (i in 1:num.lstm.layer) {
      dp <- ifelse(i==1, 0, dropout)
      next.state <- lstm(num.hidden[i], indata=hidden,
                         prev.state=last.states[[i]],
                         param=param.cells[[i]],
                         seqidx=seqidx, layeridx=i,
                         dropout=dp)
      hidden <- next.state$h
      last.states[[i]] <- next.state
    }
    # decoder
    if (seqidx > (dim[1]-tail) ){
      if (dropout > 0)  hidden <- mx.symbol.Dropout(data=hidden, p=dropout)
      last.hidden <- c(last.hidden, hidden)
    }
  }
  concat <- mx.symbol.Concat(last.hidden, tail) 
  shape <- c(tail,1,num.hidden[num.lstm.layer],-1)
  reShape<- mx.symbol.Reshape(concat, shape=shape)
  BN <- mx.symbol.BatchNorm(reShape)
  conv <- mx.symbol.Convolution(BN, kernel=c(1,1), stride = c(1,1), num_filter=1, 
                                no.bias=TRUE, pad=c(0,0))
  #  concat <- mx.symbol.Flatten(conv)
  globalPool <- mx.symbol.Pooling(conv, kernel=c(tail,1), global.pool=TRUE, pool.type='avg')
  flatten <- mx.symbol.Flatten(globalPool)
  logreg <- mx.symbol.LogisticRegressionOutput(flatten)
  return (logreg)
}

