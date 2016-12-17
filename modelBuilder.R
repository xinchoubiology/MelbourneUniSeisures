library(mxnet)

resNetUnit <- function(input, kernel, stride, depth, stepUnit=FALSE, pad=c(0,0), name =""){
  eps <- 2e-5
  bn_mom <- 0.9
  fix_gamma <- FALSE
  
  #Res unit
  BN1 <- mx.symbol.BatchNorm(input, fix.gamma = fix_gamma, name = paste(name,"BN1",sep=""))
  relu1 <- mx.symbol.LeakyReLU(data=BN1, act.type='elu')
  #relu1 <- mx.symbol.Activation(BN1, act_type='relu', name = paste(name,"relu1",sep=""))
  conv1 <- mx.symbol.Convolution(relu1, kernel=kernel, stride = stride, num_filter=depth, 
                                 no.bias=TRUE, pad=pad, name = paste(name,"conv1",sep=""))
  
  BN2 <- mx.symbol.BatchNorm(conv1, fix.gamma = fix_gamma, name = paste(name,"BN2",sep=""))
  relu2 <- mx.symbol.LeakyReLU(data=BN2,act.type='elu')
# relu2 <- mx.symbol.Activation(BN2, act_type='relu', name = paste(name,"relu2",sep=""))
  conv2 <- mx.symbol.Convolution(relu2, kernel=kernel, stride=c(1,1), num_filter=depth, 
                                 no.bias=TRUE, pad=pad, name = paste(name,"conv2",sep=""))
  
  #skip transform if needed
  if(stepUnit) input <- mx.symbol.Convolution(input, kernel=kernel, stride = stride, num_filter=depth, 
                                              no.bias=TRUE, pad=pad, name = paste(name,"step",sep=""))
  
  input + conv2
  
}

resNetBlock <- function(input, kernel, stride, depth, nUnits, stepUnit=TRUE, pad=c(0,0), name =""){
  
  for(i in 1:nUnits){
    if(i==1){
      input <- resNetUnit(input, kernel, stride, depth, stepUnit=stepUnit, pad=pad, name = paste(name,"unit",i,sep=""))
    } else {
      input <- resNetUnit(input, kernel, stride=c(1,1), depth, pad=pad, name = paste(name,"unit",i,sep=""))
    }
  }
  input
}

resNetNet <- function(nBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride, name=""){
  
  data <- mx.symbol.Variable("data")
  
  for(i in 1:nBlocks){
    data <- resNetBlock(data, c(kernel[i],1), c(stride[i],1), depth[i], stepUnit=TRUE, nUnits[i], pad=c(pad[i],0), 
                        name = paste(name,"block",i,sep=""))
    if(pool[i]){
      data <- mx.symbol.Pooling(data, kernel=c(poolKernel[i],1), stride=c(poolStride[i],1), pool_type = 'max', 
                                name = paste(name,"pool",i,sep=""))
    }
  }
  data <- mx.symbol.Flatten(data)
  data <- mx.symbol.Dropout(data)
  data <- mx.symbol.BatchNorm(data, fix.gamma=FALSE)
  data <- mx.symbol.FullyConnected(data, no.bias=TRUE, num.hidden=1L)
  logreg <- mx.symbol.LogisticRegressionOutput(data)
  return(logreg)
}
