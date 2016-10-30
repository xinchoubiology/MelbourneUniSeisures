library(mxnet)

resNetUnit <- function(input, kernel, stride, depth, stepUnit=FALSE, pad=c(0,0), name =""){
  eps <- 2e-5
  bn_mom <- 0.9
  fix_gamma <- FALSE
  
  #Res unit
  BN1 <- mx.symbol.BatchNorm(input, fix.gamma = fix_gamma, name = paste(name,"BN1",sep=""))
  relu1 <- mx.symbol.Activation(BN1, act_type='relu', name = paste(name,"relu1",sep=""))
  conv1 <- mx.symbol.Convolution(relu1, kernel=kernel, stride = stride, num_filter=depth, 
                                 no.bias=TRUE, pad=pad, name = paste(name,"conv1",sep=""))
  BN2 <- mx.symbol.BatchNorm(conv1, fix.gamma = fix_gamma, name = paste(name,"BN2",sep=""))
  relu2 <- mx.symbol.Activation(BN2, act_type='relu', name = paste(name,"relu2",sep=""))
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

resNetNet <- function(nThinBlocks, nFatBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride,
                      cropKernel, cropStride, dim, name=""){
  
  data <- mx.symbol.Variable("data")
  data <- mx.symbol.BatchNorm(data=data, fix_gamma=TRUE, eps=2e-5, momentum=0.9, name='bn_data')
  
  factor <- 1
  for(i in 1:nThinBlocks){
    factor <- factor / stride[[i]][1]
    data <- resNetBlock(data, kernel[[i]], stride[[i]], depth[i], stepUnit=TRUE, nUnits[i], pad=pad[[i]], 
                         name = paste(name,"block",i,sep=""))
    if(pool[i]){
      factor <- factor / poolStride[[i]][1]
      data <- mx.symbol.Pooling(data, kernel=poolKernel[[i]], stride=poolStride[[i]], pool_type = 'max', 
                                           name = paste(name,"pool",i,sep=""))
    }
  }
  shape <- c(dim[1]*factor,1,depth[nThinBlocks]*dim[2],-1)
  data <- mx.symbol.Reshape(data=data, shape=shape)
  
  for(i in ((1:nFatBlocks)+nThinBlocks)){
    data <- resNetBlock(data, kernel[[i]], stride[[i]], depth[i], stepUnit=TRUE, nUnits[i], pad=pad[[i]], 
                        name = paste(name,"block",i,sep=""))
    if(pool[i]) data <- mx.symbol.Pooling(data, kernel=poolKernel[[i]], stride=poolStride[[i]], pool_type = 'max', 
                                          name = paste(name,"pool",i,sep=""))
  }
  
  
  data <- mx.symbol.BatchNorm(data=data, fix_gamma=TRUE, eps=2e-5, momentum=0.9, name='bn_out')
  data <- mx.symbol.Dropout(data=data, p=0.2)
  data <- mx.symbol.Convolution(data, no.bias=TRUE, kernel=cropKernel, stride =cropStride,
                                num_filter=1, name=paste(name,"cropConv",sep=""))
  data <- mx.symbol.Pooling(data, kernel=c(8,1), global.pool=TRUE, pool.type='avg')
  data <- mx.symbol.Flatten(data)
  logreg <- mx.symbol.LogisticRegressionOutput(data)
  return(logreg)
}
