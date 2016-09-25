library(mxnet)

resNetUnit <- function(input, kernel, stride, depth, stepUnit=FALSE, pad=c(0,0)){
  
  #Res unit
  BN1 <- mx.symbol.BatchNorm(input, name="BN1")
  conv1 <- mx.symbol.Convolution(BN1, kernel=kernel, num_filter=depth, no.bias=TRUE, pad=pad)
  relu1 <- mx.symbol.Activation(conv1, act_type='relu')
  BN2 <- mx.symbol.BatchNorm(relu1, name="BN2")
  conv2 <- mx.symbol.Convolution(BN2, kernel=kernel, num_filter=depth, no.bias=TRUE)
  relu2 <- mx.symbol.Activation(conv2, act_type='relu')
  
  #skip transform if needed
  if(stepUnit) input <- mx.symbol.Convolution(input, kernel=kernel, num_filter=depth, no.bias=TRUE)
  
  input + relu2
  
}

resNetBlock <- function(input, kernel, stride, depth, stepUnit=TRUE, nUnits, pad=c(0,0)){
  
  for(i in 1:nUnits){
    if(i==1){
      input <- resNetUnit(input, kernel, stride, depth, stepUnit=stepUnit, pad=pad)
    } else {
      input <- resNetUnit(input, kernel, stride, depth)
    }
  }
  input
}


input <- mx.symbol.Variable("input")

block1 <- resNetBlock(input, kernel=c(9,1), stride=c(2,1), depth=64, nUnits=3, pad=c(4,0))
pool1 <- mx.symbol.Pooling(block1, kernel=c(2,1), stride=c(2,1), pool_type = 'max')

block2 <- resNetBlock(pool1, kernel=c(9,1), stride=c(2,1), depth=32, nUnits=3)
pool2 <- mx.symbol.Pooling(block2, kernel=c(2,1), stride=c(2,1), pool_type = 'max')

block3 <- resNetBlock(pool2, kernel=c(9,1), stride=c(2,1), depth=16, nUnits=3)
pool3 <- mx.symbol.Pooling(block3, kernel=c(2,1), stride=c(2,1), pool_type = 'max')

flattenPool3 <- mx.symbol.Flatten(pool3)
fc1 <- mx.symbol.FullyConnected(flattenPool3, num_hidden=1, name="fc1")
logReg <- mx.symbol.LogisticRegressionOutput(fc1, name="logReg")
