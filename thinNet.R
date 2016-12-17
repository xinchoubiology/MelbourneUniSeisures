setwd("~/MelbourneUniSeisures")

require(mxnet)
require(data.table)

source("./modelData.R")
source('./modelBuilder.R')
source('./customMXnet.R')

n.gpu <- 4
devices <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})



nCrops <- 8
cropSize <- 64
cropStep <- (nSteps - cropSize) / nCrops

trainDataDT <- fread("./inputfiles/trainData.csv",header=F)
nData <- nrow(trainDataDT)
trainDataDT <- t(as.matrix(trainDataDT))
trainDataDT <- array(trainDataDT, dim=c(nSteps,1,dataDepth,nData))
trainDataArr <- array(NA, dim=c(cropSize,1,dataDepth,nData*nCrops))

for(j in 1:nCrops){
  cat(j,'\n')
  trainDataArr[,,,(1:nData)+nData*(j-1)] <- trainDataDT[(1:cropSize)+(j-1)*cropStep,,,]
}
rm(trainDataDT)

eval.data <- list()
evalDataDT <- fread("./inputfiles/evalData.csv",header=F)
nData <- nrow(evalDataDT)
evalDataDT <- t(as.matrix(evalDataDT))
evalDataDT <- array(evalDataDT, dim=c(nSteps,1,dataDepth,nData))
eval.data$data <- array(NA, dim=c(cropSize,1,dataDepth,nData*nCrops))

for(j in 1:nCrops){
  cat(j,'\n')
  eval.data$data[,,,(1:nData)+nData*(j-1)] <- evalDataDT[(1:cropSize)+(j-1)*cropStep,,,]
}
rm(evalDataDT)

trainLabels <- fread('./inputfiles/trainLabels.lst', header=F)
trainLabels <- rep(trainLabels$V1,nCrops)
evalLabels <- fread('./inputfiles/evalLabels.lst', header=F)
eval.data$labels <- rep(evalLabels$V1,nCrops)

trainIter <- mx.io.CSVIter(data.csv = "./inputfiles/trainData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                           label.csv = "./inputfiles/trainLabels.lst", label.shape = 1,
                           batch.size = 256*n.gpu,shuffle = TRUE,
                           preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1)) 
evalIter <- mx.io.CSVIter(data.csv = "./inputfiles/evalData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                          label.csv = "./inputfiles/evalLabels.lst", label.shape = 1,
                          batch.size = 256*n.gpu,
                          preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1)) 
testIter <- mx.io.CSVIter(data.csv = "./inputfiles/testData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                          batch.size = 256*n.gpu,
                          preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1))

#thin Net parameters
nBlocks <- 7
kernel <- list(c(7,1), c(7,1), c(7,1), c(7,1), c(7,1), c(7,1), c(7,1))
stride <- list(c(1,1), c(1,1), c(1,1), c(1,1), c(1,1), c(1,1), c(1,1))
depth <- c(256, 128, 64, 32, 16, 8, 1)
nUnits <- c(2, 2, 2, 2, 2, 2, 2)
pad <- list(c(3,0), c(3,0), c(3,0), c(3,0), c(3,0), c(3,0), c(3,0))
pool <- c(F, F, F, F, F, F, F)
poolKernel <- list(c(2,1), c(2,1), c(2,1), c(2,1), c(2,1), c(2,1), c(2,1))
poolStride <- poolKernel
name <- "thinNet"

thinResNet <- resNetNet(nBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride, name)

batchLogger <- mx.metric.logger$new()
epochLogger <- mx.metric.logger$new()

thinNetModel <- mx.model.FeedForward.create(thinResNet, trainDataArr, trainLabels, ctx = devices,#kvstore = "device", 
                                            array.batch.size = 256*n.gpu,
                                            num.round = 2000, #optimizer = 'adam',
                                            initializer = mx.init.Xavier("gaussian", "in", 2.0),
                                            eval.data = eval.data,
                                            eval.metric = mx.metric.logloss,
                                            batch.end.callback = mx.callback.log.train.metric(10,batchLogger),
                                            epoch.end.callback= mx.callback.early.stop(20,epochLogger, maximize=FALSE, prefix='./models/thinNet001'),
                                            learning.rate = 1e-2, momentum=0.9, 
                                            wd = 0.0005,
                                            clip_gradient = 1)

trainPreds <- as.vector(predict(thinNetModel,trainIter, mx.gpu()))
testPreds <- as.vector(predict(thinNetModel,testIter, mx.gpu()))
