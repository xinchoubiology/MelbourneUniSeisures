setwd("~/MelbourneUniSeisures")

library(data.table)
library(mxnet)
library(doParallel)

source('./modelBuilder.R')
source('./matFileIter.R')
source('./customMXnet.R')

sampFreq <- 400
resampFreq <- 100
length.out <- 240000 * resampFreq / sampFreq
testDirs <- paste("test_",1:3,"_new",sep="")
cropLength <- 16384
cropStart <- length.out - cropLength - 1
noise <- 2
n.gpu <- 1
batchSize <- 32 * n.gpu
nCores <- detectCores()
queLength <- 2

key <- fread("./inputfiles/trainDataKey.csv")
evalFiles <- key[isEval==1]$image
evalLabels <- key[isEval==1]$class
trainFiles <- key[isEval==0]$image
trainLabels <- key[isEval==0]$class
testFiles <- list.files(paste("./download/",testDirs,sep=''), full.names = TRUE)
testLabels <- rep(0,length(testFiles))
                        
if(n.gpu > 0){
devices <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
}else{
  devices <- list(mx.cpu())
}


evalIter <- R.io.matFileIter(evalFiles, evalLabels, cropLength, TRUE, cropStart, sampFreq, resampFreq, noise,
                                         batchSize, nCores, queLength)
trainIter <- R.io.matFileIter(trainFiles, trainLabels, cropLength, TRUE, cropStart, sampFreq, resampFreq, noise,
                              batchSize, nCores, queLength)

#thin Net parameters
nBlocks <- 4
kernel <- c(7,5,5,5)
stride <- c(4,2,2,2)
depth <- c(64,32,32,16)
nUnits <- c(2,2,2,2)
pad <- (kernel-1) / 2
pool <- c(F,F,F,F,F,F)
poolKernel <- rep(2,4)
poolStride <- poolKernel

thinResNet <- resNetNet(nBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride)

logger <- mx.metric.logger$new()

thinNetModel <- mx.model.FeedForward.create(thinResNet, trainIter, ctx = devices,#kvstore = "device", 
                                            num.round = 5, optimizer = 'adam',
                                            initializer = mx.init.Xavier("gaussian", "in", 2.0),
                                            #eval.data = evalIter,
                                            eval.metric = mx.metric.logloss,
                                            batch.end.callback = mx.callback.log.train.metric(10,logger),
                                            #epoch.end.callback= mx.callback.early.stop(5,logger, maximize=FALSE, prefix='./models/thinNet001'),
                                            #learning.rate = 1e-2, momentum=0.9, 
                                            wd = 0.00001,
                                            clip_gradient = 2)

