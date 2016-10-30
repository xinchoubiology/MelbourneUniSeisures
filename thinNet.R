setwd("~/MelbourneUniSeisures")

require(mxnet)

source("./modelData.R")
source('./modelBuilder.R')

# Custom evaluation metric on CRPS.
mx.metric.auc<- mx.metric.custom("auc", function(label, pred) {
  return(Metrics::auc(label, pred))
})
mx.metric.logloss<- mx.metric.custom("logloss", function(label, pred) {
  return(Metrics::logLoss(label, pred))
})

n.gpu <- 4
devices <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})

#thin Net parameters
nThinBlocks <- 2
nFatBlocks <- 5
kernel <- list(c(9,1), c(9,1), c(3,1), c(3,1), c(3,1), c(3,1), c(3,1))
stride <- list(c(1,1), c(1,1), c(1,1), c(1,1), c(1,1), c(1,1), c(1,1))
depth <- c(16, 8, 128, 64, 32, 16, 8)
nUnits <- c(2, 2, 2, 2, 2, 2, 2)
pad <- list(c(4,0), c(4,0), c(1,0), c(1,0), c(1,0), c(1,0), c(1,0))
pool <- c(F, T, T, T, F, F, F)
poolKernel <- list(c(2,1), c(2,1), c(2,1), c(2,1), c(2,1), c(2,1), c(2,1))
poolStride <- poolKernel
cropKernel <- c(32,1)
cropStride <- c(1,1)
dim <- c(nSteps,nChannels)
name <- "thinNet"

thinResNet <- resNetNet(nThinBlocks, nFatBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride, 
                        cropKernel, cropStride, dim, name)

trainIter <- mx.io.CSVIter(data.csv = "./inputfiles/trainData.csv", data.shape = c(nSteps,nChannels,nMFCC),
                           label.csv = "./inputfiles/trainLabels.csv", label.shape = 1,
                           batch.size = 128*n.gpu, shuffle=TRUE,
                           preprocess.threads=4, prefetch.buffer = 4) 
evalIter <- mx.io.CSVIter(data.csv = "./inputfiles/evalData.csv", data.shape = c(nSteps,nChannels,nMFCC),
                          label.csv = "./inputfiles/evalLabels.csv", label.shape = 1,
                          batch.size = 128) 

log <- mx.metric.logger$new()

thinNetModel <- mx.model.FeedForward.create(thinResNet, trainIter, ctx = devices,#kvstore = "device", 
                                            num.round = 2, optimizer = 'adam',
                                            initializer = mx.init.Xavier("gaussian", "in", 2.0),
                                            #eval.data = evalIter,
                                            eval.metric = mx.metric.logloss,
                                            batch.end.callback = mx.callback.log.train.metric(2,logger = log),
                                            epoch.end.callback = mx.callback.save.checkpoint("./models/thinNet003",10),
                                            #learning.rate = 1e-3, momentum=0.9, 
                                            wd = 0.0001,
                                            clip_gradient = 2)

preds <- as.vector(predict(thinNetModel,evalIter, mx.gpu()))