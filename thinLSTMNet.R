setwd("~/MelbourneUniSeisures")

require(mxnet)

source("./modelData.R")
source('./lstmModelBuilder.R')
source('./customMXnet.R')

n.gpu <- 1
devices <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})

trainIter <- mx.io.CSVIter(data.csv = "./inputfiles/trainData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                           label.csv = "./inputfiles/trainLabels.lst", label.shape = 1,
                           batch.size = 128*n.gpu,shuffle = TRUE,
                           preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1)) 
evalIter <- mx.io.CSVIter(data.csv = "./inputfiles/evalData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                           label.csv = "./inputfiles/evalLabels.lst", label.shape = 1,
                           batch.size = 128*n.gpu,
                           preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1)) 
testIter <- mx.io.CSVIter(data.csv = "./inputfiles/testData.csv", data.shape = c(nSteps,nChannels,dataDepth),
                           batch.size = 128*n.gpu,
                           preprocess.threads=(n.gpu+1), prefetch.buffer = (n.gpu+1))

#thin LSTM Net parameters
nThinBlocks <- 1
nFatBlocks <- 1
depth <- c(4,32)
nUnits <- c(1, 1)
dim <- c(nSteps,nChannels)
num.lstm.layer <- 1
num.hidden <- c(8, 16)
tail <- 16
name <- "thinNet"

thinLSTMNet <- lstmThinNet(nThinBlocks, nFatBlocks, depth, nUnits, dim,
                           num.lstm.layer, num.hidden, tail, dropout=0.2) 

logger <- mx.metric.logger$new()
thinLSTMModel <- custom.mx.model.FeedForward.create(thinLSTMNet, trainIter, ctx = devices,#kvstore = "device", 
                                            num.round = 1000, optimizer = 'adam',
                                            initializer = mx.init.Xavier("gaussian", "in", 2.0),
                                            eval.data = evalIter,
                                            eval.metric = mx.metric.logloss,
                                            batch.end.callback = mx.callback.log.train.metric(10,logger),
                                            epoch.end.callback= mx.callback.early.stop(5,logger, maximize=FALSE, prefix='./models/thinLSTM001'),
                                            num.lstm.layer=num.lstm.layer, num.hidden=num.hidden,
                                            wd = 0.00001, clip_gradient = 1)

trainPreds <- as.vector(predict(thinNetModel,trainIter, mx.gpu()))
testPreds <- as.vector(custom.predict.MXFeedForwardModel(thinLSTMModel,testIter, mx.cpu(), 
                                                         num.lstm.layer=num.lstm.layer, num.hidden=num.hidden))
