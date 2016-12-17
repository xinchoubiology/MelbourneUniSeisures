setwd("~/MelbourneUniSeisures")

library(data.table)
library(doParallel)
library(stringr)
library(mxnet)

source('./modelBuilder.R')
source('./customMXnet.R')


getSize <- function(files){
  require(doParallel)
  require(R.matlab)
  
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  
  size <- foreach(file = files, .combine='c') %dopar% {
    data <- readMat(file)
    data <- data$dataStruct[[1]]
    max(abs(data))
  }
  stopImplicitCluster()
  max(size)
}

convertToImg <- function(file, sampFreq, filtFreq, outFreq, size, outChannels, savePath = './'){
  require(seewave)
  require(R.matlab)
  require(fftw)
  require(png)
  require(stringr)
  
  outName <- paste(word(file,-1,sep='/'),'.png',sep='')
  if(outName %in% dir(savePath)) return(NULL)
  outName <- paste(savePath,outName,sep='')
  
  data <- readMat(file)
  data <- data$dataStruct[[1]]
  
  data <- apply(data, 2, ffilter, f=sampFreq, to=filtFreq, fftw=TRUE)
  data <- apply(data, 2, resamp, f=sampFreq, g=outFreq)
  data <- (data + size) / 2 / size 
  data[data < 0] <- 0
  data[data > 1] <- 1
  data <- t(data)* (2^(8 * outChannels) - 1)
  
  arrData <- array(dim=c(dim(data),outChannels))
  
  for(i in 1:outChannels){
    arrData[,,i] <- floor((data %% (2 ^ (8*i))) / (2^(8*(i-1))) )
  }
  arrData <- arrData / (2^8-1)
  writePNG(arrData,outName)
  NULL
}

size <- 1.5
EEGChannels <- 16L
sampFreq <- 400
filtFreq <- 50
outFreq <- 100
length.out <- 60000
nChannels <- 4
testDirs <- paste("test_",1:3,"_new",sep="")
cropsize <- 16384 * 2
n.gpu <- 4

if(!('eval.rec' %in% dir('./inputfiles'))){
  
  if(!('evalImages' %in% dir('./download')))dir.create('./download/evalImages')
  
  key <- fread("./inputfiles/trainDataKey.csv")
  key <- key[isEval==1]
  files <- key[,image]
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  foreach(i = 1:length(files), .inorder=FALSE, .errorhandling = 'pass') %dopar% {
    cat(files[i],'\n')
    convertToImg(files[i], sampFreq, filtFreq, outFreq, size, nChannels, savePath = './download/evalImages/')
    cat(i,'\n')
  }
  stopImplicitCluster()
  key[,c('subject', 'empty', 'isEval') := NULL]
  key[,id := 1:.N]
  setcolorder(key, c('id', 'class', 'image'))
  key[, image := paste(word(image,-1,sep='/'),'.png',sep='')]
  fwrite(key, './inputfiles/evalFiles.lst', sep='\t', col.names = FALSE)
  
  system("/home/ubuntu/mxnet/bin/im2rec ./inputfiles/evalFiles.lst ./download/evalImages/ ./inputfiles/eval.rec encoding='.png' unchanged=1")
}


if(!('train.rec' %in% dir('./inputfiles'))){
  
  if(!('trainImages' %in% dir('./download')))dir.create('./download/trainImages')
  
  key <- fread("./inputfiles/trainDataKey.csv")
  key <- key[isEval==0]
  files <- key[,image]
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  foreach(i = 1:length(files), .inorder=FALSE) %dopar% {
    cat(files[i],'\n')
    convertToImg(files[i], sampFreq, filtFreq, outFreq, size, nChannels, savePath = './download/trainImages/')
    cat(i,'\n')
  }
  stopImplicitCluster()
  key[,c('subject', 'empty', 'isEval') := NULL]
  key[,id := 1:.N]
  setcolorder(key, c('id', 'class', 'image'))
  key[, image := paste(word(image,-1,sep='/'),'.png',sep='')]
  fwrite(key, './inputfiles/trainFiles.lst', sep='\t', col.names = FALSE)
  
  system("/home/ubuntu/mxnet/bin/im2rec ./inputfiles/trainFiles.lst ./download/trainImages/ ./inputfiles/train.rec encoding='.png' unchanged=1")
}

if(!('test.rec' %in% dir('./inputfiles'))){
  
  if(!('testImages' %in% dir('./download')))dir.create('./download/testImages')
  
  key <- data.table(image = list.files(paste("./download/",testDirs,sep=''), full.names = TRUE),
                    class = 0)
  files <- key[,image]
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  foreach(i = 1:length(files), .inorder=FALSE) %dopar% {
    gc()
    cat(files[i],'\n')
    convertToImg(files[i], sampFreq, filtFreq, outFreq, size, nChannels, savePath = './download/testImages/')
    cat(i,'\n')
  }
  stopImplicitCluster()
  key[,id := 1:.N]
  setcolorder(key, c('id', 'class', 'image'))
  key[, image := paste(word(image,-1,sep='/'),'.png',sep='')]
  fwrite(key, './inputfiles/testFiles.lst', sep='\t', col.names = FALSE)
  
  system("/home/ubuntu/mxnet/bin/im2rec ./inputfiles/testFiles.lst ./download/testImages/ ./inputfiles/test.rec encoding='.png' unchanged=1")
}

if(n.gpu > 0){
devices <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})
}else{
  devices <- list(mx.cpu())
}


nCores <- detectCores()
evalIter <- mx.io.ImageRecordIter(path.imglist = './inputfiles/evalFiles.lst',
                                  path.imgrec = './inputfiles/eval.rec',
                                  data.shape = c(cropsize, EEGChannels, nChannels),
                                  rand.crop = TRUE,
                                  batch.size = 48*length(devices))
                                  #preprocess.threads=4L, prefetch.buffer = 8)
trainIter <- mx.io.ImageRecordIter(path.imglist = './inputfiles/trainFiles.lst',
                                  path.imgrec = './inputfiles/train.rec',
                                  data.shape = c(cropsize, EEGChannels, nChannels),
                                  rand.crop = TRUE,
                                  #shuffle = TRUE,
                                  batch.size = 48*length(devices))
                                  #preprocess.threads=4L, prefetch.buffer = 8)
#thin Net parameters
nThinBlocks <- 1
nFatBlocks <- 3
kernel <- c(129,5,5,5,3,3)
stride <- c(64,2,2,2,1,1)
depth <- c(64,64,32,16,8,4)
nUnits <- c(2,2,2,2)
pad <- (kernel-1) / 2
pool <- c(F,T,T,T,F,F)
poolKernel <- rep(2,4)
poolStride <- poolKernel

thinResNet <- resNetNet(nThinBlocks, nFatBlocks, kernel, stride, depth, nUnits, pad, pool, poolKernel, poolStride)

logger <- mx.metric.logger$new()

thinNetModel <- thinNet.mx.model.FeedForward.create(thinResNet, trainIter, ctx = devices,#kvstore = "device", 
                                            num.round = 2000, #optimizer = 'adam',
                                            initializer = mx.init.Xavier("gaussian", "in", 2.0),
                                            eval.data = evalIter,
                                            eval.metric = mx.metric.logloss,
                                            batch.end.callback = mx.callback.log.train.metric(10,logger),
                                            epoch.end.callback= mx.callback.early.stop(5,logger, maximize=FALSE, prefix='./models/thinNet001'),
                                            init.params = list(bcWeight = array(256^((-nChannels):-1), dim=c(1,1,nChannels,1)),
                                                               bcBias = array(-.5)),
                                            learning.rate = 1e-2, momentum=0.9, 
                                            wd = 0.00001,
                                            clip_gradient = 2)

