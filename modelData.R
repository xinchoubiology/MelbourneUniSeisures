#parameters
trainDirs <- c(paste("train_",1:3,sep=""),paste("test_",1:3,sep=""))
testDirs <- paste("test_",1:3,"_new",sep="")

MFCCFreq <- 8000
nMFCCFilters <- 16

nSteps <- 128
nChannels <- 16
blockSize <- 2048
nMFCC <- 16
nMoments <- 4

dataDepth <- nChannels*(nMFCC + (nChannels-1) / 2 + nMoments)