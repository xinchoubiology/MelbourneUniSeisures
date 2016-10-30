#parameters
trainDirs <- paste("train_",1:3,sep="")
testDirs <- paste("test_",1:3,sep="")

MFCCFreq <- 16000
nMFCCFilters <- 24

nSteps <- 896
nChannels <- 16
blockSize <- 512
nMFCC <- 16