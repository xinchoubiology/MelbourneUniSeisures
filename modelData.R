#parameters
s3URI <- "s3://robinrdata/melbourne-university-seizure-prediction/"
downloadDir <- "download/"
inputFilesDir <- "inputfiles/"
trainDirs <- paste("train_",1:3,sep="")
testDirs <- paste("test_",1:3,sep="")

nSteps <- 384
nChannels <- 16
blockSize <- 1024
nMFCC <- 26