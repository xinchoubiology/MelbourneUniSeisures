setwd("~/MelbourneUniSeisures")

require(mxnet)
require(data.table)

source("./modelData.R")

model <- mx.model.load("./models/thinNet002",20)

testIter <- mx.io.CSVIter(data.csv = "./inputfiles/testData.csv", data.shape = c(nSteps,nChannels,nMFCC),
                           batch.size = 128,
                          preprocess.threads=4, prefetch.buffer = 2)



preds <- predict(model,testIter,mx.gpu())
preds <- as.vector(preds)

submission <- fread("./inputfiles/testNames.lst",header=TRUE)
submission[,Class := preds]

fwrite(submission,file="./submissionfiles/thinNet002.csv",col.names = TRUE)
