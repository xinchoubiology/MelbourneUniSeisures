setwd("~/MelbourneUniSeisures")

library(stringr)
library(data.table)

source("./PreProcessFunctions.R")
source("./modelData.R")


train <- buildMFCCInputData(trainDirs, nSteps, nChannels, blockSize, nMFCC, nMFCCFilters, MFCCFreq, shuffle = TRUE)
write.table(str_sub(train$files,-5,-5),paste("./inputfiles/trainLabels.lst"), quote = FALSE, col.names = FALSE, row.names = FALSE)
write.table(word(train$files,-1,sep='/'),paste("./inputfiles/trainNames.lst"), quote = FALSE, col.names = FALSE, row.names = FALSE)
write.big.matrix(train$data,"./inputfiles/trainData.csv")
center <- train$center
scale <- train$scale

rm(train)
gc()

test <- buildMFCCInputData(testDirs, nSteps, nChannels, blockSize, nMFCC, nMFCCFilters, MFCCFreq, center=center, scale=scale)
write.table(word(test$files,-1,sep='/'),paste("./inputfiles/testNames.lst"), quote = FALSE, col.names = FALSE, row.names = FALSE)
write.big.matrix(test$data,"./inputfiles/testData.csv")
