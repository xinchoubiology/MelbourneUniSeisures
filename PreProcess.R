setwd("~/MelbourneUniSeisures")

library(stringr)
library(data.table)
library(bigmemory)

source("./PreProcessFunctions.R")
source("./modelData.R")

key <- fread("./inputfiles/trainDataKey.csv")
testFiles <- list.files(paste("./download/",testDirs,sep=''), full.names = TRUE)

subject <- c(as.integer(substr(word(key$image,-1,sep='/'),1,1)),
             as.integer(substr(word(testFiles,-1,sep='/'),5,5)))

fnames <- c(key$image, testFiles)


isEval <- as.logical(key[,isEval])
write.table(key$class[!isEval],paste("./inputfiles/trainLabels.lst"), quote = FALSE, 
            col.names = FALSE, row.names = FALSE)
write.table(key$class[isEval],paste("./inputfiles/evalLabels.lst"), quote = FALSE, 
            col.names = FALSE, row.names = FALSE)
write.table(word(key$image[!isEval],-1,sep='/'),paste("./inputfiles/trainNames.lst"), quote = FALSE, 
            col.names = FALSE, row.names = FALSE)

write.table(word(testFiles,-1,sep='/'),paste("./inputfiles/testNames.lst"), quote = FALSE, 
            col.names = FALSE, row.names = FALSE)

data <- buildMFCCInputData(fnames, subject, nSteps, nChannels, blockSize, nMFCC, nMoments, nMFCCFilters, MFCCFreq)
fwrite(data$norms,"./inputfiles/norms.csv")

nEval <- sum(isEval)
nTrain <- sum(!isEval)
nTest <- length(fnames) - nEval - nTrain

write.big.matrix(sub.big.matrix(data$data, lastRow = nEval), "./inputfiles/evalData.csv")
write.big.matrix(sub.big.matrix(data$data, firstRow = 1+nEval, lastRow = nEval+nTrain), "./inputfiles/trainData.csv")
write.big.matrix(sub.big.matrix(data$data, firstRow = 1+nEval+nTrain), "./inputfiles/testData.csv")



