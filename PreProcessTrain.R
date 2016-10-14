setwd("~/MelbourneUniSeisures")
library(data.table)
library(stringr)
#preProcess Training examples

source("./modelData.R")

IDs <- fread(paste(inputFilesDir,"trainNames.lst",sep=""),header=FALSE)
IDs[,V1 := str_sub(V1,-5,-5)]

fwrite(IDs,paste(inputFilesDir,"trainLabels.csv",sep=""), quote=F, col.names = F)

data <- fread(paste(inputFilesDir,"trainData.csv",sep=""), header=FALSE)

for (j in 1:ncol(data)) set(data, which(!is.finite(data[[j]])), j, 0)

trainNorms <- data.table(means = colMeans(data), SDs = apply(data, 2, sd))
fwrite(trainNorms, paste(inputFilesDir,"trainColNorms.csv",sep=""))

data <- as.data.table(scale(data))
fwrite(data, paste(inputFilesDir,"trainData.csv",sep=""))
