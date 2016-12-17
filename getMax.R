setwd("~/MelbourneUniSeisures")

library(data.table)

source("./modelData.R")

key <- fread("./inputfiles/trainDataKey.csv")
testFiles <- list.files(paste("./download/",testDirs,sep=''), full.names = TRUE)

