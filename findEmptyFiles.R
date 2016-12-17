setwd("~/MelbourneUniSeisures")

library(R.matlab)
library(data.table)
library(stringr)


source("./modelData.R")


key <- fread("./download/train_and_test_data_labels_safe.csv")
key <- key[safe==1]
key[, subject := substr(image,1,1)]
key[, count := str_count(image,"_")]
key[count==2,image := paste("./download/train_",subject,"/",image,sep='')]
key[count==1,image := paste("./download/test_",subject,"/",image,sep='')]


emptyFiles <- sapply(key$image, function(file){
  cat(file)
  cat('\n')
  data <- readMat(file)
  data <- data$dataStruct[[1]]
  out <- 0
  if(mean(data==0) > .8) out <- 1
  return(out)
})

key[,empty := emptyFiles]
key <- key[empty==0 | class==0]
key[, safe := NULL]
key[, count := NULL]

key <- key[sample(1:nrow(key))]

fwrite(key,'./inputfiles/trainDataKey.csv', quote=FALSE)
