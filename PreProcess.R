setwd("~/Documents/R Work/kaggle_seizures")

library(R.matlab)
library(data.table)
library(parallel)

window <- function(x, location, width){
  pmax(0,1 - 2 * abs(x-location) / width)
}

cepstrumCoefs <- function(spec,freq,N){
  width <- 0.5 / (N-1)  # actually half width but whatever
  location <- seq(0,0.5,length.out = N)
  sapply(location, function(x){
    sum(pmax(0,1 - abs(freq-x) / width) * spec)
  })
}

buildDataSet <- function(dirs,N){
  # reads all .mat files from a list of directorys and builds feature data.table
  rbindlist(lapply(dirs,function(y){
    #for each directory read each file
    files <- dir(y)
    
    
    rbindlist(mclapply(files, function(x){
      cat(paste("\n",x,"\n"))
      #for each file read the data and convert to data.table
      data <- readMat(paste(y,"/",x,sep=""))
      data <- data.table(matrix(data[[1]][1][[1]],ncol=16))
      for (i in names(data))
        data[get(i)==0,i:=NA,with=FALSE]
      
      data <- data[complete.cases(data)]
      if(nrow(data) < 5000) return() # blast on past if there's nothing to see
      
      out <- data.table(id = x)
      
      channels <- 1:16
      
      for(ch in channels){
        #cat(paste(" ",ch))
        pergram <- spectrum(data[,paste("V",ch,sep=""), with=F], plot=F) 
        out[,paste("CH",ch,"FR",1:N,sep="") := as.list(cepstrumCoefs(pergram$spec, pergram$freq, N))]
      }
      
      corMat <- cor(data)
      out[,paste("COR",1:120,sep="") := as.list(corMat[lower.tri(corMat)])]
      
      return(out)
      
    }, mc.cores=8))
  }))
}

downloadDir <- "./download"

trainDirs <- paste(downloadDir,"/train_",1:3,sep="")
testDirs <- paste(downloadDir,"/test_",1:3,sep="")

N <- 100

train <- buildDataSet(trainDirs, N)
train[,subject := as.numeric(substr(id,1,1))]
train[,label := as.numeric(substr(id,nchar(id)-4,nchar(id)-4))]

write.csv(train,file="train.csv", row.names=FALSE)

test <- buildDataSet(testDirs, N)
test[,subject := as.numeric(substr(id,1,1))]
write.csv(test,file="test.csv", row.names=FALSE)

