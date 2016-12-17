
R.io.matFileIter <- function(files, labels, cropLength, randCrop, cropStart, sampFreq, resampFreq, noise,
                             batchSize, nCores, queLength){
  require(R.matlab)
  require(seewave)
  require(abind)
  require(doParallel)
  require(mxnet)
  
  processFile <- function(file){
    
    data <- readMat(file)
    data <- data$dataStruct[[1]]
    
    data <- apply(data, 2, resamp, f=sampFreq, g=resampFreq)
    data <- data[(1:cropLength)+cropStart,]
    if(noise !=0 )data <- data + rnorm(prod(dim(data)), sd=noise)
    data <- array(data, dim=c(dim(data)[1],1,dim(data)[2],1))
  }
  
  processBatch <- function(files){
    registerDoParallel(cores=nCores)
    batch <- foreach(file=files, .combine = 'abind') %dopar% {
      if(randCrop) cropStart <- sample(0:cropStart)
      processFile(file)
    } 
    mx.nd.array(batch)
  }
  
  iBatch <- NULL
  nBatch <- ceiling(length(files)/batchSize)
  que <- list()
  
  
  reset <- function(){
    iBatch <<- 0
    que <<- lapply(1:queLength, function(i){
      idx <- (1:batchSize)+(i-1)*batchSize
      data <- processBatch(files[idx])
      label <- mx.nd.array(labels[idx])
      list(data=data, label=label)
    })
  }
  
  iter.next <- function(){
    if(iBatch<nBatch){
      if(iBatch!=0){
        iQue <- iBatch %% queLength
        if(iQue==0)iQue <- queLength
        idx <- (1:batchSize) + (iBatch+queLength-1) * batchSize 
        idx <- idx %% length(files)
        idx[idx==0] <- length(files)
        que[[iQue]] <<- list(    
          data = processBatch(files[idx]),
          label = mx.nd.array(labels[idx]))
      }
      iBatch <<- iBatch+1
      TRUE
    }else{
      FALSE
    }
  }
  
  value <- function(){
    iQue <- iBatch %% queLength
    if(iQue==0)iQue <- queLength
    que[[iQue]]
  }
  
  reset()
  
  out <- list(reset=reset, iter.next=iter.next, value=value)
  class(out) <- "Rcpp_MXNativeDataIter"
  return(out)
}