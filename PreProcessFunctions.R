require(fftw)
require(R.matlab)
require(seewave)
require(data.table)
require(bigmemory)
require(doParallel)
require(moments)
require(fractal)

# helper functions

cepstrumCoefs <- function(x, full=TRUE, nMFCC, MFCCBank, fftPlan, dctPlan){
  # given a vector/Matrix returns MFCCs
  
  if (full){
    x <- abs(apply(x, 2, FFT, plan=fftPlan)[1:(dim(x)[1]/2),])^2 / dim(x)[1]^2
  } else {
    tmp <- abs(apply(x, 2, FFT)[1:(dim(x)[1]/2),])^2 / dim(x)[1]^2
    x <- matrix(NA, nrow=dim(MFCCBank)[1], ncol = dim(MFCCBank)[2])
    for(i in 1:dim(x)[2]){
      x[,i] <- approx(1:dim(tmp)[1], tmp[,i], n = dim(x)[1])$y
    }
  }
  x <- log(t(MFCCBank) %*% x)
  x <- apply(x, 2, DCT, plan=dctPlan, type=2)
  x[!is.finite(x)] <- NA
  return(x[1:nMFCC,])
}

processMatFile <- function(file, blockSize, nSteps, nMFCC, nMoments, MFCCBank, fftPlan, dctPlan){
  # for each file read the data, process to overlapping MFCCs and return array
  data <- readMat(file)
  data <- data$dataStruct[[1]]
  nSamples <- dim(data)[1]
  nChannels <- dim(data)[2]
  step <- (nSamples - blockSize) / (nSteps - 1)
  
  depth <- nChannels*(nMFCC + (nChannels-1) / 2 + nMoments)
  
  arr <- array(NA,c(nSteps,depth))
  
  for(iStep in 1:nSteps){  
    # for each channel at this step calculate the MFCCs
    startSample <- round(1 + (iStep-1)*step)
    endSample <- startSample + blockSize - 1
    x <- as.matrix(data[startSample:endSample,])
    pZerox <- mean(x==0)
    if(pZerox < 0.8){
      full <- TRUE
      if(pZerox > .01){
        x <- x[apply(x,1,any),]
        full <- FALSE
      }
      arr[iStep,1:(nChannels*nMFCC)] <- cepstrumCoefs(x, full, nMFCC, MFCCBank, fftPlan, dctPlan)
      n <- (nChannels*nMFCC)
      arr[iStep,(1:nChannels)+n] <- colMeans(x)
      n <- n + nChannels
      arr[iStep,(1:(nChannels*(nMoments-1)))+n] <- as.vector(all.moments(x, central = TRUE, order.max = nMoments)[-(1:2),])
      n <- n + nChannels*(nMoments-1)
      arr[iStep,(1:((nChannels-1)*nChannels/2))+n] <- cor(x)[upper.tri(cor(x))]
    }
  }
  return(arr)
}


buildMFCCInputData <- function(fileList, subject, nSteps, nChannels, blockSize, nMFCC, nMoments, nMFCCFilters, MFCCFreq,  
                               norms=NULL){
  
  
  MFCCBank <- melfilterbank(f=MFCCFreq,wl=blockSize,m=nMFCCFilters)$amp
  fftPlan <- planFFT(blockSize, effort = 3)
  dctPlan <- planDCT(nMFCCFilters, type = 2, effort =3)
  
  depth <- nChannels*(nMFCC + (nChannels-1) / 2 + nMoments)
  
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  
  out <- big.matrix(nrow=length(fileList), ncol=nSteps*depth)
  foreach(i=1:length(fileList),.inorder = FALSE) %dopar% {
    out[i,] <- as.vector(
      processMatFile(fileList[i], blockSize, nSteps, nMFCC, nMoments,  
                     MFCCBank, fftPlan, dctPlan))
    NULL 
  }
  
  cat("out done \n")
  #normalize array
  if(!is.data.table(norms)){
    for(subjID in 1:3){
      means <- foreach(i=1:ncol(out), .combine = 'c') %dopar%{
        mean(out[subject==subjID,i], na.rm=TRUE)
      }
      SDs <- foreach(i=1:ncol(out), .combine = 'c') %dopar%{
        sd(out[subject==subjID,i], na.rm=TRUE)
      }
      norms <- rbind(norms,data.table(means = means, SDs = SDs, subjID = subjID,
                                      iDepth = rep(1:depth, each=nSteps)))
    }
    norms[,c("means","SDs") := list(mean(means), mean(SDs)), by = c("subjID","iDepth")]
  }
  
  foreach(j=1:nrow(out)) %dopar%{
    out[j,] <- (out[j,]-norms[subjID == subject[j], means]) / norms[subjID == subject[j], SDs]
    NULL
  }
  
  foreach(j=1:ncol(out)) %dopar% {
    out[mwhich(out, j, NA, 'eq'),j] <- 0
    NULL
  }
  
  stopImplicitCluster()
  
  return(list(data = out, norms = norms))
}






