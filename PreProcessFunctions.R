require(fftw)
require(R.matlab)
require(seewave)
require(data.table)
require(bigmemory)
require(doParallel)

# helper functions

cepstrumCoefs <- function(x, nMFCC, MFCCBank, fftPlan, dctPlan){
  # given a vector returns MFCCs
  
  x <- abs(FFT(x, plan=fftPlan)[1:(length(x)/2)])^2
  x <- DCT(log(as.vector(t(MFCCBank) %*% x)), plan=dctPlan, type=2)
  x[!is.finite(x)] <- NA
  return(x[1:nMFCC])
}

processMatFile <- function(i, file, dataMatrix, blockSize, nSteps, nMFCC, MFCCBank, fftPlan, dctPlan){
  # for each file read the data, process to overlapping MFCCs and return array
  cat(file)
  cat("\n")
  gc()
  data <- readMat(file)
  data <- data$dataStruct[[1]]
  nSamples <- dim(data)[1]
  nChannels <- dim(data)[2]
  step <- (nSamples - blockSize) / (nSteps - 1)
  
  arr <- array(0,c(nSteps,nChannels,nMFCC))
  
  for(iStep in 1:nSteps){
    startSample <- round(1 + (iStep-1)*step)
    endSample <- startSample + blockSize - 1
    for(iChannel in 1:nChannels){
      # for each channel at this step calculate the MFCCs
      arr[iStep, iChannel, ]  <- cepstrumCoefs(
        data[startSample:endSample,iChannel], nMFCC, MFCCBank, fftPlan, dctPlan)
    }
  }
  dataMatrix[i,] <- as.vector(arr)
  return(NULL)
}


buildMFCCInputData <- function(dirList, nSteps, nChannels, blockSize, nMFCC, nMFCCFilters, MFCCFreq, shuffle = FALSE, 
                               center=FALSE, scale = FALSE, downloadDir = "./download/"){
  
  # list all the files in that directory
  fileList <- list.files(paste(downloadDir,dirList,sep=''), full.names = TRUE)
  if(shuffle) fileList <- sample(fileList)
  #fileList <- head(fileList,3)
  
  MFCCBank <- melfilterbank(f=MFCCFreq,wl=blockSize,m=nMFCCFilters)$amp
  fftPlan <- planFFT(blockSize, effort = 3)
  dctPlan <- planDCT(nMFCCFilters, type = 2, effort =3)
  
  nCores <- detectCores()
  registerDoParallel(cores=nCores)
  
  out <- big.matrix(nrow=length(fileList), ncol=nSteps * nChannels * nMFCC)
  foreach(i=1:length(fileList),.inorder = FALSE,
          .packages=c("fftw", "R.matlab", "bigmemory"),
          .export=c("fileList","out", "blockSize", "nSteps", "nMFCC", "MFCCBank", "fftPlan", "dctPlan",
                    "processMatFile", "cepstrumCoefs")) %dopar% {
                      processMatFile(i, fileList[i], out, blockSize, nSteps, nMFCC, MFCCBank, fftPlan, dctPlan)
                    }
  
  cat("out done \n")
  #normalize array
  if(!center || !scale){
    means <- foreach(i=1:ncol(out), .combine = 'c') %dopar%{
      mean(out[,i], na.rm=TRUE)
    }
    SDs <- foreach(i=1:ncol(out), .combine = 'c') %dopar%{
      sd(out[,i], na.rm=TRUE)
    }
    norms <- data.table(means = means, SDs = SDs)
    norms[, MFCC := rep(1:nMFCC, each=nSteps*nChannels)]
    norms[, channel := rep(1:nChannels, times=nMFCC, each=nSteps)]
    norms[,c("means","SDs") := list(mean(means), mean(SDs)), by = c("MFCC","channel")]
  }
  if(!center)center <- norms$means
  if(!scale)scale <- norms$SDs
  #and because sometimes it's ok to use a for loop - not today
  foreach(j=1:ncol(out)) %dopar%{
    out[,j] <- (out[,j]-center[j]) / scale[j]
    out[mwhich(out,j,NA,'eq'),j] <- 0
    NULL
  }
  stopImplicitCluster()
  
  return(list(files = fileList, data = out, center=center, scale=scale))
}






