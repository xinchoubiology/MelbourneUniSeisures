setwd("~/MelbourneUniSeisures")
library(R.matlab)
library(data.table)
library(parallel)
library(stringr)
library(seewave)
library(fftw)


# helper functions

cepstrumCoefs <- function(x,filterBank, fftPlan, dctPlan){
  power <- abs(FFT(x, plan=fftPlan)[1:(length(x)/2)])^2
  DCT(log(as.vector(t(filterBank) %*% power)), plan=dctPlan, type=2)
}


buildMFCCInputData <- function(dirList, nSteps, blockSize, nMFCC, MFCCBank, fftPlan, dctPlan){
  
  # list all the files in that directory
  fileList <- unlist(sapply(dirList, function(dir){
    if(!(dir %in% dir(downloadDir))) dir.create(paste(downloadDir,dir,sep=""))
    paste(dir,"/", word(system(paste(
      "aws s3 ls ", s3URI, downloadDir, dir,"/", sep=''
    ), intern = T),-1),sep="")
  }))
  
  fileList <- head(fileList,3)
  
  MFCCBank <- melfilterbank(f=16000,wl=blockSize,m=nMFCC)$amp
  fftPlan <- planFFT(blockSize, effort = 3)
  dctPlan <- planDCT(nMFCC, type = 2, effort =3)
  
  write.table(fileList, file=paste("./",inputFilesDir,"fileNames.lst",sep=""),
              quote=F, row.names = F, col.names = F)
  
  nCores <- detectCores()
  
  out <- rbindlist(mclapply(fileList, function(file){
    # cat(paste("\n",file,"\n"))
    # for each file download, read the data and convert to data.table
    system(paste("aws s3 cp ",s3URI,downloadDir,file," ./",downloadDir,file, sep = ""))
    data <- readMat(paste("./",downloadDir,file, sep = ""))
    file.remove(paste("./",downloadDir,file, sep = ""))
    
    data <- as.data.table(data$dataStruct[[1]])
    nChannels <- ncol(data)
    nSamples <- nrow(data)
    step <- (nSamples - blockSize) / (nSteps - 1)
    
    tmp <- array(0,c(nChannels,nSteps,nMFCC))
    
    for(iStep in 1:nSteps){
      startSample <- round(1 + (iStep-1)*step)
      endSample <- startSample + blockSize - 1
      for(iChannel in 1:nChannels){
        # for each channel at this step calculate the MFCCs
        tmp[iChannel, iStep, ]  <- cepstrumCoefs(
          data[startSample:endSample,get(paste("V",iChannel,sep=""))], MFCCBank,
          fftPlan, dctPlan)
      }
    }
    #convert to data.table
    DT <-setDT(as.list(tmp))
    rm(tmp)
    gc()
    DT
  }, mc.cores=nCores))
  fwrite(out, file=paste("./",inputFilesDir,"data.csv",sep=""),
         quote=F, col.names = F)
  rm(out)
  gc()
}

source("./modelData.R")
buildMFCCInputData(trainDirs, nSteps, blockSize, nMFCC, MFCCBank, fftPlan, dctPlan)




