# Downloads competition data files from Kaggle

#file_list <- c(paste("train_",2:3,sep=""),paste("test_",1:3,sep=""))
file_list <- paste("test_",1:3,"_new",sep="")
url <- "https://www.kaggle.com/c/melbourne-university-seizure-prediction/download/"
download_directory <- "./download/"
s3_uri <- "s3://robinrdata/melbourne-university-seizure-prediction/download/"

# download any files not present
for(file in file_list){
  download.file(
    paste(url,file,".zip",sep=""),destfile=paste(download_directory,file,".zip",sep=""),
    method="wget", extra = "--load-cookies cookies.txt -a ./download.log")
  unzip(paste(download_directory,file,".zip",sep=""), exdir=download_directory, unzip="unzip")
  file.remove(paste(download_directory,file,".zip",sep=""))
  
  # cmnd <- paste("aws s3 cp",paste(download_directory,file,sep=""), 
  #               paste(s3_uri,file,sep=""),"--recursive")
  # system(cmnd)
   # 
   # cmnd <- paste("rm -f ",download_directory,file,".zip",sep="")
   # system(cmnd)
}


