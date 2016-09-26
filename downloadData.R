# Downloads competition data files from Kaggle

file_list <- c(paste("train_",1:3,".zip",sep=""),paste("test_",1:3,".zip",sep=""))
url <- "https://www.kaggle.com/c/melbourne-university-seizure-prediction/download/"
download_directory <- "./download/"

#checks to see if files are already here
file_list <- file_list[!(file_list %in% paste(dir(download_directory),".zip",sep=""))]

# download any files not present
for(file in file_list){
  if(!(file %in% dir(download_directory))) download.file(
    paste(url,file,sep=""),destfile=paste(download_directory,file,sep=""),
    method="wget", extra = "--load-cookies cookies.txt -a ./download.log")
}

# unzip the files and delete the .zip files
for(file in file_list){
  unzip(paste(download_directory,file,sep=""), exdir = download_directory, unzip="unzip")
  file.remove(paste(download_directory,file,sep=""))
}