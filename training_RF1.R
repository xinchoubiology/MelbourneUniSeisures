

library(h2o)

h2o.init(nthreads = -1)

full.hex <- h2o.uploadFile("train.csv")
full.hex[,ncol(full.hex)] <- as.factor(full.hex[,ncol(full.hex)])
full.hex[,ncol(full.hex)-1] <- as.factor(full.hex[,ncol(full.hex)-1])

names <- colnames(full.hex)
x <- names[-c(1,length(names))]
y <- "label"

# full.hex <- h2o.splitFrame(full.hex)
# train.hex <- full.hex[[1]]
# valid.hex <- full.hex[[2]]

modelRF1 <- h2o.randomForest(x, y, full.hex ,ntrees=200, max_depth = 50)


test.hex <- h2o.uploadFile("test.csv")
test.hex[,ncol(test.hex)] <- as.factor(test.hex[,ncol(test.hex)])

fit <- h2o.predict(modelRF1,test.hex)

output <- data.table(File = test$id, Class = fit$predict)
example <- fread("./download/sample_submission.csv")
example[!(File %in% output$File)] -> example
output <- rbind(output,example)
write.csv(output,file="sub1.csv",quote=F,row.names = F)
