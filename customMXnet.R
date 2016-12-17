require(mxnet)

mx.nd.slice <- mxnet:::mx.nd.slice
mx.model.init.params.rnn <- mxnet:::mx.model.init.params.rnn
mx.model.init.iter <- mxnet:::mx.model.init.iter
mx.model.create.kvstore <- mxnet:::mx.model.create.kvstore
mx.util.filter.null <- mxnet:::mx.util.filter.null
mx.model.check.arguments <- mxnet:::mx.model.check.arguments
mx.model.extract.model <- mxnet:::mx.model.extract.model
is.MXSymbol <- mxnet:::is.MXSymbol
mx.util.str.endswith <- mxnet:::mx.util.str.endswith
mx.symbol.bind <- mxnet:::mx.symbol.bind
mx.nd.arraypacker <- mxnet:::mx.nd.arraypacker
mx.model.slice.shape <- mxnet:::mx.model.slice.shape

custom.mx.model.FeedForward.create <- function (symbol, X, y = NULL, ctx = NULL, num.round = 10, optimizer = "sgd", 
                                                initializer = mx.init.uniform(0.01), eval.data = NULL, eval.metric = NULL, 
                                                epoch.end.callback = NULL, batch.end.callback = NULL, array.batch.size = 128, 
                                                array.layout = "auto", kvstore = "local", verbose = TRUE, 
                                                arg.params = NULL, aux.params = NULL, num.lstm.layer = 1, num.hidden = 1, ...) {
  
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mx.model.select.layout.train(X, y)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  X <- mx.model.init.iter(X, y, batch.size = array.batch.size, 
                          is.train = TRUE)
  if (!X$iter.next()) {
    X$reset()
    if (!X$iter.next()) 
      stop("Empty input")
  }
  
  input.shape <- dim((X$value())$data)
  ndim <- length(input.shape)
  batchsize = input.shape[[ndim]]
  input.shape <- list(data = input.shape)
  init.states.c <- lapply(1:num.lstm.layer, function(i) {
    state.c <- paste0("l", i, ".init.c")
    input.shape[[state.c]] <<- c(num.hidden[i], batchsize)
    return (state.c)
  })
  init.states.h <- lapply(1:num.lstm.layer, function(i) {
    state.h <- paste0("l", i, ".init.h")
    input.shape[[state.h]] <<- c(num.hidden[i], batchsize)
    return (state.h)
  })
  init.states.name <- c(init.states.c, init.states.h)
  
  params <- mx.model.init.params.rnn(symbol, input.shape, initializer, 
                                     mx.cpu())
  for (name in init.states.name) {
    params$arg.params[[name]] <- mx.nd.zeros(input.shape[[name]])
  }
  if (!is.null(arg.params)) 
    params$arg.params <- arg.params
  if (!is.null(aux.params)) 
    params$aux.params <- aux.params
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    optimizer <- mx.opt.create(optimizer, rescale.grad = (1/batchsize), 
                               ...)
  }
  if (!is.null(eval.data) && !is.list(eval.data) && !is.mx.dataiter(eval.data)) {
    stop("The validation set should be either a mx.io.DataIter or a R list")
  }
  if (is.list(eval.data)) {
    if (is.null(eval.data$data) || is.null(eval.data$label)) {
      stop("Please provide the validation set as list(data=R.array, label=R.array)")
    }
    if (is.array(eval.data$data) || is.matrix(eval.data$data)) {
      if (array.layout == "auto") {
        array.layout <- mx.model.select.layout.train(eval.data$data, 
                                                     eval.data$label)
      }
      if (array.layout == "rowmajor") {
        eval.data$data <- t(eval.data$data)
      }
    }
    eval.data <- mx.model.init.iter(eval.data$data, eval.data$label, 
                                    batch.size = array.batch.size, is.train = TRUE)
  }
  kvstore <- mx.model.create.kvstore(kvstore, params$arg.params, 
                                     length(ctx), verbose = verbose)
  model <- custom.mx.model.train(symbol, ctx, input.shape, params$arg.params, 
                                 params$aux.params, 1, num.round, optimizer = optimizer, 
                                 train.data = X, eval.data = eval.data, metric = eval.metric, 
                                 epoch.end.callback = epoch.end.callback, batch.end.callback = batch.end.callback, 
                                 kvstore = kvstore, verbose = verbose)
  return(model)
}





custom.mx.model.train <- function (symbol, ctx, input.shape, arg.params, aux.params, begin.round, 
                                   end.round, optimizer, train.data, eval.data, metric, epoch.end.callback, 
                                   batch.end.callback, kvstore, verbose = TRUE) {
  
  ndevice <- length(ctx)
  if (verbose) 
    cat(paste0("Start training with ", ndevice, " devices\n"))
  sliceinfo <- custom.mx.model.slice.shape(input.shape, ndevice)
  train.execs <- lapply(1:ndevice, function(i) {
    custom.mx.simple.bind(symbol, ctx = ctx[[i]], data = sliceinfo[[i]]$shape, 
                          grad.req = "write")
  })
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, arg.params, match.name = TRUE)
    mx.exec.update.aux.arrays(texec, aux.params, match.name = TRUE)
  }
  params.index <- as.integer(mx.util.filter.null(lapply(1:length(train.execs[[1]]$ref.grad.arrays), 
                                                        function(k) {
                                                          if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
                                                        })))
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  }
  else {
    updaters <- lapply(1:ndevice, function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  input.names <- mx.model.check.arguments(symbol)
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    while (train.data$iter.next()) {
      dlist <- train.data$value()
      slices <- lapply(1:ndevice, function(i) {
        s <- sliceinfo[[i]]
        ret <- list(data = mx.nd.slice(dlist$data, s$begin, 
                                       s$end), label = mx.nd.slice(dlist$label, s$begin, 
                                                                   s$end))
        return(ret)
      })
      for (i in 1:ndevice) {
        s <- slices[[i]]
        names(s) <- input.names
        mx.exec.update.arg.arrays(train.execs[[i]], s, 
                                  match.name = TRUE)
      }
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train = TRUE)
      }
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
      })
      for (texec in train.execs) {
        mx.exec.backward(texec)
      }
      if (!is.null(kvstore)) {
        kvstore$push(params.index, lapply(train.execs, 
                                          function(texec) {
                                            texec$ref.grad.arrays[params.index]
                                          }), -params.index)
      }
      if (update.on.kvstore) {
        kvstore$pull(params.index, lapply(train.execs, 
                                          function(texec) {
                                            texec$ref.arg.arrays[params.index]
                                          }), -params.index)
      }
      else {
        if (!is.null(kvstore)) {
          kvstore$pull(params.index, lapply(train.execs, 
                                            function(texec) {
                                              texec$ref.grad.arrays[params.index]
                                            }), -params.index)
        }
        arg.blocks <- lapply(1:ndevice, function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, 
                        train.execs[[i]]$ref.grad.arrays)
        })
        for (i in 1:ndevice) {
          mx.exec.update.arg.arrays(train.execs[[i]], 
                                    arg.blocks[[i]], skip.null = TRUE)
        }
      }
      if (!is.null(metric)) {
        for (i in 1:ndevice) {
          train.metric <- metric$update(slices[[i]]$label, 
                                        out.preds[[i]], train.metric)
        }
      }
      nbatch <- nbatch + 1
      if (!is.null(batch.end.callback)) {
        batch.end.callback(iteration, nbatch, environment())
      }
    }
    train.data$reset()
    if (!is.null(metric)) {
      result <- metric$get(train.metric)
      if (verbose) 
        cat(paste0("[", iteration, "] Train-", result$name, 
                   "=", result$value, "\n"))
    }
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      while (eval.data$iter.next()) {
        dlist <- eval.data$value()
        slices <- lapply(1:ndevice, function(i) {
          s <- sliceinfo[[i]]
          ret <- list(data = mx.nd.slice(dlist$data, 
                                         s$begin, s$end), label = mx.nd.slice(dlist$label, 
                                                                              s$begin, s$end))
          return(ret)
        })
        for (i in 1:ndevice) {
          s <- slices[[i]]
          names(s) <- input.names
          mx.exec.update.arg.arrays(train.execs[[i]], 
                                    s, match.name = TRUE)
        }
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train = FALSE)
        }
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
        })
        if (!is.null(metric)) {
          for (i in 1:ndevice) {
            eval.metric <- metric$update(slices[[i]]$label, 
                                         out.preds[[i]], eval.metric)
          }
        }
      }
      eval.data$reset()
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if (verbose) 
          cat(paste0("[", iteration, "] Validation-", 
                     result$name, "=", result$value, "\n"))
      }
    }
    else {
      eval.metric <- NULL
    }
    model <- mx.model.extract.model(symbol, train.execs)
    epoch_continue <- TRUE
    if (!is.null(epoch.end.callback)) {
      epoch_continue <- epoch.end.callback(iteration, 0, 
                                           environment(), verbose = verbose)
    }
    if (!epoch_continue) {
      break
    }
  }
  return(model)
}


custom.mx.model.slice.shape <- function (shape, nsplit) {
  ndim <- length(shape[[1]])
  batchsize <- shape[[1]][[ndim]]
  step <- as.integer((batchsize + nsplit - 1)/nsplit)
  lapply(0:(nsplit - 1), function(k) {
    begin = min(k * step, batchsize)
    end = min((k + 1) * step, batchsize)
    s <- lapply(shape, function(sh){
      ndim <- length(sh)
      sh[[ndim]] = end - begin
      sh
    })
    return(list(begin = begin, end = end, shape = s))
  })
}

custom.mx.simple.bind <- function (symbol, ctx, grad.req = "null", ...) {
  if (!is.MXSymbol(symbol)) 
    stop("symbol need to be MXSymbol")
  slist <- symbol$infer.shape(...)
  if (is.null(slist)) {
    stop("Need more shape information to decide the shapes of arguments")
  }
  arg.arrays <- sapply(slist$arg.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  aux.arrays <- sapply(slist$aux.shapes, function(shape) {
    mx.nd.zeros(shape, ctx)
  }, simplify = FALSE, USE.NAMES = TRUE)
  grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
    if (!mx.util.str.endswith(nm, "label") && !mx.util.str.endswith(nm, 
                                                                    "data")) {
      grad.req
    }
    else {
      "null"
    }
  })
  mx.symbol.bind(symbol, ctx, arg.arrays = arg.arrays, aux.arrays = aux.arrays, 
                 grad.reqs = grad.reqs)
}





custom.predict.MXFeedForwardModel <- function (model, X, ctx = NULL, array.batch.size = 128, array.layout = "auto",
                                               num.lstm.layer = 1, num.hidden = 1) 
{
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mx.model.select.layout.predict(X, 
                                                     model)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  X <- mx.model.init.iter(X, NULL, batch.size = array.batch.size, 
                          is.train = FALSE)
  X$reset()
  if (!X$iter.next()) 
    stop("Cannot predict on empty iterator")
  
  input.shape <- dim((X$value())$data)
  ndim <- length(input.shape)
  batchsize = input.shape[[ndim]]
  input.shape <- list(data = dim((X$value())$data))
  init.states.c <- lapply(1:num.lstm.layer, function(i) {
    state.c <- paste0("l", i, ".init.c")
    input.shape[[state.c]] <<- c(num.hidden[i], batchsize)
    return (state.c)
  })
  init.states.h <- lapply(1:num.lstm.layer, function(i) {
    state.h <- paste0("l", i, ".init.h")
    input.shape[[state.h]] <<- c(num.hidden[i], batchsize)
    return (state.h)
  })
  init.states.name <- c(init.states.c, init.states.h)
  pexec <- custom.mx.simple.bind(model$symbol, ctx = ctx, grad.req = "null",  input.shape)
  mx.exec.update.arg.arrays(pexec, model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(pexec, model$aux.params, match.name = TRUE)
  packer <- mx.nd.arraypacker()
  X$reset()
  while (X$iter.next()) {
    dlist = X$value()
    mx.exec.update.arg.arrays(pexec, list(data = dlist$data), 
                              match.name = TRUE)
    mx.exec.forward(pexec, is.train = FALSE)
    out.pred <- mx.nd.copyto(pexec$ref.outputs[[1]], mx.cpu())
    padded <- X$num.pad()
    oshape <- dim(out.pred)
    ndim <- length(oshape)
    packer$push(mx.nd.slice(out.pred, 0, oshape[[ndim]] - 
                              padded))
  }
  X$reset()
  return(packer$get())
}

# Custom evaluation metrics on CRPS.
mx.metric.auc<- mx.metric.custom("auc", function(label, pred) {
  return(Metrics::auc(label, pred))
})
mx.metric.logloss<- mx.metric.custom("logloss", function(label, pred) {
  return(Metrics::logLoss(label, pred))
})

mx.callback.early.stop <- function (nrounds, logger = NULL, maximize=TRUE, prefix='model')
{
  function(iteration, nbatch, env, verbose=TRUE) {
    if (!is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0) stop("Only use for epoch.end.callback")
      if (!is.null(logger)) {
        if (class(logger) != "mx.metric.logger") stop("Invalid mx.metric.logger.")
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          logger$eval <- c(logger$eval, result$value)
          if(maximize) {
            bst.round <- which.max(logger$eval)
          } else {
            bst.round <- which.min(logger$eval)
          }
          if(iteration == bst.round){
            mx.model.save(env$model, prefix, iteration)
            if (verbose) 
              cat(sprintf("Model checkpoint saved to %s-%04d.params\n", 
                          prefix, iteration))
          }
          if (verbose) {
            cat(paste0("Best round: ", bst.round, "; Best metric: ", logger$eval[bst.round], "\n"))
            cat(paste0("This round: ", result$value,"\n"))
          }
          if((iteration-bst.round) >= nrounds) {
            if(verbose){
              cat(paste0("Stopping training and returning best model from Iteration: ", bst.round,"\n"))
            }
            env.model <- mx.model.load(prefix, bst.round)
            return(FALSE)
            }
        }
      }
      return(TRUE)
    }
  }
}

mx.callback.log.train.metric <- function (period, logger = NULL) 
{
  function(iteration, nbatch, env, verbose = TRUE) {
    if (nbatch%%period == 0 && !is.null(env$metric)) {
      result <- env$metric$get(env$train.metric)
      if (nbatch != 0){ 
        if(verbose) cat(paste0("Batch [", nbatch, "] Train-", result$name, 
                   "=", result$value, "\n"))
        if (!is.null(logger)) {
          if (class(logger) != "mx.metric.logger") {
            stop("Invalid mx.metric.logger.")
          }
          logger$train <- c(logger$train, result$value)
        }
      }
    }
    return(TRUE)
  }
}


thinNet.mx.model.FeedForward.create <- function (symbol, X, y = NULL, ctx = NULL, num.round = 10, optimizer = "sgd", 
                                                initializer = mx.init.uniform(0.01), eval.data = NULL, eval.metric = NULL, 
                                                epoch.end.callback = NULL, batch.end.callback = NULL, array.batch.size = 128, 
                                                array.layout = "auto", kvstore = "local", verbose = TRUE, 
                                                arg.params = NULL, aux.params = NULL, init.params = NULL, ...) {
  
  if (is.array(X) || is.matrix(X)) {
    if (array.layout == "auto") {
      array.layout <- mx.model.select.layout.train(X, y)
    }
    if (array.layout == "rowmajor") {
      X <- t(X)
    }
  }
  X <- mx.model.init.iter(X, y, batch.size = array.batch.size, 
                          is.train = TRUE)
  if (!X$iter.next()) {
    X$reset()
    if (!X$iter.next()) 
      stop("Empty input")
  }
  
  input.shape <- dim((X$value())$data)
  ndim <- length(input.shape)
  batchsize = input.shape[[ndim]]
  input.shape <- list(data = input.shape)
  init.params.name <- names(init.params)
  input.shape <- append(input.shape,
                        lapply(init.params,function(param){
                          dim(param)
                        }))
  
  params <- mx.model.init.params.rnn(symbol, input.shape, initializer, 
                                     mx.cpu())
  for (name in init.params.name) {
    params$arg.params[[name]] <- mx.nd.array(init.params[[name]])
  }
  if (!is.null(arg.params)) 
    params$arg.params <- arg.params
  if (!is.null(aux.params)) 
    params$aux.params <- aux.params
  if (is.null(ctx)) 
    ctx <- mx.ctx.default()
  if (is.mx.context(ctx)) {
    ctx <- list(ctx)
  }
  if (!is.list(ctx)) 
    stop("ctx must be mx.context or list of mx.context")
  if (is.character(optimizer)) {
    optimizer <- mx.opt.create(optimizer, rescale.grad = (1/batchsize), 
                               ...)
  }
  if (!is.null(eval.data) && !is.list(eval.data) && !is.mx.dataiter(eval.data)) {
    stop("The validation set should be either a mx.io.DataIter or a R list")
  }
  if (is.list(eval.data)) {
    if (is.null(eval.data$data) || is.null(eval.data$label)) {
      stop("Please provide the validation set as list(data=R.array, label=R.array)")
    }
    if (is.array(eval.data$data) || is.matrix(eval.data$data)) {
      if (array.layout == "auto") {
        array.layout <- mx.model.select.layout.train(eval.data$data, 
                                                     eval.data$label)
      }
      if (array.layout == "rowmajor") {
        eval.data$data <- t(eval.data$data)
      }
    }
    eval.data <- mx.model.init.iter(eval.data$data, eval.data$label, 
                                    batch.size = array.batch.size, is.train = TRUE)
  }
  kvstore <- mx.model.create.kvstore(kvstore, params$arg.params, 
                                     length(ctx), verbose = verbose)
  model <- thinNet.mx.model.train(symbol, ctx, input.shape, params$arg.params, 
                                 params$aux.params, 1, num.round, optimizer = optimizer, 
                                 train.data = X, eval.data = eval.data, metric = eval.metric, 
                                 epoch.end.callback = epoch.end.callback, batch.end.callback = batch.end.callback, 
                                 kvstore = kvstore, verbose = verbose)
  return(model)
}




thinNet.mx.model.train <- function (symbol, ctx, input.shape, arg.params, aux.params, begin.round, 
                                   end.round, optimizer, train.data, eval.data, metric, epoch.end.callback, 
                                   batch.end.callback, kvstore, verbose = TRUE) {
  
  ndevice <- length(ctx)
  if (verbose) 
    cat(paste0("Start training with ", ndevice, " devices\n"))
  sliceinfo <- mx.model.slice.shape(input.shape$data, ndevice)
  train.execs <- lapply(1:ndevice, function(i) {
    shape <- input.shape
    shape$data <- sliceinfo[[i]]$shape
    custom.mx.simple.bind(symbol, ctx = ctx[[i]], data = shape, 
                          grad.req = "write")
  })
  for (texec in train.execs) {
    mx.exec.update.arg.arrays(texec, arg.params, match.name = TRUE)
    mx.exec.update.aux.arrays(texec, aux.params, match.name = TRUE)
  }
  params.index <- as.integer(mx.util.filter.null(lapply(1:length(train.execs[[1]]$ref.grad.arrays), 
                                                        function(k) {
                                                          if (!is.null(train.execs[[1]]$ref.grad.arrays[[k]])) k else NULL
                                                        })))
  update.on.kvstore <- FALSE
  if (!is.null(kvstore) && kvstore$update.on.kvstore) {
    update.on.kvstore <- TRUE
    kvstore$set.optimizer(optimizer)
  }
  else {
    updaters <- lapply(1:ndevice, function(i) {
      mx.opt.get.updater(optimizer, train.execs[[i]]$ref.arg.arrays)
    })
  }
  if (!is.null(kvstore)) {
    kvstore$init(params.index, train.execs[[1]]$ref.arg.arrays[params.index])
  }
  input.names <- mx.model.check.arguments(symbol)
  for (iteration in begin.round:end.round) {
    nbatch <- 0
    if (!is.null(metric)) {
      train.metric <- metric$init()
    }
    while (train.data$iter.next()) {
      dlist <- train.data$value()
      slices <- lapply(1:ndevice, function(i) {
        s <- sliceinfo[[i]]
        ret <- list(data = mx.nd.slice(dlist$data, s$begin, 
                                       s$end), label = mx.nd.slice(dlist$label, s$begin, 
                                                                   s$end))
        return(ret)
      })
      for (i in 1:ndevice) {
        s <- slices[[i]]
        names(s) <- input.names
        mx.exec.update.arg.arrays(train.execs[[i]], s, 
                                  match.name = TRUE)
      }
      for (texec in train.execs) {
        mx.exec.forward(texec, is.train = TRUE)
      }
      out.preds <- lapply(train.execs, function(texec) {
        mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
      })
      for (texec in train.execs) {
        mx.exec.backward(texec)
      }
      if (!is.null(kvstore)) {
        kvstore$push(params.index, lapply(train.execs, 
                                          function(texec) {
                                            texec$ref.grad.arrays[params.index]
                                          }), -params.index)
      }
      if (update.on.kvstore) {
        kvstore$pull(params.index, lapply(train.execs, 
                                          function(texec) {
                                            texec$ref.arg.arrays[params.index]
                                          }), -params.index)
      }
      else {
        if (!is.null(kvstore)) {
          kvstore$pull(params.index, lapply(train.execs, 
                                            function(texec) {
                                              texec$ref.grad.arrays[params.index]
                                            }), -params.index)
        }
        arg.blocks <- lapply(1:ndevice, function(i) {
          updaters[[i]](train.execs[[i]]$ref.arg.arrays, 
                        train.execs[[i]]$ref.grad.arrays)
        })
        for (i in 1:ndevice) {
          mx.exec.update.arg.arrays(train.execs[[i]], 
                                    arg.blocks[[i]], skip.null = TRUE)
        }
      }
      if (!is.null(metric)) {
        for (i in 1:ndevice) {
          train.metric <- metric$update(slices[[i]]$label, 
                                        out.preds[[i]], train.metric)
        }
      }
      nbatch <- nbatch + 1
      if (!is.null(batch.end.callback)) {
        batch.end.callback(iteration, nbatch, environment())
      }
    }
    train.data$reset()
    if (!is.null(metric)) {
      result <- metric$get(train.metric)
      if (verbose) 
        cat(paste0("[", iteration, "] Train-", result$name, 
                   "=", result$value, "\n"))
    }
    if (!is.null(eval.data)) {
      if (!is.null(metric)) {
        eval.metric <- metric$init()
      }
      while (eval.data$iter.next()) {
        dlist <- eval.data$value()
        slices <- lapply(1:ndevice, function(i) {
          s <- sliceinfo[[i]]
          ret <- list(data = mx.nd.slice(dlist$data, 
                                         s$begin, s$end), label = mx.nd.slice(dlist$label, 
                                                                              s$begin, s$end))
          return(ret)
        })
        for (i in 1:ndevice) {
          s <- slices[[i]]
          names(s) <- input.names
          mx.exec.update.arg.arrays(train.execs[[i]], 
                                    s, match.name = TRUE)
        }
        for (texec in train.execs) {
          mx.exec.forward(texec, is.train = FALSE)
        }
        out.preds <- lapply(train.execs, function(texec) {
          mx.nd.copyto(texec$ref.outputs[[1]], mx.cpu())
        })
        if (!is.null(metric)) {
          for (i in 1:ndevice) {
            eval.metric <- metric$update(slices[[i]]$label, 
                                         out.preds[[i]], eval.metric)
          }
        }
      }
      eval.data$reset()
      if (!is.null(metric)) {
        result <- metric$get(eval.metric)
        if (verbose) 
          cat(paste0("[", iteration, "] Validation-", 
                     result$name, "=", result$value, "\n"))
      }
    }
    else {
      eval.metric <- NULL
    }
    model <- mx.model.extract.model(symbol, train.execs)
    epoch_continue <- TRUE
    if (!is.null(epoch.end.callback)) {
      epoch_continue <- epoch.end.callback(iteration, 0, 
                                           environment(), verbose = verbose)
    }
    if (!epoch_continue) {
      break
    }
  }
  return(model)
}

