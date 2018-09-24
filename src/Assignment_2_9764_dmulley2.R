library(glmnet)
library(Matrix)
library(foreach)

rm(list = ls())
set.seed(9764)

T = 50

calculateRowIndex <- function (model_index, t) {
  ((model_index - 1) * T) + t
}

calculateMSPE <- function (Y, Y_hat) {
  mean((Y - Y_hat) ^ 2)
}

createAndRunModel.Full = function (housingData, testIds) {
  time.start = proc.time()
  full.model = lm(Y ~ ., data = housingData[-testIds,])
  Ytest.pred = predict(full.model, newdata = housingData[testIds,])
  time.end = proc.time()
  
  c(calculateMSPE(housingData$Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    dim(X)[2] - 1)
}

createAndRunModel.AICF = function (housingData, testIds) {
  time.start = proc.time()
  full.model = lm(Y ~ ., data = housingData[-testIds, ])
  stepAIC = step(lm(Y ~ 1, data = housingData[-testIds, ]),
                 list(upper = full.model),
                 trace = 0,
                 direction = "forward")
  Ytest.pred = predict(stepAIC, newdata = housingData[testIds, ])
  time.end = proc.time()
  
  c(calculateMSPE(housingData$Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    length(stepAIC$coef) - 1)
}

createAndRunModel.AICB = function (housingData, testIds) {
  time.start = proc.time()
  full.model = lm(Y ~ ., data = housingData[-testIds, ])
  stepAIC = step(full.model,
                 trace = 0,
                 direction = "backward")
  Ytest.pred = predict(stepAIC, newdata = housingData[testIds, ])
  time.end = proc.time()
  
  c(calculateMSPE(housingData$Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    length(stepAIC$coef) - 1)
}

createAndRunModel.BICF = function (housingData, testIds) {
  time.start = proc.time()
  full.model = lm(Y ~ ., data = housingData[-testIds, ])
  stepAIC = step(lm(Y ~ 1, data = housingData[-testIds, ]),
                 list(upper = full.model),
                 trace = 0, direction = "forward", k = log(ntrain))
  Ytest.pred = predict(stepAIC, newdata = housingData[testIds, ])
  time.end = proc.time()
  
  c(calculateMSPE(housingData$Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    length(stepAIC$coef) - 1)
}

createAndRunModel.BICB = function (housingData, testIds) {
  time.start = proc.time()
  full.model = lm(Y ~ ., data = housingData[-testIds, ])
  stepAIC = step(full.model, 
                 trace = 0,
                 direction = "backward", 
                 k = log(ntrain))
  Ytest.pred = predict(stepAIC, newdata = housingData[testIds, ])
  time.end = proc.time()
  
  c(calculateMSPE(housingData$Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    length(stepAIC$coef) - 1)
}

createAndRunModel.R_min = function (X, Y, testIds) {
  time.start = proc.time()
  cv.out = cv.glmnet(X[-testIds, ], Y[-testIds], alpha = 0)
  best.lam = cv.out$lambda.min
  Ytest.pred = predict(cv.out, 
                       s = best.lam, 
                       newx = X[testIds, ])
  time.end = proc.time()
  
  ntrain = n - dim(allTestIds)[1]
  tmpX = scale(X[-testIds, ]) * sqrt(ntrain / (ntrain - 1))
  d = svd(tmpX)$d
  
  c(calculateMSPE(Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"],
    sum(d^2/(d^2 + best.lam*ntrain)))
}

createAndRunModel.R_1se = function (X, Y, testIds) {
  time.start = proc.time()
  cv.out = cv.glmnet(X[-testIds, ], Y[-testIds], alpha = 0)
  best.lam = cv.out$lambda.1se
  Ytest.pred = predict(cv.out, 
                       s = best.lam, 
                       newx = X[testIds, ])
  time.end = proc.time()
  
  ntrain = n - dim(allTestIds)[1]
  tmpX = scale(X[-testIds, ]) * sqrt(ntrain / (ntrain - 1))
  d = svd(tmpX)$d
  
  c(calculateMSPE(Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    sum(d^2/(d^2 + best.lam*ntrain)))
}

createAndRunModel.L_min = function (X, Y, testIds) {
  time.start = proc.time()
  cv.out = cv.glmnet(X[-testIds, ], Y[-testIds], alpha = 1)
  best.lam = cv.out$lambda.min
  Ytest.pred = predict(cv.out, 
                       s = best.lam, 
                       newx = X[testIds, ])
  time.end = proc.time()
  
  mylasso.coef = predict(cv.out, s = best.lam, type = "coefficients")
  
  c(calculateMSPE(Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    sum(mylasso.coef != 0) - 1)
}

createAndRunModel.L_1se = function (X, Y, test.id) {
  time.start = proc.time()
  cv.out = cv.glmnet(X[-test.id, ], Y[-test.id], alpha = 1)
  best.lam = cv.out$lambda.1se
  Ytest.pred = predict(cv.out, 
                       s = best.lam, 
                       newx = X[test.id, ])
  time.end= proc.time()
  
  mylasso.coef = predict(cv.out, s = best.lam, type = "coefficients")
  
  c(calculateMSPE(Y[test.id], Ytest.pred),
    (time.end - time.start)["elapsed"], 
    sum(mylasso.coef != 0) - 1)
}

createAndRunModel.L_Refit = function (X, Y, testIds) {
  time.start = proc.time()
  cv.out = cv.glmnet(X[-testIds, ], Y[-testIds], alpha = 1)
  best.lam = cv.out$lambda.1se
  Ytest.pred = predict(cv.out, s = best.lam, newx = X[testIds, ])
  mylasso.coef = predict(cv.out, 
                         s = best.lam, 
                         type = "coefficients")
  
  var.sel = row.names(mylasso.coef)[nonzeroCoef(mylasso.coef)[-1]]
  tmp.X = X[, colnames(X) %in% var.sel]
  mylasso.refit = coef(lm(Y[-testIds] ~ tmp.X[-testIds, ]))
  Ytest.pred = mylasso.refit[1] + tmp.X[testIds,] %*% mylasso.refit[-1]
  time.end = proc.time()
  
  c(calculateMSPE(Y[testIds], Ytest.pred), 
    (time.end - time.start)["elapsed"], 
    sum(mylasso.coef != 0) - 1)
}

## Boston Housing 1

load('BostonHousing1.Rdata')

n = nrow(Housing1)
p = ncol(Housing1) - 1

X = data.matrix(Housing1[,-1])
Y = Housing1[,1]

ntest = round(n * 0.25)
ntrain = n - ntest

allTestIds = matrix(0, nrow = ntest, ncol = T)

for (t in 1:T) {
  allTestIds[,t] = sample(1:n, ntest)
}

model_names = c("Full", 
                "AIC.F", "AIC.B", 
                "BIC.F", "BIC.B", 
                "R.min", "R.1se", 
                "L.min", "L.1se",
                "L.Refit")

nmodels = length(model_names)

metrics = data.frame(rep(model_names, each = T), numeric(nmodels * T), numeric(nmodels * T), numeric(nmodels * T))
colnames(metrics) = c("model", "mspe", "time", "modelSize")


model.index = 1

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.Full(Housing1, allTestIds[,t])
}

model.index = 2

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.AICF(Housing1, allTestIds[,t])
}

model.index = 3

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.AICB(Housing1, allTestIds[,t])
}

model.index = 4

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.BICF(Housing1, allTestIds[,t])
}

model.index = 5

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.BICB(Housing1, allTestIds[,t])
}

model.index = 6

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_min(X, Y, allTestIds[, t])
}

model.index = 7

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_1se(X, Y, allTestIds[, t])
}

model.index = 8

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_min(X, Y, allTestIds[, t])
}

model.index = 9

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_1se(X, Y, allTestIds[, t])
}

model.index = 10

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_Refit(X, Y, allTestIds[, t])
}

boxplot(mspe ~ model, 
        data = metrics,
        main = "Boston housing 1 - Prediction error by method",
        xlab = "Method", 
        las = 2,
        ylab = "Prediction error (%)")

boxplot(modelSize ~ model,
        data = metrics,
        main = "Boston housing 1 - Model size by method",
        xlab = "Method",
        las = 2,
        ylab = "Model size")

boxplot(time ~ model,
        data = metrics,
        main = "Boston housing 1 - Computation time",
        xlab = "Method",
        las = 2,
        ylab = "Computation time (s)")

# Boston Housing 2
load('BostonHousing2.Rdata')

n = nrow(Housing2)
p = ncol(Housing2) - 1

X = data.matrix(Housing2[,-1])
Y = Housing2[,1]

ntest = round(n * 0.25)
ntrain = n - ntest

allTestIds = matrix(0, nrow = ntest, ncol = T)

for (t in 1:T) {
  allTestIds[,t] = sample(1:n, ntest)
}

model_names = c("R.min", "R.1se", 
                "L.min", "L.1se",
                "L.Refit")

nmodels = length(model_names)

metrics = data.frame(rep(model_names, each = T), numeric(nmodels * T), numeric(nmodels * T), numeric(nmodels * T))
colnames(metrics) = c("model", "mspe", "time", "modelSize")

model.index = 1

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_min(X, Y, allTestIds[, t])
}

model.index = 2

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_1se(X, Y, allTestIds[, t])
}

model.index = 3

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_min(X, Y, allTestIds[, t])
}

model.index = 4

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_1se(X, Y, allTestIds[, t])
}

model.index = 5

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_Refit(X, Y, allTestIds[, t])
}

boxplot(mspe ~ model, 
        data = metrics,
        main = "Boston housing 2 - Prediction error by method",
        xlab = "Method", 
        las = 2,
        ylab = "Prediction error")

boxplot(modelSize ~ model,
        data = metrics,
        main = "Boston housing 2 - Model size by method",
        xlab = "Method",
        las = 2,
        ylab = "Model size")

boxplot(time ~ model,
        data = metrics,
        main = "Boston housing 2 - Computation time",
        xlab = "Method",
        las = 2,
        ylab = "Computation time (s)")

# Boston Housing 3

load('BostonHousing3.Rdata')

n = nrow(Housing3)
p = ncol(Housing3) - 1

X = data.matrix(Housing3[,-1])
Y = Housing3[,1]

ntest = round(n * 0.25)
ntrain = n - ntest

allTestIds = matrix(0, nrow = ntest, ncol = T)

for (t in 1:T) {
  allTestIds[,t] = sample(1:n, ntest)
}

model_names = c("R.min", "R.1se", 
                "L.min", "L.1se",
                "L.Refit")

nmodels = length(model_names)

metrics = data.frame(rep(model_names, each = T), numeric(nmodels * T), numeric(nmodels * T), numeric(nmodels * T))
colnames(metrics) = c("model", "mspe", "time", "modelSize")

model.index = 1

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_min(X, Y, allTestIds[, t])
}

model.index = 2

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.R_1se(X, Y, allTestIds[, t])
}

model.index = 3

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_min(X, Y, allTestIds[, t])
}

model.index = 4

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_1se(X, Y, allTestIds[, t])
}

model.index = 5

for (t in 1:T) {
  metrics[calculateRowIndex(model.index, t), 2:4] = createAndRunModel.L_Refit(X, Y, allTestIds[, t])
}

boxplot(mspe ~ model, 
        data = metrics,
        main = "Boston housing 3 - Prediction error by method",
        xlab = "Method", 
        las = 2,
        ylab = "Prediction error")

boxplot(modelSize ~ model,
        data = metrics,
        main = "Boston housing 3 - Model size by method",
        xlab = "Method",
        las = 2,
        ylab = "Model size")

boxplot(time ~ model,
        data = metrics,
        main = "Boston housing 3 - Computation time",
        xlab = "Method",
        las = 2,
        ylab = "Computation time (s)")