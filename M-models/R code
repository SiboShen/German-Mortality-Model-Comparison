# install.packages("StMoMo")
# install.packages("devtools")
# install.packages("StMoMo", repos = c('https://amvillegas.r-universe.dev', 'https://cloud.r-project.org'))
# install.packages("gnm")
# install.packages("patchwork")
# install.packages("reshape2")
# install.packages("magick")
library(gnm)
library(StMoMo)
library(demography)  # If you need to download data from HMD
library(tidyr)  # For data tidying
library(data.table)
library(dplyr) #For data processing
library(ggplot2)    # For plotting
library(magrittr)   # For %>% operator
library(ggplot2)
library(patchwork)
library(reshape2)
library(magick)

expo_total<-read.table("/Users/sibo/Desktop/Data/expo_total_18.csv",sep=',',header=TRUE)
deat_total<-read.table("/Users/sibo/Desktop/Data/deat_total_18.csv",sep=',',header=TRUE)

ages <- 18:100
years <- 1956:2020
ages <- as.numeric(ages)
years <- as.numeric(years)

# Construct female data
Dxt_F <- deat_total %>%
  select(Year, Age, Female) %>%
  pivot_wider(names_from = Year, values_from = Female) %>%
  arrange(Age) %>%
  select(-Age) %>%
  as.matrix()
Ec_F <- expo_total %>%
  select(Year, Age, Female) %>%
  pivot_wider(names_from = Year, values_from = Female) %>%
  arrange(Age) %>%
  select(-Age) %>%
  as.matrix()
E0_F <- Ec_F + 0.5 * Dxt_F

# Construct male data
Dxt_M <- deat_total %>%
  select(Year, Age, Male) %>%
  pivot_wider(names_from = Year, values_from = Male) %>%
  arrange(Age) %>%
  select(-Age) %>%
  as.matrix()
Ec_M <- expo_total %>%
  select(Year, Age, Male) %>%
  pivot_wider(names_from = Year, values_from = Male) %>%
  arrange(Age) %>%
  select(-Age) %>%
  as.matrix()
E0_M <- Ec_M + 0.5 * Dxt_M

if (any(!is.finite(E0_F)) || any(!is.finite(E0_M)) ||
    any(E0_F <= 0) || any(E0_M <= 0) ||
    any(Dxt_F < 0 | Dxt_F > E0_F) ||
    any(Dxt_M < 0 | Dxt_M > E0_M)) {
  stop("Invalid deaths or initial exposures in the mortality data.")
}

# StMoMoData(type = "initial") expects central exposure in the demogdata
# object and internally sets Ext = Ec + 0.5 * D. Therefore Ec, not E0, is
# supplied below; the resulting StMoMoData objects contain initial exposure.

# Create female demogdata object
myGermanyDemo_F <- list(
  year = years,
  age = ages,
  rate = list(Female = Dxt_F / Ec_F),
  pop  = list(Female = Ec_F),
  type = "mortality",
  label = "Germany Female"
) 
class(myGermanyDemo_F) <- "demogdata"

myGermanyData_F <- StMoMoData(
  data = myGermanyDemo_F,
  series = "Female",
  type = "initial"
)

# Create male demogdata object
myGermanyDemo_M <- list(
  year = years,
  age = ages,
  rate = list(Male = Dxt_M / Ec_M),
  pop  = list(Male = Ec_M),
  type = "mortality",
  label = "Germany Male"
) 
class(myGermanyDemo_M) <- "demogdata"

myGermanyData_M <- StMoMoData(
  data = myGermanyDemo_M,
  series = "Male",
  type = "initial"
)

# Define classic models
M1 <- lc(link = "logit")
M2 <- rh(link = "logit", cohortAgeFun = "1") # cohortAgeFun means the parameter bete(0)_x, here defines that bete(0)_x = 1.
M5 <- cbd(link = "logit")
M6 <- m6(link = "logit")

models_to_fit <- list(M1 = M1, M2 = M2, M5 = M5, M6 = M6)

# Fit female models
model_fits_F <- list()
for (i in 1:length(models_to_fit)) {
  model_name <- names(models_to_fit)[i]
  model_fits_F[[model_name]] <- fit(models_to_fit[[i]], data = myGermanyData_F)
}

# Fit male models
model_fits_M <- list()
for (i in 1:length(models_to_fit)) {
  model_name <- names(models_to_fit)[i]
  model_fits_M[[model_name]] <- fit(models_to_fit[[i]], data = myGermanyData_M)
}

M1_fit_F <- model_fits_F[["M1"]]
M1_fit_M <- model_fits_M[["M1"]]
pdf("/Users/sibo/Desktop/Data/M1_fit_F.pdf", width = 9, height = 5.5)
beta1_F <- M1_fit_F$ax          # β1(x)
beta2_F <- M1_fit_F$bx[,1]      # β2(x)
kappa2_F <- M1_fit_F$kt[1,]     # κ2(t)
par(mfrow=c(2,2))

plot(M1_fit_F$ages, beta1_F, type="l",
     main=expression(beta[1](x)~"vs."~x),
     xlab="age", ylab="")

plot(M1_fit_F$ages, beta2_F, type="l",
     main=expression(beta[2](x)~"vs."~x),
     xlab="age", ylab="")

plot(M1_fit_F$years, kappa2_F, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab="")
dev.off()
pdf("/Users/sibo/Desktop/Data/M1_fit_M.pdf", width = 9, height = 5.5)
beta1_M <- M1_fit_M$ax          # β1(x)
beta2_M <- M1_fit_M$bx[,1]      # β2(x)
kappa2_M <- M1_fit_M$kt[1,]     # κ2(t)
par(mfrow=c(2,2))

plot(M1_fit_M$ages, beta1_M, type="l",
     main=expression(beta[1](x)~"vs."~x),
     xlab="age", ylab="")

plot(M1_fit_M$ages, beta2_M, type="l",
     main=expression(beta[2](x)~"vs."~x),
     xlab="age", ylab="")

plot(M1_fit_M$years, kappa2_M, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab="")
dev.off()

M2_fit_F <- model_fits_F[["M2"]]
pdf("/Users/sibo/Desktop/Data/M2_fit_F.pdf", width = 9, height = 5.5)

beta1_F  <- M2_fit_F$ax          # β1(x)
beta2_F  <- M2_fit_F$bx[,1]      # β2(x)
kappa2_F <- M2_fit_F$kt[1,]      # κ2(t)
gamma3_F <- M2_fit_F$gc          # γ3(t-x)

par(mfrow=c(2,2))

plot(M2_fit_F$ages, beta1_F, type="l",
     main=expression(beta[1](x)~"vs."~x),
     xlab="age", ylab="")

plot(M2_fit_F$ages, beta2_F, type="l",
     main=expression(beta[2](x)~"vs."~x),
     xlab="age", ylab="")

plot(M2_fit_F$years, kappa2_F, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="age", ylab="")

plot(M2_fit_F$cohorts, gamma3_F, type="l",
     main=expression(gamma[3](t-x)~"vs."~(t-x)),
     xlab="cohort")
dev.off()

M2_fit_M <- model_fits_M[["M2"]]
pdf("/Users/sibo/Desktop/Data/M2_fit_M.pdf", width = 9, height = 5.5)
beta1_M  <- M2_fit_M$ax          # β1(x)
beta2_M  <- M2_fit_M$bx[,1]      # β2(x)
kappa2_M <- M2_fit_M$kt[1,]      # κ2(t)
gamma3_M <- M2_fit_M$gc          # γ3(t-x)

par(mfrow=c(2,2))

plot(M2_fit_M$ages, beta1_M, type="l",
     main=expression(beta[1](x)~"vs."~x),
     xlab="age", ylab="")

plot(M2_fit_M$ages, beta2_M, type="l",
     main=expression(beta[2](x)~"vs."~x),
     xlab="age", ylab="")

plot(M2_fit_M$years, kappa2_M, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="age", ylab="")

plot(M2_fit_M$cohorts, gamma3_M, type="l",
     main=expression(gamma[3](t-x)~"vs."~(t-x)),
     xlab="cohort")

dev.off()

M5_fit_F <- model_fits_F[["M5"]]
kappa1_F <- M5_fit_F$kt[1,]   # κ1(t)
kappa2_F <- M5_fit_F$kt[2,]   # κ2(t)
pdf("/Users/sibo/Desktop/Data/M5_fit_F.pdf", width = 9, height = 5.5)
par(mfrow=c(2,2))

# κ1(t)
plot(M5_fit_F$years, kappa1_F, type="l",
     main=expression(kappa[1](t)~"vs."~t),
     xlab="year", ylab = "")

# κ2(t)
plot(M5_fit_F$years, kappa2_F, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab = "")

# (x - x̄)
plot(M5_fit_F$ages, M5_fit_F$ages - mean(M5_fit_F$ages), type="l",
     main=expression((x-bar(x))~"vs."~x),
     xlab="age", ylab = "")
dev.off()

M5_fit_M <- model_fits_M[["M5"]]
kappa1_M <- M5_fit_M$kt[1,]   # κ1(t)
kappa2_M <- M5_fit_M$kt[2,]   # κ2(t)
pdf("/Users/sibo/Desktop/Data/M5_fit_M.pdf", width = 9, height = 5.5)
par(mfrow=c(2,2))

# κ1(t)
plot(M5_fit_M$years, kappa1_M, type="l",
     main=expression(kappa[1](t)~"vs."~t),
     xlab="year", ylab = "")

# κ2(t)
plot(M5_fit_M$years, kappa2_M, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab = "")

# (x - x̄)
plot(M5_fit_M$ages, M5_fit_M$ages - mean(M5_fit_M$ages), type="l",
     main=expression((x-bar(x))~"vs."~x),
     xlab="age", ylab = "")
dev.off()

M6_fit_F <- model_fits_F[["M6"]]
kappa1_F <- M6_fit_F$kt[1,]   # κ1(t)
kappa2_F <- M6_fit_F$kt[2,]   # κ2(t)
gamma3_F <- M6_fit_F$gc       # γ3(t-x)

pdf("/Users/sibo/Desktop/Data/M6_fit_F.pdf", width = 9, height = 5.5)
par(mfrow=c(2,2))

# κ1(t)
plot(M6_fit_F$years, kappa1_F, type="l",
     main=expression(kappa[1](t)~"vs."~t),
     xlab="year", ylab = "")

# κ2(t)
plot(M6_fit_F$years, kappa2_F, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab = "")

# (x - x̄)
plot(M6_fit_F$ages, M6_fit_F$ages - mean(M6_fit_F$ages), type="l",
     main=expression((x-bar(x))~"vs."~x),
     xlab="age", ylab = "")

# γ3(t-x)
plot(M6_fit_F$cohorts, gamma3_F, type="l",
     main=expression(gamma[3](t-x)~"vs."~(t-x)),
     xlab="cohort", ylab = "")
dev.off()

M6_fit_M <- model_fits_M[["M6"]]
kappa1_M <- M6_fit_M$kt[1,]
kappa2_M <- M6_fit_M$kt[2,]
gamma3_M <- M6_fit_M$gc
pdf("/Users/sibo/Desktop/Data/M6_fit_M.pdf", width = 9, height = 5.5)
par(mfrow=c(2,2))

plot(M6_fit_M$years, kappa1_M, type="l",
     main=expression(kappa[1](t)~"vs."~t),
     xlab="year", ylab = "")

plot(M6_fit_M$years, kappa2_M, type="l",
     main=expression(kappa[2](t)~"vs."~t),
     xlab="year", ylab = "")

plot(M6_fit_M$ages, M6_fit_M$ages - mean(M6_fit_M$ages), type="l",
     main=expression((x-bar(x))~"vs."~x),
     xlab="age", ylab = "")

plot(M6_fit_M$cohorts, gamma3_M, type="l",
     main=expression(gamma[3](t-x)~"vs."~(t-x)),
     xlab="cohort", ylab = "")
dev.off()
# Calculate number of observations
N_observations <- length(ages) * length(years)

# Goodness of fit
# Extract female model results
goodness_F <- data.frame(
  Model = paste0(names(model_fits_F), "_Female"),
  Maximum_Log_Likelihood = round(sapply(model_fits_F, logLik), 2),
  Effective_Parameters = sapply(model_fits_F, function(fit) fit$npar),
  Observations = N_observations,
  AIC = round(sapply(model_fits_F, AIC), 2),
  BIC = round(sapply(model_fits_F, BIC), 2)
)

# Extract male model results
goodness_M <- data.frame(
  Model = paste0(names(model_fits_M), "_Male"),
  Maximum_Log_Likelihood = round(sapply(model_fits_M, logLik), 2),
  Effective_Parameters = sapply(model_fits_M, function(fit) fit$npar),
  Observations = N_observations,
  AIC = round(sapply(model_fits_M, AIC), 2),
  BIC = round(sapply(model_fits_M, BIC), 2)
)

# Combine results
goodness <- rbind(goodness_F, goodness_M)
print(goodness)

# Save results to CSV file
write.csv(goodness, "/Users/sibo/Desktop/Data/goodness_of_fit_r.csv", row.names = FALSE)

# Robustness estimation
years_long  <- 1956:2020
years_short <- 1956:2000
ages.fit    <- 18:100

# Base-R robustness helper: solid blue = long period, dashed red = short period
plot_rob <- function(x_long, y_long, x_short, y_short,
                     title_expr, xlab_str) {
  ylim <- range(c(y_long, y_short), na.rm = TRUE)
  xlim <- range(c(x_long, x_short), na.rm = TRUE)
  plot(x_long, y_long,
       type = "l", col = "blue", lwd = 0.8,
       xlim = xlim, ylim = ylim,
       main = title_expr, xlab = xlab_str, ylab = "",
       cex.main = 0.9, cex.axis = 0.7, cex.lab = 0.85,
       bty = "o", tcl = -0.3)
  lines(x_short, y_short, col = "red", lty = 2, lwd = 0.8)
}

# M1 (Lee-Carter logit): beta2(x), kappa2(t)
M1_model   <- lc(link = "logit")
M1_long_F  <- fit(M1_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_long)
M1_short_F <- fit(M1_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_short)
M1_long_M  <- fit(M1_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_long)
M1_short_M <- fit(M1_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_short)

pdf(
  "/Users/sibo/Desktop/Data/robustness_M1.pdf",
  width = 9, height = 5.5, onefile = FALSE
)
par(mfrow = c(2, 2), mar = c(4, 2.5, 2.5, 1))
plot_rob(ages.fit,   as.vector(M1_long_M$bx),  ages.fit,    as.vector(M1_short_M$bx),
         expression(beta[2](x)~"vs."~x~"(Male)"),    "age")
plot_rob(years_long, as.vector(M1_long_M$kt),  years_short, as.vector(M1_short_M$kt),
         expression(kappa[2](t)~"vs."~t~"(Male)"),   "year")
plot_rob(ages.fit,   as.vector(M1_long_F$bx),  ages.fit,    as.vector(M1_short_F$bx),
         expression(beta[2](x)~"vs."~x~"(Female)"),  "age")
plot_rob(years_long, as.vector(M1_long_F$kt),  years_short, as.vector(M1_short_F$kt),
         expression(kappa[2](t)~"vs."~t~"(Female)"), "year")
dev.off()

# M2 (Renshaw-Haberman logit): beta2(x), kappa2(t), gamma3(t-x)
M2_model   <- rh(link = "logit", cohortAgeFun = "1")
M2_long_F  <- fit(M2_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_long)
M2_short_F <- fit(M2_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_short)
M2_long_M  <- fit(M2_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_long)
M2_short_M <- fit(M2_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_short)

pdf(
  "/Users/sibo/Desktop/Data/robustness_M2.pdf",
  width = 9, height = 5.5, onefile = FALSE
)
par(mfrow = c(2, 3), mar = c(4, 2.5, 2.5, 1))
plot_rob(ages.fit,   as.vector(M2_long_M$bx),
         ages.fit,   as.vector(M2_short_M$bx),
         expression(beta[2](x)~"vs."~x~"(Male)"),         "age")
plot_rob(years_long, colMeans(M2_long_M$kt, na.rm = TRUE),
         years_short, colMeans(M2_short_M$kt, na.rm = TRUE),
         expression(kappa[2](t)~"vs."~t~"(Male)"),        "year")
plot_rob(as.numeric(names(M2_long_M$gc)),  as.vector(M2_long_M$gc),
         as.numeric(names(M2_short_M$gc)), as.vector(M2_short_M$gc),
         expression(gamma[3](t-x)~"vs."~(t-x)~"(Male)"), "cohort")
plot_rob(ages.fit,   as.vector(M2_long_F$bx),
         ages.fit,   as.vector(M2_short_F$bx),
         expression(beta[2](x)~"vs."~x~"(Female)"),         "age")
plot_rob(years_long, colMeans(M2_long_F$kt, na.rm = TRUE),
         years_short, colMeans(M2_short_F$kt, na.rm = TRUE),
         expression(kappa[2](t)~"vs."~t~"(Female)"),        "year")
plot_rob(as.numeric(names(M2_long_F$gc)),  as.vector(M2_long_F$gc),
         as.numeric(names(M2_short_F$gc)), as.vector(M2_short_F$gc),
         expression(gamma[3](t-x)~"vs."~(t-x)~"(Female)"), "cohort")
dev.off()

# M5 (CBD logit): kappa1(t), kappa2(t)
M5_model   <- cbd(link = "logit")
M5_long_F  <- fit(M5_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_long)
M5_short_F <- fit(M5_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_short)
M5_long_M  <- fit(M5_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_long)
M5_short_M <- fit(M5_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_short)

pdf(
  "/Users/sibo/Desktop/Data/robustness_M5.pdf",
  width = 9, height = 5.5, onefile = FALSE
)
par(mfrow = c(2, 2), mar = c(4, 2.5, 2.5, 1))
plot_rob(years_long, as.vector(M5_long_M$kt[1, ]),  years_short, as.vector(M5_short_M$kt[1, ]),
         expression(kappa[1](t)~"vs."~t~"(Male)"),   "year")
plot_rob(years_long, as.vector(M5_long_M$kt[2, ]),  years_short, as.vector(M5_short_M$kt[2, ]),
         expression(kappa[2](t)~"vs."~t~"(Male)"),   "year")
plot_rob(years_long, as.vector(M5_long_F$kt[1, ]),  years_short, as.vector(M5_short_F$kt[1, ]),
         expression(kappa[1](t)~"vs."~t~"(Female)"), "year")
plot_rob(years_long, as.vector(M5_long_F$kt[2, ]),  years_short, as.vector(M5_short_F$kt[2, ]),
         expression(kappa[2](t)~"vs."~t~"(Female)"), "year")
dev.off()

# M6 (CBD + cohort logit): kappa1(t), kappa2(t), gamma3(t-x)
M6_model   <- m6(link = "logit")
M6_long_F  <- fit(M6_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_long)
M6_short_F <- fit(M6_model, data = myGermanyData_F, ages.fit = ages.fit, years.fit = years_short)
M6_long_M  <- fit(M6_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_long)
M6_short_M <- fit(M6_model, data = myGermanyData_M, ages.fit = ages.fit, years.fit = years_short)

pdf(
  "/Users/sibo/Desktop/Data/robustness_M6.pdf",
  width = 9, height = 5.5, onefile = FALSE
)
par(mfrow = c(2, 3), mar = c(4, 2.5, 2.5, 1))
plot_rob(years_long, as.vector(M6_long_M$kt[1, ]),  years_short, as.vector(M6_short_M$kt[1, ]),
         expression(kappa[1](t)~"vs."~t~"(Male)"),        "year")
plot_rob(years_long, as.vector(M6_long_M$kt[2, ]),  years_short, as.vector(M6_short_M$kt[2, ]),
         expression(kappa[2](t)~"vs."~t~"(Male)"),        "year")
plot_rob(as.numeric(names(M6_long_M$gc)),  as.vector(M6_long_M$gc),
         as.numeric(names(M6_short_M$gc)), as.vector(M6_short_M$gc),
         expression(gamma[3](t-x)~"vs."~(t-x)~"(Male)"),  "cohort")
plot_rob(years_long, as.vector(M6_long_F$kt[1, ]),  years_short, as.vector(M6_short_F$kt[1, ]),
         expression(kappa[1](t)~"vs."~t~"(Female)"),       "year")
plot_rob(years_long, as.vector(M6_long_F$kt[2, ]),  years_short, as.vector(M6_short_F$kt[2, ]),
         expression(kappa[2](t)~"vs."~t~"(Female)"),       "year")
plot_rob(as.numeric(names(M6_long_F$gc)),  as.vector(M6_long_F$gc),
         as.numeric(names(M6_short_F$gc)), as.vector(M6_short_F$gc),
         expression(gamma[3](t-x)~"vs."~(t-x)~"(Female)"), "cohort")
dev.off()

# Deviance residuals
out_dir <- "/Users/sibo/Desktop/Data"
model_names <- c("M1", "M2", "M5", "M6")

if (!all(model_names %in% names(model_fits_M)) ||
    !all(model_names %in% names(model_fits_F))) {
  stop("model_fits_M and model_fits_F must contain M1, M2, M5 and M6.")
}


# Calculate the residual objects once; the fitted models are not re-estimated.
residuals_by_model <- setNames(
  lapply(model_names, function(model_name) {
    list(
      Male = residuals(model_fits_M[[model_name]], type = "deviance"),
      Female = residuals(model_fits_F[[model_name]], type = "deviance")
    )
  }),
  model_names
)


residual_to_long <- function(res_obj) {
  mat <- res_obj$residuals
  ages <- res_obj$ages
  years <- res_obj$years
  
  expected_dim <- c(length(ages), length(years))
  if (!identical(dim(mat), expected_dim)) {
    stop("The dimensions of the residual matrix do not match ages and years.")
  }
  
  dat <- expand.grid(
    age = ages,
    year = years,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  dat$residual <- as.vector(mat)
  dat$cohort <- dat$year - dat$age
  
  dat[is.finite(dat$residual), , drop = FALSE]
}


plot_residual_row <- function(res_obj, colour, sex_label, residual_ylim) {
  dat <- residual_to_long(res_obj)
  
  x_values <- list(dat$age, dat$year, dat$cohort)
  x_labels <- c("Age", "Calendar year", "Year of birth")
  
  for (panel in seq_along(x_values)) {
    plot(
      x_values[[panel]],
      dat$residual,
      pch = 16,
      cex = 0.30,
      col = grDevices::adjustcolor(colour, alpha.f = 0.70),
      ylim = residual_ylim,
      xlab = x_labels[panel],
      ylab = if (panel == 1L) paste0("Residuals (", sex_label, ")") else ""
    )
    abline(h = 0, col = "black", lwd = 0.7, lty = 2)
    box()
  }
}

all_residual_values <- unlist(
  lapply(residuals_by_model, function(model_residuals) {
    c(
      model_residuals$Male$residuals,
      model_residuals$Female$residuals
    )
  }),
  use.names = FALSE
)
all_residual_values <- all_residual_values[is.finite(all_residual_values)]

if (length(all_residual_values) == 0L) {
  stop("No finite residual values were found.")
}

residual_limit <- max(
  3.5,
  ceiling(2 * max(abs(all_residual_values))) / 2
)
shared_ylim <- c(-residual_limit, residual_limit)

save_combined_residual_plot <- function(model_name, residual_pair) {
  grDevices::pdf(
    file.path(out_dir, paste0("residuals_", model_name, ".pdf")),
    width = 9,
    height = 5.5
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  
  par(
    mfrow = c(2, 3),
    mar = c(3.4, 3.5, 0.8, 0.8),
    oma = c(0, 0, 0, 0),
    mgp = c(2.0, 0.65, 0),
    tcl = -0.25,
    las = 1
  )
  
  plot_residual_row(
    residual_pair$Male,
    colour = "blue",
    sex_label = "Male",
    residual_ylim = shared_ylim
  )
  plot_residual_row(
    residual_pair$Female,
    colour = "red",
    sex_label = "Female",
    residual_ylim = shared_ylim
  )
}
invisible(
  lapply(model_names, function(model_name) {
    save_combined_residual_plot(
      model_name,
      residuals_by_model[[model_name]]
    )
  })
)
# =============================================================================
# Chapter 6
# =============================================================================

out_dir     <- "/Users/sibo/Desktop/Data"
train_years <- 1956:2010
val_years   <- 2011:2020
ages_fc     <- 18:100
h_oos       <- length(val_years)

quietly <- function(expr) {
  capture.output(result <- suppressMessages(suppressWarnings(force(expr))))
  result
}

format_arima <- function(model) {
  order <- as.integer(forecast::arimaorder(model))
  label <- sprintf("ARIMA(%d,%d,%d)", order[1], order[2], order[3])
  coefficients <- names(stats::coef(model))
  if ("drift" %in% coefficients) label <- paste(label, "with drift")
  if ("intercept" %in% coefficients && order[2] == 0L) label <- paste(label, "with mean")
  label
}

select_cohort_arima <- function(fit_obj) {
  gamma <- fit_obj$gc[is.finite(fit_obj$gc)]
  forecast::auto.arima(stats::ts(gamma), seasonal = FALSE)
}

forecast_stmomo <- function(fit_obj, cohort_model, jump_off) {
  if (is.null(cohort_model)) {
    return(quietly(forecast::forecast(
      fit_obj, h = h_oos, kt.method = "iarima", jumpchoice = jump_off
    )))
  }
  
  cohort_order <- as.integer(forecast::arimaorder(cohort_model)[1:3])
  cohort_constant <- any(c("drift", "intercept") %in% names(stats::coef(cohort_model)))
  quietly(forecast::forecast(
    fit_obj, h = h_oos, kt.method = "iarima",
    gc.order = cohort_order, gc.include.constant = cohort_constant,
    jumpchoice = jump_off
  ))
}

extract_arima_orders <- function(fc, model_name, sex, cohort_model) {
  period_models <- fc$kt.f$model$models
  period_rows <- lapply(seq_along(period_models), function(i) {
    data.frame(
      Model = model_name,
      Sex = sex,
      Index = if (length(period_models) == 1L) "kappa" else paste0("kappa_", i),
      ARIMA = format_arima(period_models[[i]]),
      stringsAsFactors = FALSE
    )
  })
  
  if (!is.null(cohort_model)) {
    period_rows[[length(period_rows) + 1L]] <- data.frame(
      Model = model_name, Sex = sex, Index = "gamma",
      ARIMA = format_arima(cohort_model), stringsAsFactors = FALSE
    )
  }
  do.call(rbind, period_rows)
}

compute_naive_mae <- function(training_q) {
  if (ncol(training_q) < 2L || any(!is.finite(training_q))) {
    stop("Training probabilities must be finite and contain at least two years.")
  }
  naive_errors <- abs(
    training_q[, 2:ncol(training_q), drop = FALSE] -
      training_q[, 1:(ncol(training_q) - 1L), drop = FALSE]
  )
  naive_mae <- mean(naive_errors)
  if (!is.finite(naive_mae) || naive_mae <= 0) {
    stop("The observed-data naive benchmark must be positive.")
  }
  naive_mae
}

compute_mae_mase <- function(observed, forecasted, naive_mae) {
  if (!identical(dim(observed), dim(forecasted)) ||
      any(!is.finite(observed)) || any(!is.finite(forecasted))) {
    stop("Observed and forecast probability matrices must be finite and aligned.")
  }
  mae_raw <- mean(abs(forecasted - observed))
  list(
    MAE = 100 * mae_raw,
    MASE = mae_raw / naive_mae
  )
}

data_list <- list(
  Male   = myGermanyData_M,
  Female = myGermanyData_F
)

arima_rows <- list()
error_rows <- list()

for (sex in names(data_list)) {
  observed_q <- data_list[[sex]]$Dxt / data_list[[sex]]$Ext
  training_q <- observed_q[
    as.character(ages_fc), as.character(train_years), drop = FALSE
  ]
  validation_q <- observed_q[
    as.character(ages_fc), as.character(val_years), drop = FALSE
  ]
  naive_mae <- compute_naive_mae(training_q)
  
  for (model_name in names(models_to_fit)) {
    fit_obj <- quietly(fit(
      models_to_fit[[model_name]], data = data_list[[sex]],
      ages.fit = ages_fc, years.fit = train_years
    ))
    cohort_model <- if (model_name %in% c("M2", "M6")) {
      quietly(select_cohort_arima(fit_obj))
    } else {
      NULL
    }
    
    fc_actual <- forecast_stmomo(fit_obj, cohort_model, jump_off = "actual")
    fc_fitted <- forecast_stmomo(fit_obj, cohort_model, jump_off = "fit")
    
    arima_rows[[length(arima_rows) + 1L]] <- extract_arima_orders(
      fc_actual, model_name, sex, cohort_model
    )
    
    for (jump_off in c("actual", "fitted")) {
      fc_obj <- if (jump_off == "actual") fc_actual else fc_fitted
      forecasted_q <- fc_obj$rates[
        as.character(ages_fc), as.character(val_years), drop = FALSE
      ]
      accuracy <- compute_mae_mase(validation_q, forecasted_q, naive_mae)
      error_rows[[length(error_rows) + 1L]] <- data.frame(
        Model = model_name,
        Sex = sex,
        JumpOff = jump_off,
        MAE = accuracy$MAE,
        MASE = accuracy$MASE,
        stringsAsFactors = FALSE
      )
    }
  }
}

arima_orders <- do.call(rbind, arima_rows)
stmomo_oos_errors <- do.call(rbind, error_rows)
arima_orders <- arima_orders[
  order(arima_orders$Sex, arima_orders$Model, arima_orders$Index),
]
stmomo_oos_errors <- stmomo_oos_errors[
  order(
    stmomo_oos_errors$Sex,
    stmomo_oos_errors$Model,
    stmomo_oos_errors$JumpOff
  ),
]
row.names(arima_orders) <- NULL
row.names(stmomo_oos_errors) <- NULL
stmomo_oos_errors$MAE <- round(stmomo_oos_errors$MAE, 4)
stmomo_oos_errors$MASE <- round(stmomo_oos_errors$MASE, 4)

write.csv(arima_orders, file.path(out_dir, "arima_orders.csv"), row.names = FALSE)
write.csv(stmomo_oos_errors, file.path(out_dir, "stmomo_oos_errors.csv"), row.names = FALSE)

# =============================================================================
# Chapter 6.2 -- Conditional long-term projections, 2021--2040
# =============================================================================

proj_full_years <- 1956:2020
proj_years      <- 2021:2040
proj_ref_ages   <- c(65L, 75L, 85L)
proj_horizon    <- length(proj_years)
proj_nsim       <- 1000L
proj_probs      <- c(0.025, 0.10, 0.25, 0.75, 0.90, 0.975)

forecast_stmomo_projection <- function(fit_obj, cohort_model) {
  if (is.null(cohort_model)) {
    return(quietly(forecast::forecast(
      fit_obj,
      h = proj_horizon,
      kt.method = "iarima",
      jumpchoice = "actual"
    )))
  }
  
  cohort_order <- as.integer(forecast::arimaorder(cohort_model)[1:3])
  cohort_constant <- any(
    c("drift", "intercept") %in% names(stats::coef(cohort_model))
  )
  quietly(forecast::forecast(
    fit_obj,
    h = proj_horizon,
    kt.method = "iarima",
    gc.order = cohort_order,
    gc.include.constant = cohort_constant,
    jumpchoice = "actual"
  ))
}

simulate_stmomo_projection <- function(fit_obj, cohort_model, seed) {
  simulation_arguments <- list(
    object = fit_obj,
    nsim = proj_nsim,
    seed = seed,
    h = proj_horizon,
    kt.method = "iarima",
    jumpchoice = "actual"
  )
  
  if (!is.null(cohort_model)) {
    simulation_arguments$gc.order <- as.integer(
      forecast::arimaorder(cohort_model)[1:3]
    )
    simulation_arguments$gc.include.constant <- any(
      c("drift", "intercept") %in% names(stats::coef(cohort_model))
    )
  }
  
  quietly(do.call(stats::simulate, simulation_arguments))
}

extract_projection_arima_orders <- function(
    central_forecast, model_name, sex, cohort_model) {
  period_models <- central_forecast$kt.f$model$models
  rows <- lapply(seq_along(period_models), function(index) {
    data.frame(
      Model = model_name,
      Sex = sex,
      Index = if (length(period_models) == 1L) {
        "kappa"
      } else {
        paste0("kappa_", index)
      },
      ARIMA = format_arima(period_models[[index]]),
      stringsAsFactors = FALSE
    )
  })
  
  if (!is.null(cohort_model)) {
    rows[[length(rows) + 1L]] <- data.frame(
      Model = model_name,
      Sex = sex,
      Index = "gamma",
      ARIMA = format_arima(cohort_model),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

summarise_stmomo_projection <- function(
    central_forecast, simulation, model_name, sex) {
  age_names <- as.character(ages_fc)
  year_names <- as.character(proj_years)
  
  central_rates <- central_forecast$rates[
    age_names, year_names, drop = FALSE
  ]
  simulated_rates <- simulation$rates[
    age_names, year_names, , drop = FALSE
  ]
  
  quantiles <- apply(
    simulated_rates,
    MARGIN = c(1, 2),
    FUN = stats::quantile,
    probs = proj_probs,
    na.rm = TRUE,
    names = FALSE,
    type = 8
  )
  
  grid <- expand.grid(
    Age = ages_fc,
    Year = proj_years,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  data.frame(
    Model = model_name,
    Sex = sex,
    Age = grid$Age,
    Year = grid$Year,
    Central = as.vector(central_rates),
    Lower_50 = as.vector(quantiles[3, , ]),
    Upper_50 = as.vector(quantiles[4, , ]),
    Lower_80 = as.vector(quantiles[2, , ]),
    Upper_80 = as.vector(quantiles[5, , ]),
    Lower_95 = as.vector(quantiles[1, , ]),
    Upper_95 = as.vector(quantiles[6, , ]),
    stringsAsFactors = FALSE
  )
}

plot_stmomo_projection_pdf <- function(model_name, projection_summary) {
  colour_95 <- grDevices::adjustcolor("#4C78A8", alpha.f = 0.16)
  colour_80 <- grDevices::adjustcolor("#4C78A8", alpha.f = 0.28)
  colour_50 <- grDevices::adjustcolor("#4C78A8", alpha.f = 0.43)
  
  grDevices::pdf(
    file.path(out_dir, paste0("projection_", model_name, ".pdf")),
    width = 9,
    height = 5.5,
    onefile = FALSE
  )
  on.exit(grDevices::dev.off(), add = TRUE)
  graphics::par(
    mfrow = c(2, 3),
    mar = c(3.3, 3.5, 1.8, 0.8),
    oma = c(2.3, 0, 0, 0),
    mgp = c(2.0, 0.65, 0),
    tcl = -0.25,
    las = 1
  )
  
  for (sex in c("Male", "Female")) {
    observed_q <- data_list[[sex]]$Dxt / data_list[[sex]]$Ext
    
    for (age in proj_ref_ages) {
      panel <- projection_summary[
        projection_summary$Model == model_name &
          projection_summary$Sex == sex &
          projection_summary$Age == age,
      ]
      panel <- panel[order(panel$Year), ]
      historical <- as.numeric(observed_q[
        as.character(age), as.character(proj_full_years)
      ])
      
      y_limits <- range(
        historical,
        panel$Lower_95,
        panel$Upper_95,
        na.rm = TRUE
      )
      graphics::plot(
        proj_full_years,
        historical,
        type = "l",
        col = "grey55",
        lwd = 0.8,
        xlim = range(c(proj_full_years, proj_years)),
        ylim = y_limits,
        xlab = "Year",
        ylab = expression(q[x * "," * t]),
        main = paste0(sex, ", age ", age)
      )
      graphics::polygon(
        c(panel$Year, rev(panel$Year)),
        c(panel$Lower_95, rev(panel$Upper_95)),
        col = colour_95,
        border = NA
      )
      graphics::polygon(
        c(panel$Year, rev(panel$Year)),
        c(panel$Lower_80, rev(panel$Upper_80)),
        col = colour_80,
        border = NA
      )
      graphics::polygon(
        c(panel$Year, rev(panel$Year)),
        c(panel$Lower_50, rev(panel$Upper_50)),
        col = colour_50,
        border = NA
      )
      graphics::lines(
        panel$Year,
        panel$Central,
        col = "#1F4E79",
        lwd = 1.2
      )
      graphics::abline(v = 2020.5, col = "firebrick", lty = 3, lwd = 0.8)
    }
  }
  
  graphics::par(
    fig = c(0, 1, 0, 1),
    mar = c(0, 0, 0, 0),
    oma = c(0, 0, 0, 0),
    new = TRUE
  )
  graphics::plot.new()
  graphics::legend(
    "bottom",
    legend = c("Observed", "Central", "50%", "80%", "95%"),
    col = c("grey55", "#1F4E79", NA, NA, NA),
    lty = c(1, 1, NA, NA, NA),
    lwd = c(0.8, 1.2, NA, NA, NA),
    fill = c(NA, NA, colour_50, colour_80, colour_95),
    border = NA,
    horiz = TRUE,
    cex = 0.72,
    bty = "n",
    xpd = NA,
    inset = c(0, 0.01)
  )
}

proj_summary_rows <- list()
proj_arima_rows <- list()
proj_row_index <- 0L
proj_arima_index <- 0L

for (sex_index in seq_along(data_list)) {
  sex <- names(data_list)[sex_index]
  
  for (model_index in seq_along(models_to_fit)) {
    model_name <- names(models_to_fit)[model_index]
    fit_full <- quietly(fit(
      models_to_fit[[model_name]],
      data = data_list[[sex]],
      ages.fit = ages_fc,
      years.fit = proj_full_years
    ))
    cohort_model_full <- if (model_name %in% c("M2", "M6")) {
      quietly(select_cohort_arima(fit_full))
    } else {
      NULL
    }
    
    central_projection <- forecast_stmomo_projection(
      fit_full,
      cohort_model_full
    )
    simulated_projection <- simulate_stmomo_projection(
      fit_full,
      cohort_model_full,
      seed = 20260825L + 100L * sex_index + model_index
    )
    
    proj_row_index <- proj_row_index + 1L
    proj_summary_rows[[proj_row_index]] <- summarise_stmomo_projection(
      central_projection,
      simulated_projection,
      model_name,
      sex
    )
    proj_arima_index <- proj_arima_index + 1L
    proj_arima_rows[[proj_arima_index]] <- extract_projection_arima_orders(
      central_projection,
      model_name,
      sex,
      cohort_model_full
    )
  }
}

stmomo_projection_summary <- do.call(rbind, proj_summary_rows)
stmomo_projection_summary <- stmomo_projection_summary[
  order(
    stmomo_projection_summary$Sex,
    stmomo_projection_summary$Model,
    stmomo_projection_summary$Age,
    stmomo_projection_summary$Year
  ),
]
row.names(stmomo_projection_summary) <- NULL

arima_orders_full <- do.call(rbind, proj_arima_rows)
arima_orders_full <- arima_orders_full[
  order(
    arima_orders_full$Sex,
    arima_orders_full$Model,
    arima_orders_full$Index
  ),
]
row.names(arima_orders_full) <- NULL

write_csv_four_decimals <- function(data, path) {
  output <- data
  decimal_columns <- names(output)[
    vapply(output, is.numeric, logical(1)) &
      !names(output) %in% c("Age", "Year")
  ]
  output[decimal_columns] <- lapply(
    output[decimal_columns],
    function(values) formatC(values, format = "f", digits = 4)
  )
  utils::write.table(
    output,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE,
    na = ""
  )
}

write_csv_four_decimals(
  stmomo_projection_summary,
  file.path(out_dir, "stmomo_projection_summary.csv")
)
write_csv_four_decimals(
  arima_orders_full,
  file.path(out_dir, "arima_orders_full.csv")
)
write_csv_four_decimals(
  stmomo_projection_summary[
    stmomo_projection_summary$Year == 2040 &
      stmomo_projection_summary$Age %in% proj_ref_ages,
  ],
  file.path(out_dir, "projection_2040_stmomo.csv")
)

invisible(lapply(
  names(models_to_fit),
  plot_stmomo_projection_pdf,
  projection_summary = stmomo_projection_summary
))
