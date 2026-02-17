########################################################
### TRAIN LATE FUSION MODEL (FFPE+RNA)
#########################################################
### This script trains the multimodal late fusion model 
### - input: 
###          - FFPE model scores (from 3_HistoPath_savescore.py)
###          - RNA model scores (from 2_GeneExpress_savescore.py)
### - output:  Concatenated FFPE and RNA model scores for Late Fusion model
###############################################################################
###############################################################################
### Example command
### $ Rscript 2_LateFusion.R
###################################################
###################################################

# 1. Set Environment
suppressMessages(library(data.table))
suppressMessages(library(survival))
suppressMessages(library(glmnet))
suppressMessages(library(survcomp))

# 2. Read Data
# Note: Using 80/20 split per paper methodology (train used for both training and validation)
train_file <- 'MyData/results/late_fusion/combined_scores_train.csv'
test_file  <- 'MyData/results/late_fusion/combined_scores_test.csv'

score_train <- read.csv(train_file)
score_test  <- tryCatch(read.csv(test_file), error=function(e) NULL)

cat("\n--- Data Loaded ---\n")
cat(sprintf("Original Train samples: %d\n", nrow(score_train)))

# =======================================================
# CRITICAL FIX: Filter non-positive survival times
# =======================================================
# Cox Regression crashes if time <= 0. We remove these samples.
score_train <- score_train[score_train$survival_months > 0, ]

if (!is.null(score_test)) {
  score_test <- score_test[score_test$survival_months > 0, ]
}

cat(sprintf("Filtered Train samples: %d (Removed non-positive times)\n", nrow(score_train)))
# =======================================================

# 3. Train Late Fusion Model (Cox Lasso)
cat("\n--- Training Fusion Model ---\n")
# Create Survival Object
obj_train <- Surv(score_train$survival_months, score_train$vital_status)
# Select Features (The Scores from previous models)
x_train   <- as.matrix(score_train[, c("path_score", "rna_score")])

# Fit model
set.seed(42)
# alpha=1 is Lasso (L1), alpha=0 is Ridge (L2). Lasso helps select best modality.
cv.fit <- cv.glmnet(x_train, obj_train, family="cox", alpha=1)

# Print Weights
coefs <- coef(cv.fit, s = "lambda.min")
cat("\nLearned Weights (Alpha):\n")
print(coefs)

# 4. Predict & Evaluate
evaluate_split <- function(df, model, split_name) {
  if (is.null(df) || nrow(df) == 0) return()
  
  # Prepare input
  x_in <- as.matrix(df[, c("path_score", "rna_score")])
  
  # Predict Risk Score
  preds <- predict(model, newx = x_in, s = "lambda.min")
  df$late_fusion_score <- as.vector(preds)
  
  # Calculate C-Index
  ci <- concordance.index(x = df$late_fusion_score, 
                          surv.time = df$survival_months, 
                          surv.event = df$vital_status)$c.index
  
  cat(sprintf("%s Set Late Fusion CI: %.4f\n", split_name, ci))
  
  # Save results
  out_path <- sprintf("MyData/results/late_fusion/scores_late_%s.csv", tolower(split_name))
  write.csv(df, out_path, row.names = FALSE)
}

cat("\n--- Results ---\n")
evaluate_split(score_train, cv.fit, "Train")
if (!is.null(score_test)) {
  evaluate_split(score_test,  cv.fit, "Test")
}