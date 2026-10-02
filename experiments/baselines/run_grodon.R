#!/usr/bin/env Rscript
# Predict minimal doubling time for each assembly from annotated CDS FASTA.

parse_options <- function(args) {
  if (length(args) == 0) return(list())
  if (length(args) %% 2 != 0 || any(!startsWith(args[seq(1, length(args), 2)], "--"))) {
    stop("Options must be --name value pairs")
  }
  values <- list()
  for (i in seq(1, length(args), 2)) {
    values[[substring(args[[i]], 3)]] <- args[[i + 1]]
  }
  values
}

option <- function(values, name, default = NULL) {
  value <- values[[name]]
  if (is.null(value)) default else value
}

first_numeric <- function(value) {
  if (length(value) == 0) NA_real_ else suppressWarnings(as.numeric(value[[1]]))
}

options <- parse_options(commandArgs(trailingOnly = TRUE))
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
repo_dir <- normalizePath(file.path(script_dir, "../.."))
data_dir <- option(options, "data-dir", file.path(repo_dir, "dse/data/ribosomal"))
cds_dir <- option(options, "cds-dir", file.path(script_dir, "cds"))
output <- option(options, "output", file.path(script_dir, "results/grodon_predictions.csv"))
mode <- option(options, "mode", "full")
training_set <- option(options, "training-set", "madin")
temperature_source <- option(options, "temperature-source", "none")
retry_failed <- tolower(option(options, "retry-failed", "false")) == "true"
limit_value <- option(options, "limit")
if (!is.null(limit_value)) {
  limit <- suppressWarnings(as.integer(limit_value))
  if (is.na(limit) || limit <= 0) stop("--limit must be a positive integer")
}
if (!mode %in% c("full", "partial")) stop("--mode must be full or partial")
if (!temperature_source %in% c("none", "growth_tmp")) {
  stop("--temperature-source must be none or growth_tmp")
}
if (!requireNamespace("gRodon", quietly = TRUE) || !requireNamespace("Biostrings", quietly = TRUE)) {
  stop("Install the gRodon and Biostrings R packages before running this script")
}
grodon_version <- as.character(utils::packageVersion("gRodon"))

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
splits <- c("train", "val", "test")
input_rows <- lapply(splits, function(split) {
  path <- file.path(data_dir, paste0("iso_rib_temp_mod_", split, ".csv"))
  rows <- read.csv(path, stringsAsFactors = FALSE)
  if (!all(c("assembly_id", "growth_tmp") %in% names(rows))) {
    stop(paste("Missing assembly_id or growth_tmp in", path))
  }
  data.frame(split = split, assembly_id = rows$assembly_id,
             growth_tmp = rows$growth_tmp, stringsAsFactors = FALSE)
})
input_rows <- do.call(rbind, input_rows)
if (anyDuplicated(input_rows$assembly_id)) stop("Assembly IDs must be unique across splits")
if (!is.null(limit_value)) input_rows <- head(input_rows, limit)

columns <- c("split", "assembly_id", "status", "doubling_hours", "lower_ci_hours",
             "upper_ci_hours", "n_cds", "n_ribosomal", "warning", "error",
             "mode", "training_set", "temperature_source", "grodon_version")
if (file.exists(output)) {
  saved <- read.csv(output, stringsAsFactors = FALSE)
  if (!all(columns %in% names(saved)) || anyDuplicated(saved$assembly_id)) {
    stop("Existing output has missing columns or duplicate assembly IDs; use a fresh --output")
  }
  if (nrow(saved) > 0 && (any(saved$mode != mode) ||
      any(saved$training_set != training_set) ||
      any(saved$temperature_source != temperature_source) ||
      any(saved$grodon_version != grodon_version))) {
    stop("Existing output uses different gRodon settings; use a fresh --output")
  }
} else {
  saved <- data.frame(
    split = character(), assembly_id = character(), status = character(),
    doubling_hours = numeric(), lower_ci_hours = numeric(), upper_ci_hours = numeric(),
    n_cds = integer(), n_ribosomal = integer(), warning = character(),
    error = character(), mode = character(), training_set = character(),
    temperature_source = character(), grodon_version = character(),
    stringsAsFactors = FALSE
  )
}

for (i in seq_len(nrow(input_rows))) {
  assembly_id <- input_rows$assembly_id[[i]]
  split <- input_rows$split[[i]]
  prior <- match(assembly_id, saved$assembly_id)
  if (!is.na(prior) && (saved$status[[prior]] == "ok" || !retry_failed)) next

  cds_path <- file.path(cds_dir, paste0(assembly_id, ".fna"))
  if (!file.exists(cds_path)) cds_path <- paste0(cds_path, ".gz")
  warnings <- character()
  n_cds <- NA_integer_
  n_ribosomal <- NA_integer_
  prediction <- tryCatch({
    if (!file.exists(cds_path)) stop("Annotated CDS FASTA is missing")
    genes <- Biostrings::readDNAStringSet(cds_path)
    n_cds <- length(genes)
    if (n_cds == 0) stop("CDS FASTA is empty")
    highly_expressed <- grepl("^(?!.*(methyl|hydroxy)).*0S ribosomal protein",
                              names(genes), ignore.case = TRUE, perl = TRUE)
    n_ribosomal <- sum(highly_expressed)
    if (n_ribosomal == 0) stop("No annotated 30S/50S ribosomal proteins found")
    temperature <- if (temperature_source == "growth_tmp") {
      as.numeric(input_rows$growth_tmp[[i]])
    } else "none"
    if (is.numeric(temperature) && !is.finite(temperature)) stop("Invalid growth_tmp")
    result <- withCallingHandlers(
      gRodon::predictGrowth(genes, highly_expressed, mode = mode,
                            temperature = temperature, training_set = training_set),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    hours <- first_numeric(result$d)
    if (!is.finite(hours) || hours <= 0) stop("gRodon returned no positive finite doubling time")
    list(status = "ok", doubling_hours = hours,
         lower_ci_hours = first_numeric(result$LowerCI),
         upper_ci_hours = first_numeric(result$UpperCI), error = "")
  }, error = function(e) {
    list(status = "failed", doubling_hours = NA_real_, lower_ci_hours = NA_real_,
         upper_ci_hours = NA_real_, error = conditionMessage(e))
  })

  record <- data.frame(
    split = split, assembly_id = assembly_id, status = prediction$status,
    doubling_hours = prediction$doubling_hours,
    lower_ci_hours = prediction$lower_ci_hours,
    upper_ci_hours = prediction$upper_ci_hours,
    n_cds = n_cds, n_ribosomal = n_ribosomal,
    warning = paste(unique(warnings), collapse = " | "), error = prediction$error,
    mode = mode, training_set = training_set,
    temperature_source = temperature_source, grodon_version = grodon_version,
    stringsAsFactors = FALSE
  )
  if (is.na(prior)) saved <- rbind(saved, record) else saved[prior, ] <- record
  utils::write.csv(saved[, columns], output, row.names = FALSE, na = "")
  cat(sprintf("[%d/%d] %s %s: %s\n", i, nrow(input_rows), split,
              assembly_id, prediction$status))
  flush.console()
}
cat(sprintf("Wrote %s (%d successful of %d processed)\n", output,
            sum(saved$status == "ok"), nrow(saved)))
