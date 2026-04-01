#!/usr/bin/env Rscript

library(rhdf5)
library(optparse)

options = list(
	make_option(c("-i", "--input"), type="character", default=NULL,
	help="full path to input file containing pval (p-values)", metavar="character"),
	make_option(c("-o", "--output"), type="character", default=NULL,
	help="full path to output file containing pval_gc (p-values with genomic control)", metavar="character"))

parser = OptionParser(option_list=options)
opt = parse_args(parser)

if(is.null(opt$input) || is.null(opt$output)){
    print_help(parser)
    stop("Must give input and output file paths", call. = FALSE)
}

cat("====================================\n")
cat("Genomic Control Correction Script\n")
cat("Input file :", opt$input, "\n")
cat("Output file:", opt$output, "\n")
cat("====================================\n")

cat("Reading p-values from HDF5...\n")

p <- tryCatch(h5read(opt$input, "/pval"), error = function(e) {
    cat("ERROR: cannot read /pval dataset\n")
    stop(e)
})

cat("Total number of p-values:", length(p), "\n")
cat("Range of p-values:", range(p, na.rm=TRUE), "\n")
cat("Number of NA:", sum(is.na(p)), "\n")
cat("Number of p == 0:", sum(p == 0, na.rm=TRUE), "\n")

# protect against zero p-values
p[p == 0] <- 1e-300

cat("Computing chi-square statistics...\n")

stat <- qchisq(p, 1, lower.tail = FALSE)

cat("Range of chi-square statistics:", range(stat, na.rm=TRUE), "\n")

cat("Computing genomic control lambda...\n")

lambda <- median(stat, na.rm = TRUE) / qchisq(0.5, df = 1)

cat("Lambda (GC inflation factor):", lambda, "\n")

cat("Applying genomic control correction...\n")

p_gc <- pchisq(stat / lambda, 1, lower.tail = FALSE)

cat("Range of corrected p-values:", range(p_gc, na.rm=TRUE), "\n")

mode(p_gc) <- "double"
mode(lambda) <- "double"

cat("Writing results to HDF5...\n")

h5createFile(opt$output)
h5write(p_gc, opt$output, "pval_gc")
h5write(lambda, opt$output, "lambda")

cat("Done!\n")
cat("Output written to:", opt$output, "\n")
cat("====================================\n")