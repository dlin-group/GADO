#!/usr/bin/env Rscript

library(rhdf5)
library(optparse)

options = list(
  make_option(c("-i","--input_list"), type="character", default=NULL,
              help="text file containing list of HDF5 files"),
  make_option(c("-o","--output"), type="character", default=NULL,
              help="output merged HDF5 file")
)

parser = OptionParser(option_list = options)
opt = parse_args(parser)

if(is.null(opt$input_list) || is.null(opt$output)){
  print_help(parser)
  stop("Must provide input list and output file", call.=FALSE)
}

# read file list
files <- readLines(opt$input_list)

cat("Number of files:", length(files), "\n")

# read p-values
p_list <- list()

for(i in seq_along(files)){
  
  f <- files[i]
  cat("Reading:", f, "\n")
  
  p <- h5read(f, "pval")
  
  p_list[[i]] <- p
  
}

# merge
p_all <- unlist(p_list, use.names = FALSE)

cat("Total SNPs:", length(p_all), "\n")

# write output
h5createFile(opt$output)

h5write(p_all, opt$output, "pval")

cat("Merged file written to:", opt$output, "\n")