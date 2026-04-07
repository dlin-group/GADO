library(optparse)
library(Matrix)
library(hdf5r)

# =========================
# CLI
# =========================
options = list(
	make_option(c("-i", "--input"), type="character", default=NULL,
	help="full path to input file", metavar="character"),
	make_option(c("-o", "--output"), type="character", default=NULL,
	help="full path to output file", metavar="character"),
	make_option(c("-m", "--matrix"), type="character", default=NULL,
	help="matrix name in R environment", metavar="character"),
	make_option(c("-t", "--threshold"), type="double", default=0.025,
	help="threshold for GRM (abs value < threshold set to 0)")
)

parser = OptionParser(option_list=options)
opt = parse_args(parser)

if(is.null(opt$input) || is.null(opt$output) || is.null(opt$matrix)){
    print_help(parser)
    stop("Must give input and output file paths and matrix name", call.=FALSE)
}

thresh <- opt$threshold
cat("Threshold:", thresh, "\n")

# =========================
# Load
# =========================
env <- new.env()
load(file.path(opt$input), envir = env)

grm <- env[[opt$matrix]]

# =========================
# dense matrix
# =========================
m <- matrix(grm,
            nrow = dim(grm)[1],
            ncol = dim(grm)[2])
mode(m) <- "double"

# =========================
# threshold
# =========================
if (thresh > 0) {
  cat("Applying threshold...\n")
  m[abs(m) < thresh] <- 0
}

# =========================
# IDs
# =========================
if (!is.null(rownames(grm))) {
  ids <- as.character(rownames(grm))
} else {
  ids <- as.character(seq_len(nrow(grm)))
}

# =========================
# Write HDF5
# =========================
h5file <- H5File$new(opt$output, "w")

h5file[[opt$matrix]] <- t(m)
h5file[["ids"]] <- ids

h5file$close_all()

cat("Wrote HDF5:", opt$output, "\n")
cat("Matrix dim:", dim(m), "\n")