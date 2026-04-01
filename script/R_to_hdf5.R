library(optparse)
library(Matrix)
library(hdf5r)

options = list(
	make_option(c("-i", "--input"), type="character", default=NULL,
	help="full path to input file", metavar="character"),
	make_option(c("-o", "--output"), type="character", default=NULL,
	help="full path to output file", metavar="character"),
	make_option(c("-m", "--matrix"), type="character", default=NULL,
	help="matrix name in R environment", metavar="character"))
parser = OptionParser(option_list=options)
opt = parse_args(parser)

if(is.null(opt$input) || is.null(opt$output) || is.null(opt$matrix)){
    print_help(parser)
    stop("Must give input and output file paths and matrix name", call.=False)
}

load(file.path(opt$input), env <- new.env())
m = matrix(env[[opt$matrix]],
           nrow=dim(env[[opt$matrix]])[1],
	   ncol=dim(env[[opt$matrix]])[2])
mode(m) <- "double"
ids = as.integer(rownames(env[[opt$matrix]]))
h5file <- H5File$new(opt$output, "w")
h5file[[opt$matrix]] <- t(m)
h5file[["ids"]] <- ids
h5file$close_all()

