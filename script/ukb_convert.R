library(Matrix)
library(hdf5r)
library(optparse)

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
grm = env[[opt$matrix]]

# get the start and end index of each block
d = diff(grm@p)
bli = rep(0, length(d))
blj = rep(0, length(d))
maxbl = 0
index = 0
hist = rep(0, 10000)
for(i in 1:length(d)){
    if(index == 0){
        index = index + 1
        bli[index] = i
    }
    else if(i == length(d)) blj[index] = i
    else if(d[i+1] <= d[i]){
	blj[index] = i
	hist[blj[index]-bli[index]+1] = hist[blj[index]-bli[index]+1] + 1
	maxbl = max(maxbl, blj[index] - bli[index] + 1)
	index = index + 1
	bli[index] = i+1
    }
}
maxbl = max(maxbl, blj[index] - bli[index] + 1)
print(sprintf("max block dim = %d", maxbl))
for(i in 1:length(hist)){
    if(hist[i] > 0) print(sprintf("%d, %d,", i, hist[i]))
}

# write the blocks to an hdf5 file
x = grm@x
p = grm@p
minrc = 1.0
file.h5 <- H5File$new(opt$output, mode="w")
for(i in 1:index){
    if((i-1) %% 1000 == 0) print(sprintf("%d of %d", i-1, index))
    n = blj[i] - bli[i] + 1
    m = matrix(rep(0.0, n*n), nrow=n, ncol=n)
    for(j in 1:n){
        for(k in 1:j){
            m[j,k] = 2*x[p[bli[i]+j-1]+k]
	    if(j != k) m[k,j] = m[j,k]
	    if(abs(m[j,k]) > 0.0) minrc = min(minrc, abs(m[j,k]))
        }
    }
    file.h5[[sprintf("block_%d", i-1)]] <- m
}
file.h5$close_all()
print(sprintf("min rel coeff = %f", minrc))

