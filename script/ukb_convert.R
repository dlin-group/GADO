library(Matrix)
library(hdf5r)
library(optparse)
library(igraph)

# =========================
# CLI
# =========================
options = list(
  make_option(c("-i", "--input"), type="character", default=NULL),
  make_option(c("-o", "--output"), type="character", default=NULL),
  make_option(c("-m", "--matrix"), type="character", default=NULL),
  make_option(c("-t", "--threshold"), type="double", default=0.025,
              help="threshold for GRM (abs value < threshold set to 0)")
)

parser = OptionParser(option_list=options)
opt = parse_args(parser)

if(is.null(opt$input) || is.null(opt$output) || is.null(opt$matrix)){
  print_help(parser)
  stop("Must give input/output/matrix", call.=FALSE)
}

thresh <- opt$threshold
cat("Threshold:", thresh, "\n")

# =========================
# Load GRM
# =========================
env <- new.env()
load(opt$input, envir = env)
sGRM <- env[[opt$matrix]]

cat("Matrix dim:", dim(sGRM), "\n")

# =========================
# Step 1: threshold
# =========================
if (thresh > 0) {
  cat("Applying threshold...\n")
  
  sGRM@x[abs(sGRM@x) < thresh] <- 0
  
  sGRM <- drop0(sGRM)
  
  cat("Non-zero entries after threshold:", length(sGRM@x), "\n")
}

# =========================
# Step 2: block detection（connected components）
# =========================
g <- graph_from_adjacency_matrix((sGRM != 0),
                                mode = "undirected",
                                diag = FALSE)

comp <- components(g)
membership <- comp$membership

blk_ids <- sort(unique(membership))
B <- length(blk_ids)

cat("Detected blocks:", B, "\n")

block_list <- lapply(blk_ids, function(b) which(membership == b))
block_sizes <- vapply(block_list, length, integer(1))

cat("Max block size:", max(block_sizes), "\n")

# =========================
# Write HDF5
# =========================
if (file.exists(opt$output)) file.remove(opt$output)

h5 <- H5File$new(opt$output, mode = "w")

block_info <- data.frame(
  block_id = integer(),
  size = integer()
)

for (i in seq_len(B)) {

  if ((i-1) %% 1000 == 0) {
    cat(sprintf("%d of %d\n", i-1, B))
    flush.console()
  }

  rows <- block_list[[i]]

  sub <- as.matrix(sGRM[rows, rows])

  ds_name <- sprintf("block_%d", i - 1L)
  h5[[ds_name]] <- sub

  block_info <- rbind(block_info,
                      data.frame(
                        block_id = i - 1L,
                        size     = length(rows)
                      ))
}

# =========================
# ids
# =========================
if (!is.null(rownames(sGRM))) {
  h5[["ids"]] <- rownames(sGRM)
} else {
  h5[["ids"]] <- seq_len(nrow(sGRM))
}


h5$close_all()

cat("Wrote H5:", opt$output, "\n")
cat("Total samples:", nrow(sGRM), "\n")
cat("Max block size:", max(block_sizes), "\n")