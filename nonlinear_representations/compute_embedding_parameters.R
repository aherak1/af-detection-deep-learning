suppressMessages(library(nonlinearTseries))

save_parameters <- function(class_name, chunk_size = 1000) {
  # ulazni CSV
  file_path <- paste0("/content/", class_name, ".csv")
  # ucitaj sve podatke
  mdata <- read.csv(file_path, header = TRUE)

  total_rows <- nrow(mdata)
  num_chunks <- ceiling(total_rows / chunk_size)

  cat("Ukupno redova:", total_rows, " -> chunkova:", num_chunks, "\n")

  for (chunk_idx in 1:num_chunks) {
    start_row <- (chunk_idx - 1) * chunk_size + 1
    end_row <- min(chunk_idx * chunk_size, total_rows)
    cat("Obrada reda:", start_row, "-", end_row, "\n")

    chunk <- mdata[start_row:end_row, ]
    results_df <- data.frame()

    for (i in 1:nrow(chunk)) {
      segment_id <- chunk$segment_id[i]
      time_series <- as.numeric(chunk[i, -1])  # sve osim prve kolone

      tryCatch(
        {
          # time-lag (ACF metoda, prvi minimum)
          tau.acf <- timeLag(time_series,
                             technique = "acf",
                             selection.method = "first.minimum",
                             lag.max = NULL,
                             do.plot = FALSE)

          # embedding dimenzija (FNN)
          emb_dim <- estimateEmbeddingDim(time_series,
                                          time.lag = tau.acf,
                                          max.embedding.dim = 30)

          results_df <- rbind(results_df,
                              data.frame(segment_id = segment_id,
                                         DIM = emb_dim,
                                         TAU = tau.acf))
        },
        error = function(e) {
          results_df <- rbind(results_df,
                              data.frame(segment_id = segment_id,
                                         DIM = 0,
                                         TAU = NA))
        }
      )
    }

    # upisi rezultate za ovaj chunk
    csv.path <- paste0("/content/", class_name, "_parameters_part", chunk_idx, ".csv")
    write.csv(results_df, file = csv.path, row.names = FALSE)
    cat("Sačuvan:", csv.path, "\n")
  }
}

# --- pokreni SAMO za jednu klasu (npr. "normal")
save_parameters("normal", chunk_size = 1000)
