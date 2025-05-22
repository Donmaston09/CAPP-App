
    library(Peptides)

    calculate_boman_index <- function(sequence) {
      boman_index <- boman(sequence)
      return(boman_index)
    }

    result <- calculate_boman_index("{sequence}")
    cat(result)
    