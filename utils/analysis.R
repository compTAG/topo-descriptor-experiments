# Reads data from output/combined_data/*/
# Generates graphs in figs/smallest_angle_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio number of generated stratum using smallest size/number of
#   stratum
options(defaultPackages = c("methods", "utils", "grDevices", "graphics", "stats"))

# Main function for grabbing combined experiment files
get_exp_files <-function(approx, exp, data_type){
  setwd(file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", data_type))
  file_list <- list.files()
  dataset <- read.table(file_list[1], header=TRUE, sep=",")
  if(length(file_list) > 1) {
    for (i in 2:length(file_list)){
      temp_dataset <-read.table(file_list[i], header=TRUE, sep=",")
      dataset<-rbind(dataset, temp_dataset)
      rm(temp_dataset)
    }
  }

  setwd('../../../..')
  return(dataset)

}

# Old way of grabbing combined experiment files 
get_uniform_sample_exp_files <-function(){
mnist_file <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "mnist", "sample_stats.txt")), header=TRUE, sep=",")

mpeg7_file <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "mpeg7", "sample_stats.txt")), header=TRUE, sep=",")

rand_three <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_3.txt")), header=TRUE, sep=",")
rand_five <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_5.txt")), header=TRUE, sep=",")
rand_ten <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_10.txt")), header=TRUE, sep=",")
rand_twenty <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_20.txt")), header=TRUE, sep=",")
rand_thirty <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_30.txt")), header=TRUE, sep=",")
rand_forty <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_40.txt")), header=TRUE, sep=",")
rand_fifty <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_50.txt")), header=TRUE, sep=",")
rand_sixty <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_60.txt")), header=TRUE, sep=",")
rand_seventy <- read.table((file.path("analysis_001_approx", "uniform_sample_exp", "combined_data", "random", "sample_stats_70.txt")), header=TRUE, sep=",")


rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
              rand_forty, rand_fifty, rand_sixty, rand_seventy)
              #, rand_eighty,

file_list <- list("mnist" = mnist_file, "mpeg7" = mpeg7_file,"random" = rand)
return(file_list)
}


# Old way of grabbing combined experiment files
get_smallest_angle_exp_files <- function(approx, exp, file_name){
mnist_file <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "mnist", paste0(file_name, ".txt"))), header=TRUE, sep=",")

mpeg7_file <- read.table((file.path(paste0("analysis_", approx , "_approx"),exp, "combined_data", "mpeg7", paste0(file_name, ".txt"))), header=TRUE, sep=",")

rand_three <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_3.txt"))), header=TRUE, sep=",")
rand_five <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_5.txt"))), header=TRUE, sep=",")
rand_ten <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_10.txt"))), header=TRUE, sep=",")
rand_twenty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_20.txt"))), header=TRUE, sep=",")
rand_thirty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_30.txt"))), header=TRUE, sep=",")
rand_forty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_40.txt"))), header=TRUE, sep=",")
rand_fifty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_50.txt"))), header=TRUE, sep=",")
rand_sixty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_60.txt"))), header=TRUE, sep=",")
rand_seventy <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_70.txt"))), header=TRUE, sep=",")
rand_eighty <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_80.txt"))), header=TRUE, sep=",")
rand_ninety <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_90.txt"))), header=TRUE, sep=",")
rand_hundred <- read.table((file.path(paste0("analysis_", approx , "_approx"), exp, "combined_data", "random", paste0(file_name, "_100.txt"))), header=TRUE, sep=",")

rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
              rand_forty, rand_fifty, rand_sixty, rand_seventy, rand_eighty,
              rand_ninety, rand_hundred)


file_list <- list("mnist" = mnist_file, "mpeg7" = mpeg7_file,"random" = rand)
return(file_list)

}

# Create smallest angle pdfs anc perform statistical analysis
smallest_angle_exp_analysis_pdf <-function(in_file, exp, data_type, approx){
  if (data_type == "random") {
    pdf(file.path("figs", exp, data_type, paste0(data_type, "_", exp, ".pdf")))
  } else {
    pdf(file.path("figs", exp, data_type, paste0(data_type, "_", approx, "_approx_", exp, ".pdf")))
  }
  par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
  plot(log(in_file$n), log(in_file$minSize),
    xlab="Number of Vertices",
    ylab="Smallest Stratum Size",
    main=NULL,
    family="serif",cex.lab=3, cex.main=2,
    cex.sub=2,cex.axis=2)

  #after looking at data, a linear model seems appropriate
  m1 <- lm(log(in_file$minSize) ~ log(in_file$n))
  coef(m1)
  # (exp(coef(m1)["x"]) - 1) * 100
  summary(m1)
  print("Length")
  print(length(in_file$n))
  #MSE
  print("MSE")
  mean(m1$residuals^2)
  # lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
  abline(m1,lty=1,col="red",lwd=3)
  dev.off()

  return(m1)
}

smallest_angle_exp_analysis_png <-function(in_file, exp, data_type, approx){
  if (data_type == "random") {
    png(file.path("figs", exp, data_type, paste0(data_type, "_", exp, ".png")))
  } else {
    png(file.path("figs", exp, data_type, paste0(data_type, "_", approx, "_approx_", exp, ".png")))
  }
  par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
  plot(log(in_file$n), log(in_file$minSize),
    xlab="Number of Vertices",
    ylab="Smallest Stratum Size",
    main=NULL,
    family="serif",cex.lab=3, cex.main=2,
    cex.sub=2,cex.axis=2)

  #after looking at data, a linear model seems appropriate
  m1 <- lm(log(in_file$minSize) ~ log(in_file$n))
  coef(m1)
  # (exp(coef(m1)["x"]) - 1) * 100
  summary(m1)
  print("Length")
  print(length(in_file$n))
  #MSE
  print("MSE")
  mean(m1$residuals^2)
  # lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
  abline(m1,lty=1,col="red",lwd=3)
  dev.off()
}


# Create uniform sample pdfs anc perform statistical analysis
uniform_sample_exp_analysis_pdf <-function(in_file, exp, data_type, approx){
  if (data_type == "random") {
    pdf(file.path("figs", exp, data_type, paste0(data_type, "_", exp, ".pdf")))
    par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
    ratio <- in_file$hits / in_file$num_stratum
    # # plot(poly(rand$n,2)[,1], rand_ratio,
    plot(in_file$n, ratio,
      xlab="Number of Vertices",
      ylab="Hits over Number of Stratum",
      main=NULL,
      family="serif",cex.lab=3, cex.main=2,
      cex.sub=2,cex.axis=2)
    m1 <- lm(ratio ~ poly(in_file$n,2))
    summary(m1)
    print("Length")
    print(length(in_file$n))
    print("MSE")
    mean(m1$residuals^2)
    lines(sort(in_file$n), predict(m1)[sort(in_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
    # # abline(m3)
    dev.off()
    pdf("diag-rand")
    plot(m1)
    dev.off()
  }

  else {
    pdf(file.path("figs", exp, data_type, paste0(data_type, "_", approx, "_approx_", exp, ".pdf")))
    par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
    ratio <- in_file$hits / in_file$num_stratum
    plot(in_file$n, ratio,
      xlab="Number of Vertices",
      ylab="Hits over Number of Stratum",
      main=NULL,
      family="serif",cex.lab=3, cex.main=2,
      cex.sub=2,cex.axis=2)
     m1 <- lm(ratio ~ in_file$n)
     summary(m1)
     print("Length")
     print(length(in_file$n))
     print("MSE")
     mean(m1$residuals^2)
     abline(m1,lty=1,col="red",lwd=3)
    # # lines(sort(in_file$num_stratum), predict(m1)[sort(in_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
     dev.off()
}
 return(m1)
}


uniform_sample_exp_analysis_png <-function(in_file, exp, data_type, approx){
  if (data_type == "random") {
    png(file.path("figs", exp, data_type, paste0(data_type, "_", exp, ".png")))
    par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
    ratio <- in_file$hits / in_file$num_stratum
    # # plot(poly(rand$n,2)[,1], rand_ratio,
    plot(in_file$n, ratio,
      xlab="Number of Vertices",
      ylab="Hits over Number of Stratum",
      main=NULL,
      family="serif",cex.lab=3, cex.main=2,
      cex.sub=2,cex.axis=2)
    m1 <- lm(ratio ~ poly(in_file$n,2))
    summary(m1)
    print("Length")
    print(length(in_file$n))
    print("MSE")
    mean(m1$residuals^2)
    lines(sort(in_file$n), predict(m1)[sort(in_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
    # # abline(m3)
    dev.off()
    pdf("diag-rand")
    plot(m1)
    dev.off()
  }

  else {
    png(file.path("figs", exp, data_type, paste0(data_type, "_", approx, "_approx_", exp, ".png")))
    par(mar = c(5, 6, 4, 2), oma = c(0, 1, 0, 0))
    ratio <- in_file$hits / in_file$num_stratum
    plot(in_file$n, ratio,
      xlab="Number of Vertices",
      ylab="Hits over Number of Stratum",
      main=NULL,
      family="serif",cex.lab=3, cex.main=2,
      cex.sub=2,cex.axis=2)
     m1 <- lm(ratio ~ in_file$n)
     summary(m1)
     print("Length")
     print(length(in_file$n))
     print("MSE")
     mean(m1$residuals^2)
     abline(m1,lty=1,col="red",lwd=3)
    # # lines(sort(in_file$num_stratum), predict(m1)[sort(in_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
     dev.off()
  }
  
}


# Functions for grabbing combined data and running statistical analysis
perform_smallest_stratum_exp_analysis <- function(random, mnist, mpeg7, approx){
m1 <- smallest_angle_exp_analysis_pdf(random, "smallest_stratum_exp", "random", approx)
m2 <- smallest_angle_exp_analysis_pdf(mnist, "smallest_stratum_exp", "mnist", approx)
m3 <- smallest_angle_exp_analysis_pdf(mpeg7, "smallest_stratum_exp", "mpeg7", approx)
smallest_angle_exp_analysis_png(random, "smallest_stratum_exp", "random", approx)
smallest_angle_exp_analysis_png(mnist, "smallest_stratum_exp", "mnist", approx)
smallest_angle_exp_analysis_png(mpeg7, "smallest_stratum_exp", "mpeg7", approx)

print("Best Fit Summary RANDPTS")
print(summary(m1))

print("Best Fit Summary MNIST")
print(summary(m2))

print("Best Fit Summary MPEG7")
print(summary(m3))
}


perform_uniform_sample_analysis <- function(random, mnist, mpeg7,approx){
m1 <- uniform_sample_exp_analysis_pdf(random, "uniform_sample_exp", "random", approx)
m2 <- uniform_sample_exp_analysis_pdf(mnist, "uniform_sample_exp", "mnist", approx)
m3 <- uniform_sample_exp_analysis_pdf(mpeg7, "uniform_sample_exp", "mpeg7", approx)
uniform_sample_exp_analysis_png(random, "uniform_sample_exp", "random", approx)
uniform_sample_exp_analysis_png(mnist, "uniform_sample_exp", "mnist", approx)
uniform_sample_exp_analysis_png(mpeg7, "uniform_sample_exp", "mpeg7", approx)

print("Best Fit Summary RANDPTS")
print(summary(m1))

print("Best Fit Summary MNIST")
print(summary(m2))

print("Best Fit Summary MPEG7")
print(summary(m3))
}