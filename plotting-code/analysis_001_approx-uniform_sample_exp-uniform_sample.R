# Reads data from output/combined_data/*/
# Generates graphs in figs/smallest_angle_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio number of generated stratum using smallest size/number of
#		stratum

uniform_sample_exp_stat_001 <- function(){

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
#rand_eighty <- read.table(("analysis_001_approx/uniform_sample_exp/combined_data/random/angle_stats_80.txt"), header=TRUE, sep=",")
#rand_ninety <- read.table(("analysis_001_approx/uniform_sample_exp/combined_data/random/angle_stats_90.txt"), header=TRUE, sep=",")
#rand_hundred <- read.table(("analysis_001_approx/uniform_sample_exp/combined_data/random/angle_stats_100.txt"), header=TRUE, sep=",")

rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
              rand_forty, rand_fifty, rand_sixty, rand_seventy)
              #, rand_eighty,
              #rand_ninety, rand_hundred)


############
############ MNIST
############
##### Code to generate graphs for the samples
pdf(file.path("figs", "uniform_sample_exp", "mnist", "mnist_001_approx_uniform_sample_exp.pdf"))
par(mar=c(5, 5, 5, 5))
mnist_ratio <- mnist_file$hits / mnist_file$num_stratum
plot(mnist_file$n, mnist_ratio,
  xlab="Number of Vertices",
  ylab="Hits over Number of Stratum",
  main="MNIST Uniform Sampling",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
 m1 <- lm(mnist_ratio ~ mnist_file$n)
 summary(m1)
 print("Length")
 print(length(mnist_file$n))
 print("MSE")
 mean(m1$residuals^2)
 abline(m1,lty=1,col="red",lwd=3)
# # lines(sort(mnist_file$num_stratum), predict(m1)[sort(mnist_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
 dev.off()

##Create PNG for Jupyter Notebook
png(file.path("figs", "uniform_sample_exp", "mnist", "mnist_001_approx_uniform_sample_exp.png"))
par(mar=c(5, 5, 5, 5))
mnist_ratio <- mnist_file$hits / mnist_file$num_stratum
plot(mnist_file$n, mnist_ratio,
  xlab="Number of Vertices",
  ylab="Hits over Number of Stratum",
  main="MNIST Uniform Sampling",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
 m1 <- lm(mnist_ratio ~ mnist_file$n)
 summary(m1)
 print("Length")
 print(length(mnist_file$n))
 print("MSE")
 mean(m1$residuals^2)
 abline(m1,lty=1,col="red",lwd=3)
# # lines(sort(mnist_file$num_stratum), predict(m1)[sort(mnist_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
 dev.off()

 pdf("diag-mnist")
 plot(m1)
 dev.off()



############
############ MPEG7
############
pdf(file.path("figs", "uniform_sample_exp", "mpeg7", "mpeg7_001_approx_uniform_sample_exp.pdf"))
par(mar=c(5, 5, 5, 5))
mpeg7_ratio <- mpeg7_file$hits / mpeg7_file$num_stratum
plot(mpeg7_file$n, mpeg7_ratio,
  xlab="Number of Vertices",
  ylab="Hits over Number of Stratum",
  main="MPEG7 Uniform Sampling",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
m2 <- lm(mpeg7_ratio ~ mpeg7_file$n)
summary(m2)
print("Length")
print(length(mpeg7_file$n))
print("MSE")
mean(m2$residuals^2)
abline(m2,lty=1,col="red",lwd=3)
# lines(sort(mpeg7_file$num_stratum^2), predict(m2)[sort(mpeg7_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()

##Create PNG for Jupyter Notebook
png(file.path("figs", "uniform_sample_exp", "mpeg7", "mpeg7_001_approx_uniform_sample_exp.png"))
par(mar=c(5, 5, 5, 5))
mpeg7_ratio <- mpeg7_file$hits / mpeg7_file$num_stratum
plot(mpeg7_file$n, mpeg7_ratio,
  xlab="Number of Vertices",
  ylab="Hits over Number of Stratum",
  main="MPEG7 Uniform Sampling",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
m2 <- lm(mpeg7_ratio ~ mpeg7_file$n)
summary(m2)
print("Length")
print(length(mpeg7_file$n))
print("MSE")
mean(m2$residuals^2)
abline(m2,lty=1,col="red",lwd=3)
# lines(sort(mpeg7_file$num_stratum^2), predict(m2)[sort(mpeg7_file$num_stratum,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()

pdf("diag-mpeg7")
plot(m2)
dev.off()

# get rid of large cooks distance points method 1
w <- abs(cooks.distance(m2)) < 4/nrow(m2$model)
lm2 <- update(m2, weights=as.numeric(w))
print("Cleaned of large cooks distance points")
summary(lm2)
print("MSE")
mean(lm2$residuals^2)
print(length(lm2$residuals))
pdf("diag-mpeg7_clean")
plot(lm2)
dev.off()

# get rid of large cooks distance points method 2
print("Method 2 of removing outliers")
#code to double check the number removed
# idea used from https://www.statology.org/how-to-identify-influential-data-points-using-cooks-distance/
cooksD<-cooks.distance(m2)
influential_obs <- as.numeric(names(cooksD)[(cooksD >= (4/nrow(m2$model)))])
mpeg7_file_cleaned <- mpeg7_file[-influential_obs, ]
mpeg7_ratio_cleaned <- mpeg7_file_cleaned$hits / mpeg7_file_cleaned$num_stratum
lm22 <- lm(mpeg7_ratio_cleaned ~ mpeg7_file_cleaned$n)
summary(lm22)
print("MSE")
mean(lm22$residuals^2)
print(length(lm22$residuals))
print("Number of influential_obs")
print(length(influential_obs))
pdf("cleaned_mpeg7.pdf")
par(mar=c(5, 5, 5, 5))
plot(mpeg7_file_cleaned$n, mpeg7_ratio_cleaned,
  xlab="Number of Vertices",
  ylab="Hits over Number of Stratum",
  # main="MPEG7 Uniform Sampling (16384 samples)",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
abline(lm22,lty=1,col="red",lwd=3)
dev.off()


############
############ RANDOM
############
 pdf(file.path("figs", "uniform_sample_exp", "random", "random_uniform_sample_exp.pdf"))
# # par(mar=c(5, 5, 5, 5))
 rand_ratio <- rand$hits / rand$num_stratum
# # plot(poly(rand$n,2)[,1], rand_ratio,
 plot(rand$n, rand_ratio,
   xlab="Number of Vertices",
   ylab="Hits over Number of Stratum",
   main="Random Data Uniform Sampling",
   family="serif",cex.lab=3, cex.main=2,
   cex.sub=2,cex.axis=2)
 m3 <- lm(rand_ratio ~ poly(rand$n,2))
 summary(m3)
 print("Length")
 print(length(rand$n))
 print("MSE")
 mean(m3$residuals^2)
 lines(sort(rand$n), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
# # abline(m3)
 dev.off()
 pdf("diag-rand")
 plot(m3)
 dev.off()

 ##Create PNG for Jupyter Notebook
 png(file.path("figs", "uniform_sample_exp", "random", "random_uniform_sample_exp.png"))
 # # par(mar=c(5, 5, 5, 5))
 rand_ratio <- rand$hits / rand$num_stratum
 # # plot(poly(rand$n,2)[,1], rand_ratio,
 plot(rand$n, rand_ratio,
      xlab="Number of Vertices",
      ylab="Hits over Number of Stratum",
      main="Random Data Uniform Sampling",
      family="serif",cex.lab=3, cex.main=2,
      cex.sub=2,cex.axis=2)
 m3 <- lm(rand_ratio ~ poly(rand$n,2))
 summary(m3)
 print("Length")
 print(length(rand$n))
 print("MSE")
 mean(m3$residuals^2)
 lines(sort(rand$n), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
 # # abline(m3)
 dev.off()
 pdf("diag-rand")
 plot(m3)
 dev.off()

print("Best Fit Summary RANDPTS")
print(summary(m3)) 

print("Best Fit Summary Mnist 001")
print(summary(m1))

print("Best Fit Summary MPEG7 001")
print(summary(m2))



}

#IFrame("figs/smallest_angle_exp/smallest_angle_exp.png", width="1500", height="500")