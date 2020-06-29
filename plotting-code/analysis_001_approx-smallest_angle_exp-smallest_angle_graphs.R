# Reads data from output/combined_data/*/
# Generates graphs in figs/smallest_angle_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio number of generated stratum using smallest size/number of
#		stratum

smallest_angle_exp_stat_001 <- function(){

mnist_file <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/mnist/angle_stats.txt"), header=TRUE, sep=",")

mpeg7_file <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/mpeg7/angle_stats.txt"), header=TRUE, sep=",")

rand_three <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_3.txt"), header=TRUE, sep=",")
rand_five <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_5.txt"), header=TRUE, sep=",")
rand_ten <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_10.txt"), header=TRUE, sep=",")
rand_twenty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_20.txt"), header=TRUE, sep=",")
rand_thirty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_30.txt"), header=TRUE, sep=",")
rand_forty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_40.txt"), header=TRUE, sep=",")
rand_fifty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_50.txt"), header=TRUE, sep=",")
rand_sixty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_60.txt"), header=TRUE, sep=",")
rand_seventy <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_70.txt"), header=TRUE, sep=",")
rand_eighty <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_80.txt"), header=TRUE, sep=",")
rand_ninety <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_90.txt"), header=TRUE, sep=",")
rand_hundred <- read.table(("analysis_001_approx/smallest_angle_exp/combined_data/random/angle_stats_100.txt"), header=TRUE, sep=",")

rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
              rand_forty, rand_fifty, rand_sixty, rand_seventy, rand_eighty,
              rand_ninety, rand_hundred)

##### Code to generate graphs for the ratios
# pdf("../../figs/smallest_angle_exp/mnist/mnist_001_approx_ratio_smallest_angle_exp.pdf")
# par(mar=c(5, 5, 5, 5))
# mnist_ratio <- mnist_file$fineStratum / mnist_file$necessaryStratum
# plot(log(mnist_file$n), log(mnist_ratio),
#   xlab="Number of Vertices",
#   ylab="Ratio N to n",
#   # main="Oversampling in MNIST Graphs",
#   family="serif",cex.lab=3, cex.main=2,
#   cex.sub=2,cex.axis=2)

# # Run linear regression on log of ratio
# # variable and n as dependent
# m1 <- lm(log(mnist_ratio) ~ log(mnist_file$n))
# summary(m1)
# print("MSE")
# mean(m1$residuals^2)
# # lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
# abline(m1,lty=1,col="red",lwd=3)
# dev.off()
# pdf("diag_over_mnist")
# plot(m1)
# dev.off()

# pdf("../../figs/smallest_angle_exp/mpeg7/mpeg7_001_approx_ratio_smallest_angle_exp.pdf")
# par(mar=c(5, 5, 5, 5))
# mpeg7_ratio <- mpeg7_file$fineStratum / mpeg7_file$necessaryStratum
# plot(log(mpeg7_file$n), log(mpeg7_ratio),
#   xlab="Number of Vertices",
#   ylab="Ratio N to n",
#   # main="Oversampling in MPEG7 Graphs",
#   family="serif",cex.lab=3, cex.main=2,
#   cex.sub=2,cex.axis=2)
# m2 <- lm(log(mpeg7_ratio) ~ log(mpeg7_file$n))
# summary(m2)
# print("MSE")
# mean(m2$residuals^2)
# # lines(sort(log(mpeg7_file$n)), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
# abline(m2,lty=1,col="red",lwd=3)
# dev.off()
# pdf("diag_over_mpeg7")
# plot(m2)
# dev.off()


####
#  #
####
#  #
####

##### Code to generate graphs for just the smallest angle
pdf("figs/smallest_angle_exp/mnist/mnist_001_approx_smallest_angle_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(log(mnist_file$n), log(mnist_file$minSize),
  xlab="Number of Vertices",
  ylab="Smallest Stratum Size",
  main="Minimum Stratum Size in MNIST Graphs",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)

#after looking at data, a linear model seems appropriate
m1 <- lm(log(mnist_file$minSize) ~ log(mnist_file$n))
coef(m1)
# (exp(coef(m1)["x"]) - 1) * 100
summary(m1)
print("Length")
print(length(mnist_file$n))
#MSE
print("MSE")
mean(m1$residuals^2)
# lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m1,lty=1,col="red",lwd=3)
dev.off()

##Create PNG for Jupyter Notebook
png("figs/smallest_angle_exp/mnist/mnist_001_approx_smallest_angle_exp.png")
par(mar=c(5, 5, 5, 5))
plot(log(mnist_file$n), log(mnist_file$minSize),
     xlab="Number of Vertices",
     ylab="Smallest Stratum Size",
     main="Minimum Stratum Size in MNIST Graphs",
     family="serif",cex.lab=3, cex.main=2,
     cex.sub=2,cex.axis=2)

#after looking at data, a linear model seems appropriate
m1 <- lm(log(mnist_file$minSize) ~ log(mnist_file$n))
coef(m1)
# (exp(coef(m1)["x"]) - 1) * 100
summary(m1)
print("Length")
print(length(mnist_file$n))
#MSE
print("MSE")
mean(m1$residuals^2)
# lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m1,lty=1,col="red",lwd=3)
dev.off()

png("diag_smallest_mnist")
plot(m1)
dev.off()

pdf("figs/smallest_angle_exp/mpeg7/mpeg7_001_approx_smallest_angle_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(log(mpeg7_file$n), log(mpeg7_file$minSize),
  xlab="Number of Vertices",
  ylab="Smallest Stratum Size",
  
  main="Minimum Stratum Size in MPEG7 Graphs",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)

# due to the distribution of n_0, we will test to see if log(n) is normal
m2 <- lm(log(mpeg7_file$minSize) ~ log(mpeg7_file$n))
# coef(m2)
summary(m2)
print("Length")
print(length(mpeg7_file$n))
print("MSE")
mean(m2$residuals^2)
# lines(sort(log(mpeg7_file$n)), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m2,lty=1,col="red",lwd=3)
dev.off()
png("diag_smallest_mpeg7")
plot(m2)
dev.off()

png("figs/smallest_angle_exp/mpeg7/mpeg7_001_approx_smallest_angle_exp.png")
par(mar=c(5, 5, 5, 5))
plot(log(mpeg7_file$n), log(mpeg7_file$minSize),
     xlab="Number of Vertices",
     ylab="Smallest Stratum Size",
     
     main="Minimum Stratum Size in MPEG7 Graphs",
     family="serif",cex.lab=3, cex.main=2,
     cex.sub=2,cex.axis=2)

# due to the distribution of n_0, we will test to see if log(n) is normal
m2 <- lm(log(mpeg7_file$minSize) ~ log(mpeg7_file$n))
# coef(m2)
summary(m2)
print("Length")
print(length(mpeg7_file$n))
print("MSE")
mean(m2$residuals^2)
# lines(sort(log(mpeg7_file$n)), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m2,lty=1,col="red",lwd=3)
dev.off()
png("diag_smallest_mpeg7")
plot(m2)
dev.off()


pdf("figs/smallest_angle_exp/random/random_smallest_angle_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(log(rand$n), log(rand$minSize),
     xlab="Number of Vertices",
     ylab="Smallest Stratum Size",
     main="Minimum Stratum Size in Random Graphs",
     family="serif",cex.lab=3, cex.main=2,
     cex.sub=2,cex.axis=2)
m3 <- lm(log(rand$minSize) ~ log(rand$n))
coef(m3)
summary(m3)
print("Length")
print(length(rand$n))
print("MSE")
mean(m3$residuals^2)
abline(m3,lty=1,col="red",lwd=3)
# lines(sort(log(rand$n)), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()
pdf("diag_smallest_rand")
plot(m3)
dev.off()

png("figs/smallest_angle_exp/random/random_smallest_angle_exp.png")
par(mar=c(5, 5, 5, 5))
plot(log(rand$n), log(rand$minSize),
     xlab="Number of Vertices",
     ylab="Smallest Stratum Size",
     # main="Minimum Stratum Size in Random Graphs",
     family="serif",cex.lab=3, cex.main=2,
     cex.sub=2,cex.axis=2)
m3 <- lm(log(rand$minSize) ~ log(rand$n))
coef(m3)
summary(m3)
print("Length")
print(length(rand$n))
print("MSE")
mean(m3$residuals^2)
abline(m3,lty=1,col="red",lwd=3)
# lines(sort(log(rand$n)), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()
pdf("diag_smallest_rand")
plot(m3)
dev.off()
}
