# Reads data from output/combined_data/*/
# Generates graphs in figs/smallest_angle_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio number of generated stratum using smallest size/number of
#		stratum

mnist_file <- read.table(("combined_data/mnist/angle_stats.txt"), header=TRUE, sep=",")

mpeg7_file <- read.table(("combined_data/mpeg7/angle_stats.txt"), header=TRUE, sep=",")


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
pdf("../../figs/smallest_angle_exp/mnist/mnist_001_approx_smallest_angle_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(log(mnist_file$n), log(mnist_file$minSize),
  xlab="Number of Vertices",
  ylab="Smallest Stratum Size",
  # main="Minimum Stratum Size in MNIST Graphs",
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
pdf("diag_smallest_mnist")
plot(m1)
dev.off()

pdf("../../figs/smallest_angle_exp/mpeg7/mpeg7_001_approx_smallest_angle_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(log(mpeg7_file$n), log(mpeg7_file$minSize),
  xlab="Number of Vertices",
  ylab="Smallest Stratum Size",
  # main="Minimum Stratum Size in MPEG7 Graphs",
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
pdf("diag_smallest_mpeg7")
plot(m2)
dev.off()


