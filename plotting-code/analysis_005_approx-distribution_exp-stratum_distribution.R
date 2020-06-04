# Reads data from output/distribution_exp/*/
# Generates graphs in figs/distribution_exp/*/
# Graphs show:
#		x-axis: number of stratum
#		y-axis: average stratum size


mnist_file <- read.table(("averages/mnist/distribution_stats.csv"), header=TRUE, sep=",")

mpeg7_file <- read.table(("averages/mpeg7/distribution_stats.csv"), header=TRUE, sep=",")

rand_three <- read.table(("averages/random/distribution_stats_3.csv"), header=TRUE, sep=",")
rand_five <- read.table(("averages/random/distribution_stats_5.csv"), header=TRUE, sep=",")
rand_ten <- read.table(("averages/random/distribution_stats_10.csv"), header=TRUE, sep=",")
rand_twenty <- read.table(("averages/random/distribution_stats_20.csv"), header=TRUE, sep=",")
rand_thirty <- read.table(("averages/random/distribution_stats_30.csv"), header=TRUE, sep=",")
rand_forty <- read.table(("averages/random/distribution_stats_40.csv"), header=TRUE, sep=",")
rand_fifty <- read.table(("averages/random/distribution_stats_50.csv"), header=TRUE, sep=",")
rand_sixty <- read.table(("averages/random/distribution_stats_60.csv"), header=TRUE, sep=",")
rand_seventy <- read.table(("averages/random/distribution_stats_70.csv"), header=TRUE, sep=",")
rand_eighty <- read.table(("averages/random/distribution_stats_80.csv"), header=TRUE, sep=",")
rand_ninety <- read.table(("averages/random/distribution_stats_90.csv"), header=TRUE, sep=",")
rand_hundred <- read.table(("averages/random/distribution_stats_100.csv"), header=TRUE, sep=",")

rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
  rand_forty, rand_fifty, rand_sixty, rand_seventy, rand_eighty,
  rand_ninety, rand_hundred)


pdf("../../figs/distribution_exp/mnist/mnist_dist_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(mnist_file$n, log(mnist_file$avgSize),xlab="Number of Vertices",
  ylab="Average Stratum Size (Radians)",main="Average Stratum Size
  for MNIST Graphs",family="serif",cex.lab=2, cex.main=2,pch=20,
  cex.sub=2,cex.axis=2)

# Run linear regression on log of avg size as response
# variable and n as dependent
m1 <- lm(log(mnist_file$avgSize) ~ poly(mnist_file$n, 2))
coef(m1)
summary(m1)
lines(sort(mnist_file$n), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()

pdf("../../figs/distribution_exp/mpeg7/mpeg7_dist_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(mpeg7_file$n, log(mpeg7_file$avgSize),xlab="Number of Vertices",
  ylab="Average Stratum Size (Radians)",main="Average Stratum Size
  for MPEG7 Graphs",family="serif",cex.lab=2, cex.main=2,
  cex.sub=2,cex.axis=2)
m2 <- lm(log(mpeg7_file$avgSize) ~ poly(mpeg7_file$n, 2))
coef(m2)
summary(m2)
lines(sort(mpeg7_file$n), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()

pdf("../../figs/distribution_exp/random/random_dist_exp.pdf")
par(mar=c(5, 5, 5, 5))
plot(rand$n, log(rand$avgSize),xlab="Number of Vertices",
  ylab="Average Stratum Size (Radians)",main="Average Stratum Size
  for Random Graphs",family="serif",cex.lab=2, cex.main=2,
  cex.sub=2,cex.axis=2)
m3 <- lm(log(rand$avgSize) ~ poly(rand$n, 2))
coef(m3)
summary(m3)
lines(sort(rand$n), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
dev.off()

