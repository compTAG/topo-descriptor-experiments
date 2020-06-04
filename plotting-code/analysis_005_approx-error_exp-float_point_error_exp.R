# Reads data from output/combined_data/*/
# Generates graphs in figs/error_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio height/width

mnist_file <- read.table(("combined_data/mnist/error_stats.txt"), header=TRUE, sep=",")

mpeg7_file <- read.table(("combined_data/mpeg7/error_stats.txt"), header=TRUE, sep=",")

rand_three <- read.table(("combined_data/random/error_stats_3.txt"), header=TRUE, sep=",")
rand_five <- read.table(("combined_data/random/error_stats_5.txt"), header=TRUE, sep=",")
rand_ten <- read.table(("combined_data/random/error_stats_10.txt"), header=TRUE, sep=",")
rand_twenty <- read.table(("combined_data/random/error_stats_20.txt"), header=TRUE, sep=",")
rand_thirty <- read.table(("combined_data/random/error_stats_30.txt"), header=TRUE, sep=",")
rand_forty <- read.table(("combined_data/random/error_stats_40.txt"), header=TRUE, sep=",")
rand_fifty <- read.table(("combined_data/random/error_stats_50.txt"), header=TRUE, sep=",")
rand_sixty <- read.table(("combined_data/random/error_stats_60.txt"), header=TRUE, sep=",")
rand_seventy <- read.table(("combined_data/random/error_stats_70.txt"), header=TRUE, sep=",")
rand_eighty <- read.table(("combined_data/random/error_stats_80.txt"), header=TRUE, sep=",")
rand_ninety <- read.table(("combined_data/random/error_stats_90.txt"), header=TRUE, sep=",")
rand_hundred <- read.table(("combined_data/random/error_stats_100.txt"), header=TRUE, sep=",")

rand <- rbind(rand_three, rand_five, rand_ten, rand_twenty, rand_thirty,
  rand_forty, rand_fifty, rand_sixty, rand_seventy, rand_eighty,
  rand_ninety, rand_hundred)

pdf("../../figs/error_exp/mnist/mnist_error_exp.pdf")
par(mar=c(5, 5, 5, 5))
#### TODO: RIGHT NOW I'M ADDING .000001...
mnist_ratio <- mnist_file$height / mnist_file$width
plot(log(mnist_file$n), log(mnist_ratio),xlab="Number of Vertices",
  ylab="Min Height over Max Width",
  # main="Error Prevelance in MNIST Graphs",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
m1 <- lm(log(mnist_ratio) ~ log(mnist_file$n))
summary(m1)
print("Length")
print(length(mnist_file$n))
print("MSE")
mean(m1$residuals^2)
print("Min h")
print(min(mnist_file$height))
print("Max w")
print(max(mnist_file$width))
print("Min ratio")
print(min(mnist_ratio))
abline(m1,lty=1,col="red",lwd=3)
# lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(h=-13.81)
dev.off()
pdf("diag-mnist")
plot(m1)
dev.off()

pdf("../../figs/error_exp/mpeg7/mpeg7_error_exp.pdf")
par(mar=c(5, 5, 5, 5))
mpeg7_ratio <- mpeg7_file$height / mpeg7_file$width
plot(log(mpeg7_file$n), log(mpeg7_ratio),xlab="Number of Vertices",
  ylab="Min Height over Max Width",
  # main="Error Prevelance in MPEG7 Graphs",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
m2 <- lm(log(mpeg7_ratio) ~ log(mpeg7_file$n))
summary(m2)
print("Length")
print(length(mpeg7_file$n))
print("MSE")
mean(m2$residuals^2)
print("Min h")
print(min(mpeg7_file$height))
print("Max w")
print(max(mpeg7_file$width))
print("Min ratio")
print(min(mpeg7_ratio))
abline(m2,lty=1,col="red",lwd=3)
# lines(sort(log(mpeg7_file$n)), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(h=-13.81)
dev.off()
pdf("diag-mpeg7")
plot(m2)
dev.off()

pdf("../../figs/error_exp/random/random_error_exp.pdf")
par(mar=c(5, 5, 5, 5))
rand_ratio <- rand$height / rand$width
plot(log(rand$n), log(rand_ratio),xlab="Number of Vertices",
  ylab="Min Height over Max Width",
  # main="Error Prevelance in Random Graphs",
  family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
m3 <- lm(log(rand_ratio) ~ log(rand$n))
summary(m3)
print("Length")
print(length(rand$n))
print("MSE")
mean(m3$residuals^2)
print("Min h")
print(min(rand$height))
print("Max w")
print(max(rand$width))
print("Min ratio")
print(min(rand_ratio))
abline(m3,lty=1,col="red",lwd=3)
# lines(sort(log(rand$n)), predict(m3)[sort(rand$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(h=-13.81)
dev.off()
pdf("diag-rand")
plot(m3)
dev.off()

