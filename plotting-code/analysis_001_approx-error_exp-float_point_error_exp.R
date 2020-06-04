# Reads data from output/combined_data/*/
# Generates graphs in figs/error_exp/*/
# Graphs show:
#   x-axis: number of vertices
#   y-axis: the ratio height/width

mnist_file <- read.table(("combined_data/mnist/error_stats.txt"), header=TRUE, sep=",")

mpeg7_file <- read.table(("combined_data/mpeg7/error_stats.txt"), header=TRUE, sep=",")

pdf("../../figs/error_exp/mnist/mnist_001_approx_error_exp.pdf")
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
# lines(sort(log(mnist_file$n)), predict(m1)[sort(mnist_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m1,lty=1,col="red",lwd=3)
abline(h=-13.81)
dev.off()
pdf("diag-mnist")
plot(m1)
dev.off()

pdf("../../figs/error_exp/mpeg7/mpeg7_001_approx_error_exp.pdf")
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
# lines(sort(log(mpeg7_file$n)), predict(m2)[sort(mpeg7_file$n,index.return=TRUE)$ix],lty=1,col="red",lwd=3)
abline(m2,lty=1,col="red",lwd=3)
abline(h=-13.81)
dev.off()
pdf("diag-mpeg7")
plot(m2)
dev.off()

