# Reads data from output/distribution_exp/*/
# Generates graphs in figs/distribution_exp/*/
# Graphs show:
#		x-axis: number of stratum
#		y-axis: average stratum size


mnist_file <- read.table(("../../output_001_approx/delta_exp/mnist/deltas.txt"), header=TRUE, sep=",")

mpeg7_file <- read.table(("../../output_001_approx/delta_exp/mpeg7/deltas.txt"), header=TRUE, sep=",")



pdf("../../figs/delta_exp_figs/mnist/mnist_delta_exp_001.pdf")
par(mar=c(5, 5, 5, 5))
hist(mnist_file$delta,ylab="Number of Graphs",xlab="size (radians)",
	main="",
	family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
print("Min delta for MNIST")
print(min(mnist_file$delta))
print("Max delta for MNIST")
print(max(mnist_file$delta))
print("Total number of graphs")
print(length(mnist_file$n))
print(summary(mnist_file$delta))
dev.off()

pdf("../../figs/delta_exp_figs/mpeg7/mpeg7_delta_exp_001.pdf")
par(mar=c(5, 5, 5, 5))
hist(mpeg7_file$delta,ylab="Number of Graphs",xlab="size (radians)",
	main="",
	family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
print("Min delta for MPEG7")
print(min(mpeg7_file$delta))
print("Max delta for MPEG7")
print(max(mpeg7_file$delta))
print("Total number of graphs")
print(length(mpeg7_file$n))
print(summary(mpeg7_file$delta))
dev.off()

pdf("../../figs/delta_exp_figs/mnist/mnist_delta_exp_plot_001.pdf")
par(mar=c(5, 5, 5, 5))
plot(mnist_file$n, mnist_file$delta, ylab="Delta (radians)",xlab="Vertices",
	main="",
	family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
dev.off()

pdf("../../figs/delta_exp_figs/mpeg7/mpeg7_delta_exp_plot_001.pdf")
par(mar=c(5, 5, 5, 5))
plot(mpeg7_file$n, mpeg7_file$delta, ylab="Delta (radians)",xlab="Vertices",
	main="",
	family="serif",cex.lab=3, cex.main=2,
  cex.sub=2,cex.axis=2)
dev.off()


