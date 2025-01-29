nloop <- 1000000
sampmean <- numeric(nloop)
# Null distribution simulation
for (iloop in 1:nloop) {
x <- sample(0:4, 20, replace = TRUE, prob = c (.15, .3,.25, .2,.1))
sampmean[iloop] <- mean(x)
}