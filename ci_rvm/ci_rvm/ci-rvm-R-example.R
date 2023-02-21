# ========== imports ===========

# We start by importing the library that builds a bridge to python.
# If the package is not installed yet, run
  #---
  # install.packages("reticulate")
  #---
library(reticulate)

# Now we import the python package to compute profile likelihood confidence 
# intervals. 
# If python is not yet installed on your system, you will be asked
# to install Miniconda, which I advise you to do. If python is installed 
# already, make sure it is in the PATH environment variable so that it can be
# found by R. Otherwise, specify your python executable via
  #---
  # use_python("/my/path/to/python")
  #---
# If you use python via Anaconda, activate your environment of preference, e.g.
  #---
  # use_virtualenv("base")
  #---
# where "base" is the name of the environment. 
# If the package is not installed on the system yet, run
  #---
  # py_install("ci-rvm", pip=TRUE)
  #---
# If this does not work, do it from the command line outside of R by typing
#---
# python -m pip install ci-rvm
#---
ci_rvm = import("ci_rvm")


# ========== loading data and defining likelihood function ===========

# We consider the built-in data on height an weight of women. We assume that the
# weight is normally distributed around a polynomial function of the height:
# meanWeight = a + b * height^c
# where a, b, and c are parameters.
# We assume that the variance is proportional to the mean height:
# varianceWeight = d * meanWeight
# The log-likelihood function looks as follows (up to a constant)

logLikelihood = function(parameters) {
  a = abs(parameters[1]) # constrain this parameter to the positive range
  b = parameters[2]
  c = abs(parameters[3]) # constrain this parameter to the positive range
  meanWeight = a * women["height"]^b
  varianceWeight = c * meanWeight
  
  result = (-sum(log(varianceWeight))/2 
         -sum((women["weight"]-meanWeight)^2/(2*varianceWeight)))
  return(result)
}

# ========== defining gradient and Hessian of the log-likelihood ===========

# Maximizing the likelihood and finding confidence intervals requires knowledge
# of the gradient (vector of first derivatives) and Hessian 
# (matrix of second derivatives) of the log-likelihood function.
# For the considered example, we could compute these by hand on paper. However,
# to show how we could proceed in more complicated cases, we use packages to
# compute the derivatives.
# If we do not provide gradient and Hessian, the package will compute them
# for us, but this could potentially be less efficient when working from R.

library("numDeriv")

gradientLL = function(parameters) {
  return(grad(logLikelihood, parameters))
}

hessianLL = function(parameters) {
  return(hessian(logLikelihood, parameters))
}

# ========== maximizing the likelihood ===========

# define initial guess
guess = c(1, 1, 1)

# maximize the likelihood
result = optim(guess, logLikelihood, gr=gradientLL, 
               control=list(fnscale=-1, maxit=3000, trace=0))

# estimated parameters
estimate = result$par

# check the result by plotting estimated mean and data
height = (min(women["height"])-5):(max(women["height"])+5)
weight = abs(estimate[1]) * height^estimate[2]
plot(height, weight, type="l")
points(unlist(women["height"], use.names=FALSE), unlist(women["weight"], use.names=FALSE))


# ========== computing profile likelihood confidence intervals ===========

# Compute all confidence intervals at once. We obtain a matrix with a row for
# each parameter and the lower and upper confidence interval limit as columns.
# If return_success=TRUE, a second matrix will be returned, indicating which 
# of the bounds have been found with success.
# Set disp=TRUE to print status information. Especially if something goes wrong
# (which could, e.g., happen if the provided estimate is not close to the actual 
# maximum), this information can be very useful. 
# alpha denotes the desired confidence level (0.95 by default)
confidenceIntervals = ci_rvm$find_CI(estimate, logLikelihood, gradientLL, 
                                     hessianLL, alpha=0.95, disp=TRUE, 
                                     return_success=FALSE)
print(confidenceIntervals)

# Note that automatic parallel search for multiple confidence interval bounds 
# does not work if the code is called from R. You will have to do the 
# parallelization manually from within R. To that end, you can determine an 
# individual confidence interval bound as follows.

# Set the index of the parameter under investigation. Note that python indices 
# start at 0. 
index = 0

# Set the search direction. Use a positive value or TRUE to find the upper
# end point of the confidence interval and a negative value or FALSE to find
# the lower end point.
direction = -1

# Now execute the search. 
confidenceInterval1Lower = ci_rvm$find_CI(estimate, logLikelihood, 
    gradientLL, hessianLL, index, direction, alpha=0.95, disp=FALSE,
    track_x=TRUE,            # If we want to track what the algorithm did
    return_full_results=TRUE # If we want to access details on the result
    )


# The resulting object also contains information on what the other parameters
# were when the parameter under consideration assumed its extreme value. 
# This information can be helpful to detect connections between parameters.
print(confidenceInterval1Lower$x)


# ------- Additional fun stuff ---------

# Similarly, we could plot the trajectory of the search, This requires that
# track_x=TRUE when searching the confidence interval.
plot(confidenceInterval1Lower$x_track[,1],confidenceInterval1Lower$x_track[,1],
     type="o", xlab="Weight factor", ylab="Weight exponent")

# The line above seems to be super straight and we may wonder why so many 
# steps were needed to find the confidence interval end point. 
plot(confidenceInterval1Lower$x_track[,1],confidenceInterval1Lower$x_track[,2],
     type="o", xlab="Weight factor", ylab="Variance factor")

# We see the search trajectory is curved - a sign that the likelihood surface
# is not trivial.

# We can also compute the confidence interval for a function of parameters, 
# such as the mean expected value of the distribution

meanExpectedValue = function(parameters) {
  a = abs(parameters[1]) # constrain this parameter to the positive range
  b = parameters[2]
  meanWeight = a * women["height"]^b
  return(sum(meanWeight))
}


gradientMean = function(parameters) {
  return(grad(meanExpectedValue, parameters))
}

hessianMean = function(parameters) {
  return(hessian(meanExpectedValue, parameters))
}

ciOfMean = ci_rvm$find_function_CI(estimate, meanExpectedValue, logLikelihood, 
                                   gradientMean, hessianMean, gradientLL, 
                                   hessianLL, alpha=0.95, disp=FALSE
)

print(ciOfMean)