Package: ci\_rvm
================

.. toctree::
   :titlesonly:

   ci_rvm <ci_rvm/ci_rvm.ci_rvm>
   test_ci_rvm <ci_rvm/ci_rvm.test_ci_rvm>

Installation
------------

The package can be installed via `pip <https://pypi.org/project/pip/>`_. To install the package, you can use 

.. code-block::

	pip install ci_rvm

Usage
-----

The most convenient way to use the algorithm is to use the method 
`find_CI <ci_rvm/ci_rvm.ci_rvm.html#ci_rvm.ci_rvm.find_CI>`_. An example is below.

.. code-block:: python
   
   # Example for finding profile likelihood confidence intervals for a 
   # negative binomial model
   
   # We import some packages for convenience
   import numpy as np                  # for numerical operations
   from scipy import stats             # for stats functions
   from scipy import optimize as op    # to maximize the likelihood
   
   from ci_rvm import find_CI          # to determine confidence intervals
   from ci_rvm import find_function_CI # to determine confidence intervals
                                       # for functions of parameters
   
   
   # Define the size of the data set
   n = 100
   
   # Define the true parameters
   k, p = 5, 0.1
   
   # Generate the data set 
   data = np.random.negative_binomial(k, p, size=n)
   
   # Because the parameters are constrained to the positive range and the
   # interval (0, 1), respectively, we work on a transformed parameter space
   # with unbounded domain.
   def transform_parameters(params):
      k, p = params
      return np.exp(k), 1/(1+np.exp(-p))
   
   # Log-Likelihood function for a negative binomial model
   def logL(params):
       k, p = transform_parameters(params)
       return stats.nbinom.logpmf(data, k, p).sum()
   
   # negative log-Likelihood function for optimization (because we use 
   # minimization algorithms instead of maximization algorithms)
   negLogL = lambda params: -logL(params)
   
   # Initial guess
   x0 = [0, 0]
   
   # Maximize the likelihood
   result = op.minimize(negLogL, x0)
   
   # Print the result (we need to transform the parameters to the original 
   # parameter space to make them interpretable)
   print("The estimate is: k={:5.3f}, p={:5.3f}".format(*transform_parameters(result.x)))
   
   # Find 95% confidence intervals for all parameters.
   # Note: For complicated problems, it is worthwhile to do this in parallel.
   #       However, then we would need to encapsulate the procedure in a 
   #       method and define the likelihood function on the top level of the module.
   CIs = find_CI(result.x, logL, alpha=0.95,
                 disp=True) # the disp argument lets the algorithm print 
                            # status messages.
   
   # CIs is a 2D numpy array with CIs[i, 0] containing the lower bound of the 
   # confidence interval for the i-th parameter and CIs[i, 1] containing the
   # respective upper bound. 
   
   # Print the confidence intervals. Note: we need to transform the parameters
   # back to the original parameter space.
   original_lower = transform_parameters(CIs[:,0])
   original_upper = transform_parameters(CIs[:,1])
   print("Confidence interval for k: [{:5.3f}, {:5.3f}]".format(
      original_lower[0], original_upper[0]))
   print("Confidence interval for n: [{:5.3f}, {:5.3f}]".format(
      original_lower[1], original_upper[1]))

   # Now we find the confidence interval for a function of the parameters,
   # such as the sum of them
   def myFunction(params):
       return np.sum(transform_parameters(params))
   
   CI = find_function_CI(result.x, myFunction, logL, alpha=0.95,
                          disp=True)
   print(CI)
   
   # Computing the gradient and Hessian is necessary for the algorithm 
   # but can be computationally expensive. If you want to have full
   # control over this, define them manually:

   # import a package to compute gradient and Hessian numerically
   import numdifftools as nd

   # Define gradient and Hessian
   jac = nd.Gradient(logL)
   hess = nd.Hessian(logL)
   
   CIs = find_CI(result.x, logL, jac=jac, hess=hess, alpha=0.95,
                 disp=True) # the disp argument lets the algorithm print 
                            # status messages.
   

Scientific Publication
----------------------

The theory behind the algorithm implemented in this package is explained in the paper "`A robust and efficient algorithm to find profile likelihood confidence intervals <https://link.springer.com/article/10.1007/s11222-021-10012-y>`_". Please cite this publication if you have used the package in your own research.
