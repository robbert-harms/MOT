*********
Changelog
*********

v0.3.12 (2018-02-22)
====================

Added
-----
- Adds CL context cache to fix issue #5.
- Adds singularity boolean matrix to the output of the Hessian to covariance matrix.


v0.3.11 (2018-02-16)
====================
- Simplified the CL context generation in the hope it fixes issue #5.


v0.3.10 (2018-02-14)
====================

Changed
-------
- Changed the default load balancing batch size.


v0.3.9 (2018-01-30)
===================

Added
-----
- Numerical Hessian now with OpenCL support
- Adds method to get the initial parameters of a model.
- Adds initial lower and upper bound support to the numerical Hessian method.
- Adds a method to the sampling statistics to compute the distance to the mean.
- Adds InputDataParameter as superclass of ProtocolParameter and StaticMapParameter.
- Adds support for restrict keyword in CL functions.

Changed
-------
- Updates to the numerical Hessian calculation, translated more functions to OpenCL.
- Updated the buffer allocation in some methods to the new way of doing it.
- Updates to the numerical Hessian calculation, small improvement in local workgroup reductions.
- Changed the interface of the input data object to get the value for a parameter using a method call.

Other
-----
- Sets the default step size to 0.1 for the numerical differentiation, small updates to the numerical Hessian computation.
- Most of the numerical Hessian computations are now in OpenCL. Only thing remaining is median outlier removal.
- Made the KernelInputDataManager smarter such that it can detect duplicate buffers and only load those once. Furthermore, KernelInputScalars are now inlined in the kernel call.
- Made the method wrapping in the wrapped model easier.
- Lets the random restart use the model objective function instead of the L2 error. Furthermore, removed residual calculations in favor of objective function calculating.
- Renamed EvaluationModels to LikelihoodFunctions, which covers the usage better.
- Removed the GPU accelerated truncated gaussian fit since it was not doing the right thing. Added a MLE based truncated normal statistic calculator.
- In MCMC, changed the order of processing such that the starting point is stored as the first sample.


v0.3.8 (2017-09-26)
===================
- Small fix to the work group size, this will fix a INVALID_WORK_GROUP_SIZE issue with the procedure runner.


v0.3.7 (2017-09-22)
===================

Added
-----
- Adds a GPU based truncated gaussian fit.
- Adds a GPU based univariate ESS algorithm.

Changed
-------
- Updates to the model function priors.
- Updates to the KernelInputDataManager.
- Changed the sample statistic to use the CPU again for the easy statistics, for large samples this is faster than using the GPU.
- Updates to the function evaluator, made the input argument r/w by default and allows for void output functions.

Other
-----
- Prepared new release.
- Refactored the residual calculator, small performance update in MCMC.
- Removed two old mapping routines, the objective calculators.
- Project renaming.
- Work on the log likelihood calculator.
- Simplified some sampling post processing after changes in MOT.
- Removed the GPU multivariate ESS again, it was only marginally faster.
- Small speed update to the GPU univariate ESS method.
- More work on the procedure evaluator. Moved more data management tasks to the kernel input data manager.
- Renamed CLHeader to CLPrototype, covers the usage better.


v0.3.6 (2017-09-06)
===================

Added
-----
- Adds CL header containing the signature of a CL function. Modified the evaluation models to not be a model but contain a model.
- Adds a method finalize_optimized_parameters to the optimize model interface. This should be called once by the optimization routine after optimization to finalize the optimization. This saves the end user from having to to this manually in the case of codec decorated models.
- Adds mot_data_struct as a basic type for communicating data to the user provided functions.

Fixed
-----
- Fixed the rician MLE estimator. The square root was missing since the optimization routines do the squaring.

Other
-----
- Converted all priors to CLFunctions.
- Instead of the square root in the model, we take the square root in the LM method instead.
- Made the KernelInputData not contain the name, but let the encapsulating dictionary contain it instead. Made more things a CLFunction and made the library functions such that the contain just one function (trying to). Updates to the evaluation model to be more of a builder for the LL and evaluation function rather then to have the evaluation model be a function itself. The latter needs more work.
- Aligned the interface of the NamedCLFunction with the CLFunction for a possible merge in the future.
- Refactored the interface of the CLFunction class from properties to get methods.
- Small updates in various places. Local memory bug fix in the sampler.
- Made two functions for the Gamma functions.
- Made the library and model functions a subclass of a CLFunction. Adds a general CL procedure runner and a more specific CLFunction evaluator to the mapping routines. Adds the method ``evaluate`` to the CLFunction class such thatit is possible to ask a model to evaluate itself against some input."
- Moved the mot_data_struct generation from the model to the kernel functions.
- More changes to adding the mot_data_struct type.
- Intermediate work on the sampling mle and map calculator.


v0.3.5 (2017-08-29)
===================

Added
-----
- Adds support for static maps per compartment overriding the static maps only per parameter.

Changed
-------
- Updated the changelog generation slightly.
- Updated the problem data to be a perfect interface.
- Updates the parser to the latest version of Grako.

Fixed
-----
- Fixed the link to the AMD site in the docs.

Other
-----
- Renamed AbstractInputData to just InputData, which is more in line with the rest of the naming scheme.
- Renamed problem data to input data.
- Code cleanup in and variable renaming.
- Removed get_free_param_names as a required function of a model.
- Removed the DataAdapter and in return added a util function convert_data_to_dtype.


v0.3.4 (2017-08-22)
===================

Added
-----
- Adds a residual CL function to the model.

Other
-----
- Removed the const keyword from the data pointer in the model functions. Allows the user more freedom.
- Removed the get observation return function from the model interface.


v0.3.3 (2017-08-17)
===================

Added
-----
- Adds gitchangelog support for automatic changelog generation.
- Adds a positive constraint to the library.
- Adds the get_composite_model_function() function to the model builder returning a barebones CL version of the constructed model.

Changed
-------
- Updates to the changelog.

Fixed
-----
- Fixed WAIC memory.
- Fixed small indexing problem in the sampler.

Other
-----
- Small updates to the interfaces. Different batch size mechanism in MH, works now with larger number of parameters.
- Removed support for dependencies in the parameter transformations.
- Moved the cartesian product method to the utils.
- Small fix in handling static maps.
- Makes sure the calculated residuals are always a number and not NaN or INF.
- Small cosmetic changes.
- Small updates to the documentation. CLFunctionParameter now accepts strings as data type and will do the conversion itself.


v0.3.2 (2017-07-26)
===================

Changed
-------
- Update to the documentation


v0.3.1 (2017-07-25)
===================

Added
-----
- Adds some Gamma functions with tests.

Other
-----
- The model builder now actually follows the builder pattern, as such model.build() needs to be called before the model and the result needs to be passed to the optimization routines.
- Adds Gamma related library functions
- Removed the objective function and LL function and replaced it with objective_per_observation and LL_per_observation.
- Introduced get_pre_eval_parameter_modifier in the model interface for obvious speed gains.
- Undid previous commit, it was not needed.
- Small update to allow the model to signal for bounds.
- Some updates to work with static maps in the model simulation function.
- Small update to the calculation of the dependent weight (the non-optimized weight could have been smaller than 0, which is not possible).
- Made the processing strategy log statement debug level instead of info level.
- Refactored the model builders to the actual builder pattern. Small change in the OffsetGaussian objective per observation function to properly account for the noise. Removed the objective function and LL function and replaced it with objective_per_observation and LL_per_observation. Introduced get_pre_eval_parameter_modifier in the model interface for obvious speed gains.
- Introduced the KernelDataInfo as an intermediate object containing the information about the kernel data of the model.


v0.3.0 (2017-06-11)
===================

Added
-----
- Adds fixed check in the init value method. This to prevent overwriting fixations by initialization.
- Added priors to the model functions.
- Add a routine that calculates the WAIC information criteria.

Changed
-------
- Changed support for the post optimization modifiers. Small change in the sampling statistics.
- Changed the rand123 library such that it no longer automatically adds the global id to the random state. Initializing the proper state is now part of the caller.

Fixed
-----
- Fixed small regression in the model builder, it did not correctly read in the fixed values.

Other
-----
- The get_extra_results_maps function of the compartments now receives and gives the dictionaries without the compartment name, making things easier.
- Moved the data from the model builder to the ModelFunctionsInfo class.
- Adds a mechanism for adding model wide priors.
- Removed redundant comment Refactored one of the priors.
- Moved the codec out of the optimization routines.
