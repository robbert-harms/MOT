*********
Changelog
*********


v0.3.5 (2017-08-29)
===================

Added
-----
- Adds support for static maps per compartments, that will if given override the static maps per parameter.

Changed
-------
- Updated the changelog generation slightly.
- Updated the problem data to be a perfect interface.
- Updates the parser to the latest version of Grako.
- Changed the DataAdapter and in return added a util function convert_data_to_dtype.

Fixed
-----
- Fixed the link to the AMD site in the docs.

Other
-----
- Renamed AbstractInputData to just InputData, which is more in line with the rest of the naming scheme.
- Renamed problem data to input data.
- Removed get_free_param_names as a required function of a model.


v0.3.4 (2017-08-22)
===================

Added
-----
- Adds a residual CL function to the model.

Other
-----
- Version bump.
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
- Updates in this version:
  - Update to the documentation
- Updates to the docs.


v0.3.1 (2017-07-25)
===================

Added
-----
- Adds some Gamma functions with tests.

Other
-----
- Main highlights of this new version:
  - The model builder now actually follows the builder pattern,
    as such model.build() needs to be called before the model and
    the result needs to be passed to the optimization routines.
  - Adds Gamma related library functions
  - Removed the objective function and LL function and replaced it with
    objective_per_observation and LL_per_observation.
  - Introduced get_pre_eval_parameter_modifier in the model interface
    for obvious speed gains.
- Undid previous commit, it was not needed.
- Small update to allow the model to signal for bounds.
- Some updates to work with static maps in the model simulation function.
- Small update to the calculation of the dependent weight (the non-optimized weight could have been smaller than 0, which is not possible).
- Made the processing strategy log statement debug level instead of info level.
- Refactored the model builders to the actual builder pattern. Small change in the OffsetGaussian objective per observation function to properly account for the noise. Removed the objective function and LL function and replaced it with objective_per_observation and LL_per_observation. Introduced get_pre_eval_parameter_modifier in the model interface for obvious speed gains.
- Introduced the KernelDataInfo as an intermediate object containing the information about the kernel data of the model.
- Small typo fix.


v0.3.0 (2017-06-11)
===================

Added
-----
- Adds fixed check in the init value method. This to prevent overwriting fixations by initialization.

Changed
-------
- Changed support for the post optimization modifiers. Small change in the sampling statistics.

Fixed
-----
- Fixed small regression in the model builder, it did not correctly read in the fixed values.

Other
-----
- Minor version bump.
- Some refactorings. Implements a routine that calculates the WAIC information criteria.
- More refactoring, added priors to the model functions.
- The get_extra_results_maps function of the compartments now receives and gives the dictionaries without the compartment name, making things easier.
- - Changed the rand123 library such that it no longer automatically adds the global id to the random state. Initializing the proper state is now part of the caller. - Moved the data from the model builder to the ModelFunctionsInfo class. - Adds a mechanism for adding model wide priors.
- Removed redundant comment Refactored one of the priors.
- Moved the codec out of the optimization routines.
- Small change to readme.


v0.2.42 (2017-05-29)
====================
- New version, containing the Subplex method.
- Removed non-ascii characters from a few of the comments.
- Small improvements to the NMSimplex method (better initialization), moved the NMSimplex algorithm to a library function, added the Subplex method as Sbplex.
- Small update to the test functions.
- Cleaned up the code in the model_builder.


v0.2.41 (2017-05-18)
====================
- Renamed 'get_optimized_param_names' in the model to 'get_free_param_names'


v0.2.40 (2017-05-17)
====================

Fixed
-----
- Fixed indexing problem with very large kernels.

Other
-----
- Moved the dependencies to the fixes API.
- Default back to single core processing if we run out of memory in the ESS calculations.
- Ulong to long in some parts of the averaging methods.
- Uses ulong now for global index locations, this fixes a long standing issue with memory corruption issues.
- Removed events as synchronization point and uses queue finish instead.
- First working version of the new MCMC sampler.
- Some refactoring in MCMC.


v0.2.39 (2017-04-09)
====================
- Reverted previous update.


v0.2.38 (2017-04-09)
====================
- Small update in the dependent parameter computation. This should be more friendly to low memory devices.
- Small update to the release-github in the Makefile.


v0.2.37 (2017-04-03)
====================

Added
-----
- Adds unit test for the model interfaces.
- Adds ESS maps to the sampling output.
- Adds a multiple lower and upper bound setter to the model builder.
- Adds the possibility to describe in a data adapter if the data can be stored in a local pointer if possible.
- Adds the AxialNormalPDF prior distribution. Small update to the model builder to now accept parameters with a dot in the name, useful for the priors.
- Adds some gc collect statements in the hope that it fixes the memory issues.
- Adds some mcmc diagnostic functionality like univariate ESS and multivariate ESS (Effective Samples Size)
- Adds the ability to unset some compile flags if we ware operating in double precision. Previously, the compile flag -cl-single-precision-constant was always enabled. When running in double precision mode this led to problems. Now, we added some switches that made sure that this flag is disabled when running in double.
  Also added a few small tweaks to the LM model for better accuracy.
- Adds exception handling to detecting double capability of a device.
- Adds a function get the log likelihood per observation.
- Adds first draft of an ARD prior.
- Adds support for hyperparameters to the priors.
- Adds comments to simplex model.
- Added a ModelDataToKernel clas that is able to convert the model data (Variable, Protocol, Static) data to buffers and CL kernel elements. This required a lot of refactoring in most of the CL routines.
- Adds version log entry to the base optimizer.
- Adds the random restart optimizer.
- Added range bounds to the cossqrclamptransform and the sinsqrclamptransform to prevent NaN.
- Adds memory release calls to most of the Worker classes, to hopefully prevent the memory allocation errors. Adds a GridSearch optimization routine. Adds a multi step optimization meta-optimizer.
- Adds links to the downloadable .whl.
- Adds a little more spacing between the paragraphs.
- Adds sudo to the installation commands.
- Adds the function docs again to git.
- Adds a calculator for the objective lists.
- Adds config checking for the cl environments setter.
- Adds debian specific make files.
- Adds meta sampler.
- Adds support for the current observation special parameter.
- Adds support for data transformation function in the model builder.
- Adds scalar test function.
- Adds equals function to the CL environments. Made the CL env and load balancer kwargs in the optimizer routines.
- Adds a smart device selection function to the CL environments factory. This enables adding filters for certain devices or platforms.
- Adds simulated annealing, adds circular gaussian proposal. Small bugfix in sample statistics.
- Adds memory pointers back to the optimizer.
- Adds static parameters. This also changes the model builder to accept these static parameters. Also changed the default batch size setting in the load balancing strategies.
- Adds changes to the powell routine.
- Adds initial Bessel function and Rician noise model.
- Adds additional stopping criteria to NMSimlex. The one by P. E. Gill, W. Murray, and M. H. Wright. Practical Optimization. Academic Press, New York, 1981.
- Adds support for return codes to the optimization routines. Adds return codes for LM method.
- Adds factor 2 to the offset gaussian noise model.
- Adds super call in one of the classes.
- Adds more qualifiers to the DataType class.
- Adds float version of the dawson, erfi and im_w_of_x functions.
- Adds initial work on adapters. Adds a data adapter.
- Adds runtime context function.
- Adds an attribute to the model builder to allow for analyzing only a selection of the problems.
- Adds model building dir and moved some components to the model building.
- Adds two more error measures, sse and mse.
- Adds ellipsis for smaller code, moved some of the buffer creation to a separate function.
- Adds a specific struct for containing the cl context. I thought this might improve things, but it does not.
- Adds the praxis optimization routine.
- Adds step bound option to LM.
- Adds the ability to set the optimization options.
- Adds method to set the noise level standard deviation in the evaluation models.
- Adds loglikelihood calculator, bugfixes to the evaluation model offsetgaussian.
- Adds str function to cl_environments.
- Adds optimization in model builder. If a protocol parameter is constant for all rows then we add the value directly in the function call.
- Adds pretty print for the routines for logging and the factory method.
- Adds some logging information, fixed bugs in calc_dependent_params.
- Adds logging, some optimizations.
- Adds new worker class for load balancing. Converted half of the old workers to the new one.
- Adds support for pertubation functions in the parameters.
- Adds routine for calculating the maps of the dependent parameters.
- Added a function for checking if a protocol has the right columns to the model builders file.

Changed
-------
- Updates to the Rand123 implementation. Changed the default key length to 2 and made it fixed. Counters are now implemented correctly in the Rand123 front-end. Added more state information to the MHState object in Metropolis Hastings.
- Changed some of the MCMC state variables from local to global pointers.
- Changed the return type to double in a few places for better accuracy.
- Updates to the mcmc diagnostics.
- Updates to the calculation of the work group size in the MCMC algorithm.
- Updates to the library functions classes. Refactored to a better layout.
- Updates to the priors.
- Changes to install docs.
- Changes to install docs.
- Changes to install docs.
- Changes to install docs.
- Changed the lower bound to 0 in the clamp in sinsqrclamptransform (from -1), it does not change anything.
- Changed the default NMSimplex functioning to use adaptive coefficients.
- Updates to the install guide.
- Updates to the rng.
- Updates to the documentation structure.
- Updates to install.
- Updates to the documentation.
- Updates to the documentation.
- Updates to the configuration file, adds VoidConfigurationAction.
- Changed the introduction document page.
- Updates to the install guide.
- Updates to readme.
- Updates to docs, adds device selection function to the init module.
- Updates to the installation of Linux docs.
- Updates to the installation of Linux docs.
- Updates to the documentation.
- Updates to the readme file.
- Updates to the ubuntu packaging.
- Updates to the ubuntu packaging in makefile.
- Updates to the ubuntu packaging in makefile.
- Updates to the installation guide.
- Updates to the docs.
- Updates to gitignore.
- Updates to gitignore.
- Updates to the debian packaging.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to docs.
- Updates to docs.
- Updates to docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the docs.
- Updates to the documentation.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the documentation config.
- Updates to the doc config.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to readme.
- Updates to the first legendre term function.
- Updates the simplex default patience.
- Updates to the documentation, removed the quick hack in the all_devices call.
- Changed 'get_new_context' to 'get_cl_context', which is semantically more correct.
- Changed the runtime configuration to a module singleton. The AbstractCLRoutine now loads default configuration from the configuration module. It is now no longer necessary to provide every CL routine with a device and load balancer.
- Changed return codes to char array.
- Updates to the enqueue map readout.
- Changed ranlux, and as a result could change the buffer allocation in the optimizers and mcmc sampling.
- Changed memory pointer flag in likelihood calculator to use_host_ptr.
- Changed memory pointers in final param transform. Reverted the memory hack in Powell. This did not work with Ball & Stick Stick with MDT.
- Changed MOT_FLOAT_TYPE to mot_float_type.
- Updates to the filterings. Different memory strategies.
- Updates to the helper routines.
- Changed powell (for a large part) back to the original code. That is, I separated the functions again.
- Updates to the packaging and distribution.
- Changes to the NMSimplex routine. Adds initial simplex scale array to set the scale per parameter. Adds initial support for Subplex method.
- Changed the _create_workers function in the CL routines. It now accepts a single python callback to generate the workers instead of generating the workers directly.
- Updates to the lmmin euclidian norm function.
- Updates to the lmmin euclidian norm function.
- Updates to the lmmin lm_lmpar.
- Updates to lmmin qrfaq.
- Updates to lmmin qrsolv.
- Changed CLContext class to CLRunContext.
- Changed the models and optimizers float to MOT_FLOAT_TYPE.
- Updates to PrAxis.
- Updates to the comments.
- Updates to the evaluation models, bugfixes in MH.
- Changed the CL code construction slightly. Now the var_data params in the data structure are initialized to a value instead of an array in the case of single dimensional arrays.
- Changed the default nmr of iterations in MCMC, back to defaulf of 1500.
- Updates to create_workers, updates to the sampling routine.
- Updates to LM.
- Changed the AR calculation back.
- Updates to the requirements.
- Updates to the load balancers.
- Updates to the meta optimizer.
- Updates from Toronto.

Fixed
-----
- Fixed the bug in the full log likelihood of the gaussian and offset-gaussian evaluation models.
- Fixed small typo in the docs.
- Fixed small typo in docs.
- Fixed sampling setting defaults in MCMC.
- Fixed bug in the noise std kernel value.
- Fixed array init bug in the rand123 initializer.
- Fixed regression in the codec.
- Fixed regression.
- Fixed dependencies and updated version.
- Fixed setup.py make issue.
- Fixed bug with scalar static map values.
- Fixed some regressions due to the previous commit.
- Fixed the enqueue map buffer readout problems.
- Fixed the right setting for the load balancing batches.
- Fixed comments in load balancer.
- Fixed small unicode/int/string bug in the cl parameters.
- Fixes the bug that the selected voxels where not selected when fetching the fixed parameters.
- Fixed some syntax warnings.
- Fixed error in comment.
- Fixed bug in the calculation of the dependent parameters.
- Fixed bug with loading 'Any' device from the cl environment factory.

Other
-----
- Preparing for github releases.
- Small refactoring in the balance strategies.
- Renamed the SumOfSquares method to SumOfSquaresEvaluationModel. Added a config switch for specifying which flags to remove when running in double precision.
- Merge branch 'master' of github.com:cbclab/MOT.
- Reverted the static map changes in the model builders. The static maps are handled now again as protocol params, one value for multiple compartments.
- Renamed test cases.
- Removed the rand123 module and moved the functions to the generate_random module.
- More updates to how the CL library functions are handled.
- Removed the cl_header functionality in favor of simply cl_code.
- Small refactorings in the random123 library. Adds unit tests for the utils module.
- Version bump.
- Version bump.
- The Metropolis Hastings routine now outputs an output object with additional information like a MHState object which contains information about the current state of the sampler. This allows one to continue sampling from the last state.
- Moved some of the optimization and sampling post-processing out of the optimizers and samplers. The optimizers and samplers now return output classes as an intermediate interface. Also, removed the gridsearch functionality, it was not really useful.
- Implements a working version of the univariate ess using the autocorrelations.
- A push towards interfaces for most objects.
- Set the burnin default length to 0.
- Small bugfixes in several places. Updates to MCMC: added some global arrays to contain the state of the sampler. This in the future would allow one to interrupt sampling and continue later with the exact same state as if there were no interruption.
- Version bump.
- Removed debugging tools.
- Version bump.
- Version bump.
- Removed the clipped gaussian proposal.
- Finalizes the work on the proposal update functions.
- Parallelized MCMC within a problem using workgroups. Adds proposal update functions.
- Removed the objective_list function in the model and replaced it with a function that returns the evaluation value per observation.
- Version bump.
- Work on the sampling.
- Moved the weights dependency to the model builder.
- Merge branch 'master' of github.com:cbclab/MOT.
- Internal updates to the way bounds are handled in the model builder. It now fully accepts maps for the bounds.
- Reformatted the priors and added vector (map based) bounds to the priors.
- Removed wily from the upload targets, adds explicit cast to the transformations.
- Renamed MutableMapping to Mapping in a few places, it is more general.
- Version bump.
- Small updates to the grid search, got it working again.
- Merge branch 'master' of github.com:cbclab/MOT.
- More refactoring in the model builder.
- Moved the buffer generation back to the CL routines.
- Version bump.
- The parameter transformations (codec) now accept maps for the lower and upper bounds of the parameters.
- Moved the codec generation functions to the model class. The encoding and decoding transformations now also accept the model data as an argument, paving the way to maps for the bounds.
- Made the model data buffer generation part of the model class.
- Removed a few old methods, updates to some comments.
- Small fix to the Powell identity reset method.
- Merge branch 'master' of github.com:cbclab/MOT.
- Update install.rst.
  Adds an install dependency
- Update README.rst.
- Replaces the old RanLux RNG with the Random123 RNG.
- Finished adding the Random123 RNG.
- Created the RNG with Random123, now proceeding with adding it to the code.
- More work on the Random123 RNG.
- More workon on Random123.
- More work on the Random123 RNG.
- More work on properly implementing the Random123 RNG.
- Initial work on the new RNG.
- MOT now uses the CosSqrClampTransform for the Weights instead of the CosSqrTransform which did not check for bounds.
- Small changes to the docs.
- Small doc updates.
- Merge branch 'master' of github.com:cbclab/MOT.
- A few adds to the install  docs.
- Removed unused import.
- Removed the get from apt-get.
- Small update to the readme.
- Edits to the install docs and added binary 2015.2.4 whl for download.
- Removed praxis from factory.
- Removed praxis.
- Merge branch 'master' of github.com:cbclab/MOT.
- Version bump for the function added to the mot init module in a previous commit.
- Working Ubuntu PPA packaging, updates to the README files to reflect the basic requirements.
- First complete version of the installation guide.
- Small updates to the credits and installation instructions in the documentation.
- Removed the changelog from the docs. Considering to use the GitHub Releases for this using the Git commit messages as a base.
- Merge branch 'master' of github.com:cbclab/MOT.
- More work on the documentation.
- Moved all model building aspects into a separate subpackage.
- Some restructuring of the codebase, updates to the documentation, version bump.
- Merge branch 'master' of github.com:robbert-harms/MOT.
- Update README.rst.
- First public version.
- Moved one of the big private arrays in the LM method to global memory. The problem was that the compiler sometimes failed to find a contiguous memory block and returnd a out of resources error.
- Version bump.
- Removed the meta optimizer.
- Removed the perturbation from the parameters and the models.
- Some refactoring on the model optimization.
- It is safer to check for collections.MutableMapping instead of dict.
- Small update to the checks in calculate_model_estimates.
- Only sets noise level if not None in the single model.
- Small fix to LM.
- Removed smoothing from the meta optimizer.
- Small updates to the problem data class.
- Moved the noise std to the problem data.
- Small fix for 4d static maps.
- The codec runner now no longer needs the specific cl environment and load balancer.
- Made the model estimate code accept both an array and an ndarray.
- Model estimate code now uses the given array for the estimations.
- Disables Clover for now.
- Small changes to make it 2.7 compatible.
- Modified model estimates calculator, adds it as default output map to the meta optimizer.
- Comments'
- Removed float warning from MCMC, version bump.
- Simplified the demo implementation of SA.
- First final draft of simulated annealing.
- Small updates to the constructors.
- Slight speedup in error measures calculation, small bugfix in model builders.
- Small bugfix to the model builder in the case of only one problem data instance.
- LevenbergMarquardt now uses the user defined noise model.
- Small update to the readout of the exit code from the optimizer.
- Reverted the default runtime configuration settings to all devices with GPUPreferred load balancer.
- Reverted back to a single parameters buffer for read and write.
- Version bump.
- Resets the load balance batch size.
- Small bugfix to powell.
- Version bump.
- Removed old post processing test code and removed the voxels processed buffer from the optimizers.
- Tried to fix python2.7 bug with unicode.
- Version bump.
- Removed the -cl-strict-aliasing compile flag.
- The compile flags are set per abstract cl routine. This allows per kernel compile flag settings. Set the default flags to 'unsafe' flags for speed.
- Small update to the correct logging position of the sampling log file. Bug fix to memory mapping MH sampling.
- Reverted change in Powell. Changed pointers flags in MH sampling.
- Made the load balancer accept a list of wait events.
- Evaluation function speed-up in Powell, this now uses the same array for the decoding.
- More updates to the memory pointers in OpenCL.
- Testing new memory buffer layout with the optimizer.
- Testing new memory buffer layout with the log likelihood.
- Testing new memory buffer layout.
- Testing new way of defining buffers and kernels with global work offset.
- Reverted back to explicit memory readout, the implicit did not work with nvidia.
- Made a few changes here and there to the buffer allocation. Removed the additional stopping criteria in NMSimplex. Made the MH work with float again.
- Version bump.
- Sampler now uses the incomplete log likelihood for sampling. This is faster and does not change the results.
- Sampler working fully again.
- Sampling works, but without burnin.
- Working on the sampler, trying to move to float.
- Made some structural changes to Powell.
- In the transformations of the weights, adds fabs() call. Updates to MH sampling, inlined the scalars.
- Removed some of the fma calls. This returns the code to original state.
- Removed some of the pown function calls.
- Reverted some of the changes to powell, and the erfi functions. Also removed the constant terms in the evaluation models during maximum likelihood estimation.
- Small update to powell.
- Removed the previous changes with the pointer flags. They do not work properly on Windows machines.
- Moving to use_host_ptr.
- Slight updates to powell.
- Small update to powell.
- Removed unused windows only import from balance strategies.
- Merge branch 'master' of ssh://137.120.141.88:7999/mts/mot.
- Small updates to the CL runtime coordination.
- Large changes to the erfi functions. Small update to the evaluation models. Made the legendre function double again.
- Large updates to the evaluation models.
- Renamed prtcl to protocol.
- Small update to the unit tests to make them run.
- Renamed the global fixed parameters to model_data, this better covers the semantics.
- Legendre back to MOT_FLOAT_TYPE.
- Small updates to the Rician evaluation function.
- Made the first legendre calculation double by default.
- Made the bessel functions double by default. Updates to the Rician evaluation model. The log likelihood calculator now accepts the evaluation model you want to use. This is needed if the model has a Rician eval model but you want to have the Gaussian eval model for the BIC calculations.
- Version bump.
- Reverted some of the changes to NMSimplex. The Subplex algorithm will have to have its own Simplex (probably)
- Removed some of the variable resuses in LM.
- Trying to get LM to compile again with Noddi.
- Finished updating LM to latest version.
- Small updates to the comments, small updates to the sampling datastructure in MCMC.
- Small updates to the comments.
- Small updates to the comments.
- Small updates to make signal generation possible.
- Small updates to the models, adds a parser for the CLDataType.
- Version bump.
- Small semantic changes to the loglikelihood and residual calculators.
- Small updates to the grammar of the model tree's.
- Bugfix to the LM decode function twice.
- More work on the DataAdapters, everything now seems to be working again.
- The kernel code generators are now accepting DataAdapters.
- Removed some old code.
- Version bump.
- Small bugfix in the model builders.
- Moved more to the model building.
- Moved more items to model_building.
- Small update to the model builders. It needs more work, specifically for the new slicing routines in MDT.
- Removed opencl 1.1 support.
- Small update to the load balance strategy.
- Removed ; from the dependencies.
- Removed ; from the dependencies.
- Slight changes to the optimizer.
- Small updates to simplex.
- Completes the work on the PrAxis method.
- More updates to PrAxis method. Now only need to add the rand function.
- Slight update to the nm simplex.
- Small bugfix in the logging in MCMC.
- Improved the evaluation models, we use a sigma of 1 now.
- Small bugfix in the eval function from model builders.
- Slight changes to allow adapting the eval function.
- Small changes in the logging.
- Removed some old calls.
- Bug fix to the evaluate_model function. Initial work on adding the BIC map to the optimization results.
- Working sampling in float. However, sampling in float quickly gets out of precision. Need to add a warning for that.
- Small performance updates.
- Moved more stuff to float.
- More updates to the float workings. LM now seems to be working again.
- Removed grid search and python callbacks.
- Working powell and nmsimplex in float space.
- More float updates.
- More float support.
- More updates towards float.
- More update towards floating point support.
- Renamed use_double to double_precision.
- Fourth push towards float support.
- Third push towards model_float typedef.
- Second push towards model_float typedef.
- First push towards model_float typedef.
- Push towards python 3.4.
- Bugfix in load balancer. When the number of batches was lower than the number of workers, no workers were executed.
- Again, moved from repr to str when generating CL code. On some platforms repr returns things like 5L instead of 5. That is, repr generates the representation of a long instead of an int. str does not have that problem.
- Again, moved from repr to str when generating CL code. On some platforms repr returns things like 5L instead of 5. That is, repr generates the representation of a long instead of an int. str does not have that problem.
- Moved from repr to str when generating CL code. On some platforms repr returns things like 5L instead of 5. That is, repr generates the representation of a long instead of an int. str does not have that problem.
- Improved comments.
- Improved the logging in the optimization routine.
- Improved logging in the optimization routine, model builders now can handle models without a period in between. Like NDI instead of NDI.ndi.
- Renamed PPPE to MOT (Maastricht Optimization Toolbox)
- Get it to workon windows with nvidia.
- Moved the cl memory flags funtion to the cl environment class.
- Removed acceptance rate counter from the MH routine.
- Complete working adaptable proposals in MCMC.
- Sampler now works with adaptable proposals.
  It is not complete yet, see the todo in MCMC
- Removed sampling from meta optimizer.
- Working on the sampler.
- Some interface changes to the model.
- All CL routines now have the cl environment and load balancer as obligatory parameters.
- Trying to solve the global environment problem.
- Tesla bug fixed in median filter.
- Improvements to the filters. Median filter now runs also for larger sizes.
- Renamed smoothing to filters.
- Slightly raised the batch size in the mappers.
- Lot of work on the load balancers.
- In optimizers, renamed the class definition of patience to default_patience. Moved calculating in batches to the root load balancer. Adds a meta load balancer for a specific device. Adds a factory for creating optimizers and smoothers by name.
- Removed the old load balancing.
- Only gaussian smoother needs to be changed to the new worker style.
- Converted more routines to the new worker setup.
- Simplified error measures, it is not in the CPU and only returns l2 norm.
- Made type changes in place.
- Bugfixes to the cl_python_callbacks generator, tried to get sampling to work better.
- Removed the 'is protocol sufficient' function from the model builder.
- Small comment update.
- In model builder the function post_optimization is renamed to finalize_optimization_results, and in the models a function is added get_extra_results_maps. The idea is that the models already contain most of the functionality for computing the extra maps fromt that model. The model builder takes those into account when computing the final optimization results.
- Small refactorings to the utils module.
- Bugfix to generate random.
- Renamed tools to utils. Removed bessel_root function from utils and moved it to MDT.
- Reformatted the cl_python callbacks generator module.
- Removed some functions from the tools which are better placed in MDT.
- Small changes to the cl python callbacks.
- Initial commit.


