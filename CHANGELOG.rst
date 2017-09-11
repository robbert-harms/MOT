*********
Changelog
*********


v0.3.6 (2017-09-06)
===================

Added
-----
- Adds CL header containing the signature of a CL function.
- Adds a method finalize_optimized_parameters to the optimize model interface. This should be called once by the optimization routine after optimization to finalize the optimization. This saves the end user from having to to this manually in the case of codec decorated models.
- Adds mot_data_struct as a basic type for communicating data to the user provided functions.
- Adds a general CL procedure runner and a more specific CLFunction evaluator to the mapping routines.
- Adds the method 'evaluate' to the CLFunction interface, allowing it to evaluate itself against some input.

Changed
-------
- All optimization routines now linearly sum the evaluation values from the evaluation model. Previously it summed the squares.
- Modified the evaluation models to not be a model but contain a model.
- Converted all priors to CLFunctions.
- Made the KernelInputData not contain the name, but let the encapsulating dictionary contain it instead.
- Work in progress on making everything that looks like a funcion an actual CLFunction instance
- Refactored the interface of the CLFunction class from properties to get methods.
- Moved the mot_data_struct generation from the model to the kernel functions.
- Aligned the interface of the NamedCLFunction with the CLFunction for a possible merge in the future.

Fixed
-----
- Local memory timing bug fix in the sampler.

Other
-----
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
