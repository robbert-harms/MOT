from pppe.cl_environments import CLEnvironmentFactory
from pppe.load_balance_strategies import PreferGPU, PreferCPU

__author__ = 'Robbert Harms'
__date__ = "2015-07-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""The default cl_environment and load balancer to use. They can be overwritten at run time.

The problem this solves is the following. During optimization we run user defined scripts in the Model definition,
for example during the post optimization function. If a user accelerates the calculations using OpenCL it needs to know
the device preferences we have.

It is a matter of message passing, if we want to run all calculations one one specific device we need some way of
telling the user scripts which devices it should use. This would either involve a lot of message passing or a global
variable.

Note that this variable should only be used as a last resort. It is for example used in the base class of all
CL routines 'AbstractCLRoutine', for the moment there is no environment or load balancer passed as a function parameter.

"""
runtime_config = {
    'cl_environments': CLEnvironmentFactory.all_devices(compile_flags=('-cl-strict-aliasing', '-cl-no-signed-zeros')),
    'load_balancer': PreferGPU()
}