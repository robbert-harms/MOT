from contextlib import contextmanager
from .cl_environments import CLEnvironmentFactory
from .load_balance_strategies import PreferGPU

__author__ = 'Robbert Harms'
__date__ = "2015-07-22"
__maintainer__ = "Robbert Harms"
__email__ = "robbert.harms@maastrichtuniversity.nl"


"""The default cl_environment and load balancer to use. They can be overwritten at run time.

The problem this solves is the following. During optimization we run user defined scripts in the Model definition (
for example during the post optimization function). If a user accelerates the calculations using OpenCL it needs to know
the device preferences we have.

It is a matter of reducing message passing, if we want to run all calculations one one specific device we need some way
of telling the user scripts which devices it should use. This would either involve a lot of message passing or a global
variable.

We require that all CL routines are instantiated with the CL environment and the load balancer to use. If they are not
known from the context defaults can be obtained from this module.

"""
ignore_kernel_warnings = True

runtime_config = {
    'cl_environments': CLEnvironmentFactory.all_devices(),
    'load_balancer': PreferGPU(),
}


@contextmanager
def runtime_config_context(cl_environments=None, load_balancer=None):
    old_config = {k: v for k, v in runtime_config.items()}

    if cl_environments:
        runtime_config['cl_environments'] = cl_environments

    if load_balancer:
        runtime_config['load_balancer'] = load_balancer

    yield

    for key, value in old_config.items():
        runtime_config[key] = value
