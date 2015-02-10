import re
import warnings
from os.path import dirname, join as joinpath

import numpy as np
import pyopencl as cl


def get_env():
    return GpuEnvironment.get()

# for now, this returns one device in the context
# TODO: create a context that contains all devices on all platforms
class GpuEnvironment(object):
    """
    GPU context, with possibly many devices, and one queue per device.
    """
    ENV = None
    @staticmethod
    def get():
        if GpuEnvironment.ENV is None:
            GpuEnvironment.ENV = GpuEnvironment()
        return GpuEnvironment.ENV

    def __init__(self):
        self.context = cl.create_some_context()
        self.queues = [cl.CommandQueue(self.context, d)
                       for d in self.context.devices]
        self.boundary = max(d.min_data_type_align_size
                            for d in self.context.devices)
        self.has_double = all(has_double(d) for d in self.context.devices)
        self.compiled = {}

    def compile_program(self, name, source, dtype):
        if name not in self.compiled:
            #print "compiling",name
            self.compiled[name] = compile_model(self.context, source, dtype)
        return self.compiled[name]

    def release_program(self, name):
        if name in self.compiled:
            self.compiled[name].release()
            del self.compiled[name]

    def dtype(self, dtype):
        # here we're creating the empty matrices that will be filled by the C++ function
        dtype = np.dtype(dtype)
        #print "dtype",dtype
        if dtype is np.dtype(np.float64) and not self.has_double:
            warnings.warn("GPU supports 32-bit only")
            dtype = np.dtype(np.float32)  # force 32 bit on 32-bit only devices
        return GPU_Types(dtype)

class GPU_Types:
    def __init__(self, dtype):
        self.complex = np.complex128 if dtype is np.dtype(np.float64) else np.complex64
        self.real = np.float64 if dtype is np.dtype(np.float64) else np.float32
        self.int = np.int32
        self.complex_size = np.dtype(self.complex).itemsize
        self.real_size = np.dtype(self.real).itemsize
        self.int_size = np.dtype(self.int).itemsize

def has_double(device):
    """
    Return true if device supports double precision.
    """
    return "cl_khr_fp64" in device.extensions

def get_warp(kernel, queue):
    """
    Return the size of an execution batch for *kernel* running on *queue*.
    """
    return kernel.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                      queue.device)

def compile_model(context, source, dtype):
    """
    Build a model to run on the gpu.

    Returns the compiled program and its type.  The returned type will
    be float32 even if the desired type is float64 if any of the
    devices in the context do not support the cl_khr_fp64 extension.
    """
    dtype = np.dtype(dtype)
    if dtype==np.float64 and not all(has_double(d) for d in context.devices):
        raise RuntimeError("Double precision not supported for devices")

    F64_DEFS = """\
#ifdef cl_khr_fp64
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
"""

    # Add opencl complex header to source; do this before converting
    # source from double precision to single precision

    source = load_header("cl_fake.h") + load_header("cl_complex.h") + source

    header = F64_DEFS if dtype == np.float64 else ""
    if dtype == np.float32:
        source = float_source(source)
    # Note: USE_SINCOS makes the intel cpu slower under opencl
    if context.devices[0].type == cl.device_type.GPU:
        header += "#define USE_SINCOS\n"
    program  = cl.Program(context, header+source).build()
    return program

HEADERS = {}
def load_header(header):
    if header not in HEADERS:
        path = joinpath(dirname(__file__), header)
        with open(path) as fid:
            HEADERS[header] = fid.read()
    return HEADERS[header]

def float_source(source):
    """
    Convert code from double precision to single precision.
    """
    # Convert double keyword to float.  Accept an 'n' parameter for vector
    # values, where n is 2, 4, 8 or 16. Assume complex numbers are represented
    # as cdouble which is typedef'd to double2.
    source = re.sub(r'(^|[^a-zA-Z0-9_]c?)double(([248]|16)?($|[^a-zA-Z0-9_]))',
                    r'\1float\2', source)
    # Convert floating point constants to single by adding 'f' to the end.
    # OS/X driver complains if you don't do this.
    source = re.sub(r'[^a-zA-Z_](\d*[.]\d+|\d+[.]\d*)([eE][+-]?\d+)?',
                    r'\g<0>f', source)
    return source


