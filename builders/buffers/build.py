from functools import partial
from advantage.protos.buffers import buffers_pb2
from advantage.utils.proto_parsers import parse_which_one, parse_which_one_cls
from advantage.builders.buffers.builders import BufferBuilders
import advantage.buffers as buffers

"""Build function for constructing the various Buffers
"""

def build_buffer(buffers_config):
    """ Builds a Buffer based on configuration
            Args:
                buffers_config: configuration from protobuf

            Returns:
                a Buffer object
    """
    if not isinstance(buffers_config, buffers_pb2.Buffers):
        raise ValueError("buffers_config not of type buffers_pb2")

    buffer_name = parse_which_one_cls(buffers_config, "buffer")

    try:
        buffer_builder = getattr(BufferBuilders, "build_" + buffer_name)
    except AttributeError:
        raise ValueError("Buffer %s in configuration does not exist" % buffer_name)

    try:
        buffer_obj = getattr(buffers, buffer_name)
    except AttributeError:
        raise ValueError("Buffer %s in configuration does not exist" % buffer_name)

    buffer_obj = partial(buffer_obj,
                         buffers_config.bufferSize)

    specific_buffer_config = getattr(buffers_config,
                                     parse_which_one(buffers_config, "buffer"))

    return buffer_builder(buffer_obj, specific_buffer_config)
