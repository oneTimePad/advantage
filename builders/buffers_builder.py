from protos.buffers import buffers_pb2
import utils.buffers as buffers


class BufferBuilders:

    @staticmethod
    def build_ExperienceReplayBuffer(buffer_obj, config):
        return buffer_obj()


def build(buffers_config):
    """ Builds a Buffer based on configuration
            Args:
                buffers_config: configuration from protobuf

            Returns:
                a Buffer object
    """
    if not isinstance(buffers_config, buffers_pb2.Buffers):
        raise ValueError("buffers_config not of type buffers_pb2")

    buffer_name_lower = buffers_config.WhichOneof("buffer")
    buffer_name = buffer_name_lower[0].capitalize() + buffer_name_lower[1:]

    try:
        buffer_builder = eval("BufferBuilders.build_" + buffer_name) #TODO use getattr
    except AttributeError:
        raise ValueError("Buffer %s in configuration does not exist" % buffer_name)

    try:
        buffer_obj = getattr(buffers, buffer_name)
    except AttributeError:
        raise ValueError("Buffer %s in configuration does not exist" % buffer_name)

    buffer_obj = partial(buffer_obj,
                        buffers_config.bufferType,
                        buffers_config.bufferSize)

    return buffer_builder(buffer_obj, buffers_config)
