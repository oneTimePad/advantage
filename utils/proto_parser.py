from google.protobuf import text_format
import tensorflow as tf

def parse_approximators_from_file(approximators_config_file):
    from protos.approximators import approximators_pb2
    """ For testing: Reads from configuration file the approximators configuration
        Args:
            approximators_config_file: string path to approx file
        Returns:
            approximators protobuf object
    """
    approximators_config = approximators_pb2.Approximators()
    with tf.gfile.GFile(approximators_config_file, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, approximators_config)

    return approximators_config



def parse_configs_from_file(config_file):
    """Reads from configuration file the various messages involved
        Args:
            config_file: string file path to config file
        Returns:
            dictionary of protobuf objects
    """
    pass
