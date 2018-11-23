from google.protobuf import text_format
import tensorflow as tf



def parse_obj_from_file(config_file_path, proto_buf_cls):
    """ For testing: Reads from configuration file the specific object configuration
        Args:
            config_file_path : proto file to parse
            proto_buf_cls : class to deserialize to
        Returns:
            protobuf object
    """
    proto_buf_obj = proto_buf_cls()
    with tf.gfile.GFile(config_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, proto_buf_obj)

    return proto_buf_obj

def proto_to_dict(proto_obj):
    """Converts Protobuf Python objs to dictionaries
        Args:
            proto_obj: *_pb2 protbuf python obj

        Returns:
            dict of string field names and values

        Raises:
            ValueError: for not a _pb2 protobuf python obj
    """
    if not hasattr(proto_obj, "ListFields"):
        raise ValueError("Requires instance of _pb2 python protobuf obj")

    proto_dict = {}
    for field, value in proto_obj.ListFields():
        proto_dict[field.name] = value

    return proto_dict

def parse_which_one(config, field):
    """ Retrieves the selected "oneof"
        Args:
            config: protobuf config file obj
            field: the "oneof" field

        Return:
            selected str
    """
    return config.WhichOneof(field)

def parse_which_one_cls(config, field):
    """ Retrieves the selected "oneof"
    field from the protobuf config file and
    converts it to the corresponding
    class name (pretty much capitalizes it).
        Args:
            config: protobuf config file obj
            field: the "oneof" field

        Return:
            class name as str
    """
    cls_name_lower = parse_which_one(config, field)
    return cls_name_lower[0].capitalize() + cls_name_lower[1:]

def parse_hooks(config):
    """ Fetches the list of hooks from the protobuf.
    """
    return [] # TODO hooks

def parse_enum_to_str(proto, enum, enum_value):
    """ convert proto enum to str
            Args:
                prot: pb2 object
                enum: enum attr from proto as str
                enum_value: the value to convert

            Returns:
                enum as str
    """
    proper_enum_name = "_" + enum.upper()
    proto_enum = getattr(proto, proper_enum_name)

    return proto_enum.values_by_number[enum_value].name
