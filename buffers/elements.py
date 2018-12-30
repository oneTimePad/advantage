from enum import Enum
import numpy as np
import attr

""" Elements are utilized for storing information about the agent's interactions
with the environment. They are mostly used for collection information for ReplayBuffers
but can be used on their own as well. Collecting information in this manner is a common
Reinforcement learning paradigm. The internal module used in the attr module. This is wrapped
by the Element class. So the client only has to import one module.
"""

_STDDEV_EPS = 0.001 # floor stddev to avoid div-by-zero

def _normalize(attribute, normalization_dict, stats, mutate):
    """ Performs actual normalization
            Args:
                attribute: attr name
                normalization_dict: dict to contain normalizaed values
                stats: {"mean": .., "stddev": ...}
                mutate: whether to give normalization a copy
                    of the value to normalize or mutate it

            Modifies normalization_dict
    """

    if "mean" in stats:
        if not mutate:
            normalization_dict[attribute] = normalization_dict - stats["mean"]
        else:
            normalization_dict[attribute] -= stats["mean"]

    if "stddev" in stats:
        stddev = np.maximum(_STDDEV_EPS, stats["stddev"])
        if not mutate:
            normalization_dict[attribute] = normalization_dict[attribute] / stddev
        else:
            normalization_dict[attribute] /= stddev

class ElementAttr:
    """ Mixin for a Element that uses Numpy.ndarray
    attrs
    """

    @staticmethod
    def _validate(np_dtype, normalization, norm_mean, norm_std):
        """ Validates the parameters of the ElementAttr
                Args:
                    np_dtype: numpy dtype
                    normalization: `NormType`
                    norm_mean: bool whether to normalize mean
                    norm_std: bool whether to normalize stddev

                Returns:
                    bool whether `value` is valid
        """
        def valid(instance, attribute, value):
            if not isinstance(value, np.ndarray):
                raise ValueError("%s must be of type np.ndarray"
                                 " but was %s" % (attribute.name, type(value)))
            if not value.dtype == np_dtype:
                raise ValueError("%s must have dtype %s but had dtype=%s"
                                 % (attribute.name, str(np_dtype), str(value.dtype)))

            if normalization not in ElementNormType:
                raise ValueError("`normalization` must be a valid"
                                 " attribute of `NormType` enum")

            if normalization and not norm_mean and not norm_std:
                raise ValueError("With a `normalization` set, `norm_mean`"
                                 " and `norm_std` cannot both be false")
        return valid

    @classmethod
    def attr(cls,
             np_dtype,
             normalization=None,
             norm_mean=True,
             norm_std=True):

        """ Represents an attribute of the an Element
                            Args:
                                np_dtype: numpy dtype
                                normalization: `NormType`
                                norm_mean: bool whether to normalize mean
                                norm_std: bool whether to normalize stddev

                            Returns:
                                new attr
        """
        return attr.ib(kw_only=True,
                       validator=cls._validate(np_dtype, normalization, norm_mean, norm_std),
                       metadata={"np_attr" : True, "np_dtype": np_dtype,
                                 "normalization" : normalization,
                                 "norm_mean": norm_mean, "norm_std": norm_std})

class ElementNormType(Enum):
    """ Represents an `ElementAttr`'s
    normalization type
    """
    BATCH_NORMALIZE = 1
    RUNNING_NORMALIZE = 2

class ElementStats:
    """ Represents Global(Local) depending
    on how agent uses it running normalization
    statistics
    """
    def __init__(self, element_attrs):


        self._attrs = {attribute.name: self._zero_stats_attr(attr)
                       for attribute in element_attrs
                       if attribute.metadata["normalization"] is ElementNormType.RUNNING_NORMALIZE}

        self._stddev = lambda count, sums_sqr, sums: np.sqrt((1. / count) * (sums_sqr - ((sums ** 2) / count)))
        self._mean = lambda count, sums: sums / count

    @staticmethod
    def _zero_stats_attr(attribute):
        """ Build zeroed out normalization stats
                Args:
                    attribute: `ElementAttr`

                Returns:
                    initial stats dict
        """
        stats = {"count" : 0, "sums" : np.array([0.], dtype=attribute.metadata["np_dtype"])}

        if attribute.metadata["norm_std"]:
            stats["sums_sqr"] = np.array([0.], dtype=attribute.metadata["np_dtype"])

        stats["stats_to_norm"] = [x for x in ["norm_mean", "norm_std"] if attribute.metadata[x]]

        return stats

    def update(self, new):
        """ Update stats based on `new`
        """
        for attribute, stats in self._attrs.items():
            new_attr_value = getattr(new, attribute)
            stats["count"] += 1
            stats["sums"] += new_attr_value
            stats["sums_sqr"] += new_attr_value ** 2

    def fetch_stats(self, attribute):
        """ Computes mean and/or stddev
        for `attribute`
            Args:
                attribute: `ElementAttr` name to compute
                    stats for

            Returns:
                {"mean": , "stddev": }
        """

        if attribute not in self._attrs:
            raise ValueError("attribute=%s doesn't have"
                             " normalization=RUNNING_NORMALIZE" % attribute)

        stats = self._attrs[attribute]
        fetched_stats = {}

        if "norm_mean" in stats["stats_to_norm"]:
            fetched_stats["mean"] = self._mean(stats["count"], stats["sums"])

        if "norm_std" in stats["stats_to_norm"]:
            fetched_stats["stddev"] = self._stddev(stats["count"],
                                                   stats["sums"],
                                                   stats["sums_sqr"])

        return fetched_stats


def element(cls):
    """ Class Decorator for making an `Element`
    """

    cls = attr.s(cls, frozen=True)

    class Element:
        """ Represents a type to be placed in a Replay buffer (however, they can be used
        for other purposes as well, and do not require a buffer)
        There purpose is to serve as keepers of steps taken by the agent
        and the associated changes in the environment.
        """

        def __init__(self, **kwargs):
            self._wrapped = cls(**kwargs)

        def __getattr__(self, attribute):
            return getattr(self._wrapped, attribute)

        @property
        def attributes(self):
            """ property for fetching
            attribute names
            """
            return list(attr.fields_dict(cls).keys())

        def unzip_to_tuple(self):
            """ Puts attributes to tuple
            """
            return [(attribute_key, getattr(self, attribute_key))
                    for attribute_key in self.attributes]

        def unzip_to_dict(self):
            """ Puts attributes in dict
            """
            return {attribute_key: getattr(self, attribute_key)
                    for attribute_key in attr.fields_dict(cls).keys()}

        def deep_copy(self):
            """ Performs deep copy
            of element
            """
            return self.make_element_from_dict(self.unzip_to_dict())

        @classmethod
        def make_element_from_dict(cls, dictionary):
            """ Factory: consructs element from dict of
            corresponding attrs
            """
            return cls.make_element(**dictionary)

        @classmethod
        def make_element(cls, **kwargs):
            """ Factory: Instantiates a Element of subclass 'cls'
                    Up to the subclass how it wants to handle data-sources.
                    There just must be at least one 'make' method
                    Args:
                        kwargs: specify attrs needed to instantiate

                    Returns:
                        instance of 'cls'
            """
            return cls(**kwargs)

        @classmethod
        def _fetch_batch_stats(cls, attribute, stacked):
            """ Fetch stats (mean, stddev) for batch
                Args:
                    attribute: attribute name
                    stacked: stacked NumpyAttrs

                Return : {"mean": , "stddev": }

            """
            attrs_dict = attr.fields_dict(cls)

            if attribute not in attrs_dict:
                raise AttributeError("attribute=%s is not valid" % attribute)

            attr_metadata = attrs_dict[attribute].metadata

            if attr_metadata["normalization"] is not ElementNormType.BATCH_NORMALIZE:
                raise ValueError("attribute=%s doesn't have"
                                 " normalization=BATCH_NORMALIZE" % attribute)

            fetched_stats = {}

            if attr_metadata["norm_mean"]:
                fetched_stats["mean"] = stacked[attribute].mean()

            if attr_metadata["norm_std"]:
                fetched_stats["stddev"] = stacked[attribute].stddev()

            return fetched_stats


        @classmethod
        def stack(cls, element_list):
            """ Combines common np_attrs in the elements in the list
            into one np_attr (stacked np.ndarray)

                Args:
                    element_list: list of BufferElement's to reduce
                    normalize_attrs: attrs to normalize just in stack

                Returns:
                    reduced Element with

                Raises:
                    ValueError: not all element types match in
                        `element_list`

            """
            stacked_dict = {attribute.name: [] for attribute in attr.fields_dict(cls).keys()}

            for elem in element_list:
                if not isinstance(elem, cls):
                    raise ValueError("all elements in `element_list`"
                                     " must be an instance of %s" % cls)

                for attribute, stacked_list in stacked_dict.items():
                    stacked_list.append(getattr(elem, attribute))

            return cls.make_element_from_dict({k: np.vstack(v) for k, v in stacked_dict.items()})


        @classmethod
        def stack_run_normalize(cls, stacked, element_stats=None, mutate=False):
            """ Normalize a `stack` of elements
            from `cls.stack` using running stats
                Args:
                    stacked: stacked dict of elements
                    element_stats : `ElementStats` object (if any)
                    mutate: whether to mutate stack or use copies

                Returns:
                    normalized stack
            """
            normalization = {}
            attr_metadata = attr.fields_dict(cls)

            for attribute, value in stacked.items():
                norm_type = attr_metadata[attribute]
                stats = {}

                if norm_type is ElementNormType.RUNNING_NORMALIZE:
                    stats = element_stats.fetch_stats(attribute)

                normalization[attribute] = value

                _normalize(attribute, normalization, stats, mutate)

                return normalization

        @classmethod
        def elem_run_normalize(cls, elem, element_stats=None, mutate=False):
            """ Normalize a single element using running stats
                Args:
                    elem: `Element`
                    element_stats : `ElementStats` object (if any)
                    mutate: whether to mutate stack or use copies

                Returns:
                    normalized element
            """
            return cls.stack_run_normalize(cls.unzip_to_dict(elem), element_stats, mutate)

        @classmethod
        def stack_batch_normalize(cls, stacked, mutate=False):
            """ Normalize a `stack` of elements
            from `cls.stack` using batch stats
                Args:
                    stacked: stacked dict of elements
                    mutate: whether to mutate stack or use copies

                Returns:
                    normalized stack
            """
            normalization = {}
            attr_metadata = attr.fields_dict(cls)

            for attribute, value in stacked.items():
                norm_type = attr_metadata[attribute]
                stats = {}

                if norm_type is ElementNormType.BATCH_NORMALIZE:
                    stats = cls._fetch_batch_stats(attribute, stacked)

                normalization[attribute] = value

                _normalize(attribute, normalization, stats, mutate)

                return normalization

        @classmethod
        def build_stats(cls):
            """ Constructs all global `Element` cls stats. This are
            used solely for `ElementAttr`'s with the normalization option
            set to `RUNNING_NORMALIZE`.
                Returns:
                    ElementStats object
            """
            return ElementStats(attr.fields(cls))

    return Element

@element
class Sarsa:
    """ Sarsa  represents the typical buffer Element
    used in RL
    """
    state = ElementAttr(np.float32, ElementNormType.RUNNING_NORMALIZE, norm_mean=True, norm_std=True)
    state = ElementAttr(np.float32)
    reward = ElementAttr(np.float32, ElementNormType.RUNNING_NORMALIZE, norm_std=True)
    done = ElementAttr(np.bool)
    next_state = ElementAttr(np.float32, ElementNormType.RUNNING_NORMALIZE, norm_mean=True, norm_std=True)
    n_step_return = ElementAttr(np.float32)
