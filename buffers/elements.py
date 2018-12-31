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

def _normalize(attribute, value, stats, mutate, out=None):
    """ Performs actual normalization
            Args:
                attribute: attr name
                value: value to normalize
                stats: {"mean": .., "stddev": ...}
                mutate: whether to give normalization a copy
                    of the value to normalize or mutate it
                out: dict to output value to {attribute : norm_value}
    """

    if "mean" in stats:
        if not mutate:
            out[attribute] = value - stats["mean"]
        else:
            value -= stats["mean"]

    if "stddev" in stats:
        stddev = np.maximum(_STDDEV_EPS, stats["stddev"])
        if not mutate:
            out[attribute] = value / stddev
        else:
            value /= stddev

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


class ElementIter:
    """ Iterator for `Element`
    """
    def __init__(self, elem):
        self._element = elem
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self._element[self._idx]
        except IndexError:
            raise StopIteration()
        self._idx += 1
        return item

@attr.s
class ElementIdxAttrMapEntry:
    """ Entry for idx_to_attr_map
    containg all info to identify
    an entry in the `Element`
    """
    attribute = attr.ib(kw_only=True)
    cond_fn = attr.ib(kw_only=True)
    max = attr.ib(kw_only=True)
    min = attr.ib(kw_only=True)


def build_idx_to_attr_map(elem):
    """ Builds the index to attribute map used
    by an `Element`'s __getitem__.
    This only needs to be called for the first
    element added to the buffer. Other `Element`'s
    can use the result.
        Args:
            element: element to compute the map for

        Returns:
            (length: total index length (number of columns), the map)
    """
    length = sum([getattr(element, attribute).shape[1] for attribute in elem.attributes])
    attributes = elem.attributes
    idx_to_attr_map = [ElementIdxAttrMapEntry(attribute=attributes[0],
                                              cond_fn=lambda idx: 0 <= idx < getattr(element, attributes[0]).shape[1],
                                              max=getattr(element, attributes[0]).shape[1],
                                              min=0)]

    for prev, cur in zip(idx_to_attr_map, attributes[1:]):
        new_max = prev.max + getattr(element, cur).shape[1]

        cond_fn = lambda idx, prev=prev, new_max=new_max: prev.max <= idx < new_max
        idx_to_attr_map.append(ElementIdxAttrMapEntry(attribute=cur,
                                                      cond_fn=cond_fn,
                                                      max=new_max,
                                                      min=prev.max))

    return length, idx_to_attr_map

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

            self.len = None
            self.idx_to_attr_map = []

        def __getattr__(self, attribute):
            return getattr(self._wrapped, attribute)

        def __len__(self):
            if not self.len:
                raise AttributeError("len attribute must be set"
                                     " this is done by using the result"
                                     " from `build_idx_to_attr_map`")
            return self._len

        def __iter__(self):
            return ElementIter(self)

        def __getitem__(self, idx):
            if not self.idx_to_attr_map:
                raise AttributeError("idx_to_attr_map attribute must be set"
                                     " this is done by using the result"
                                     " from `build_idx_to_attr_map`")

            if not isinstance(idx, int):
                raise ValueError("`idx` must be of type `int`"
                                 "not %s" % type(idx))

            entry = next(x for x in self.idx_to_attr_map if x.cond_fn(idx))

            attr_entry = getattr(self, entry.attribute)

            if len(attr_entry.shape) > 1:
                return attr_entry[:, idx - entry.min]

            return attr_entry[idx - entry.min]

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
        def from_numpy(cls, ndarray, idx_to_attr_map=None, length=None):
            """ Builds `Element` from a 2D-numpy ndarray
            useful creating a `stacked` element
                    Args:
                        ndarray: the 2D array
                        idx_to_attr_map: the index-to-attr mapping
                        length: element len

                    Returns:
                        `Element` instance
            """
            kwargs = {entry.attribute: ndarray[:, entry.min:entry.max] for entry in idx_to_attr_map}
            elem = cls(**kwargs)
            elem.idx_to_attr_map = idx_to_attr_map
            elem.len = length
            return elem


        @classmethod
        def running_normalize(cls, elem, element_stats=None, mutate=True):
            """ Normalize a `stack` of elements
            from `cls.stack` using running stats
                Args:
                    elem: `Element`(s)
                    element_stats : `ElementStats` object (if any)
                    mutate: whether to mutate stack or use copies

                Returns:
                    normalized element if mutate=False
            """
            normalization = {}

            attr_data = attr.fields_dict(cls)

            for attribute, value in attr_data.items():

                data = getattr(elem, attribute)

                stats = {}
                if value.metadata["normalization"] is ElementNormType.RUNNING_NORMALIZE:
                    stats = element_stats.fetch_stats(attribute)

                    _normalize(attribute,
                               data,
                               stats,
                               mutate,
                               out=normalization)
                else:
                    normalization[attribute] = data

            return normalization if not mutate else None

        @classmethod
        def batch_normalize(cls, elem, mutate=False):
            """ Normalize a `stack` of elements
            from `cls.stack` using batch stats
                Args:
                    elem: `Element`(s)
                    mutate: whether to mutate stack or use copies

                Returns:
                    normalized element if mutate=False
            """
            normalization = {}

            attr_data = attr.fields_dict(cls)

            for attribute, value in attr_data.items():

                data = getattr(elem, attribute)

                stats = {}
                if value.metadata["normalization"] is ElementNormType.BATCH_NORMALIZE:
                    stats = cls._fetch_batch_stats(attribute, data)

                    _normalize(attribute,
                               data,
                               stats,
                               mutate,
                               out=normalization)
                else:
                    normalization[attribute] = data

            return normalization if not mutate else None

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
