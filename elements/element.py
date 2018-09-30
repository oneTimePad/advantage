from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import attr


""" Elements are utilized for storing information about the agent's interactions
with the environment. They are mostly used for collection information for ReplayBuffers
but can be used on their own as well. Collecting information in this manner is a common
Reinforcement learning paradigm. The internal module used in the attr module. This is wrapped
by theElement class. So the client only has to import one module.
"""

class NumpyElementMixin(object):
    """ Mixin for a Element that uses Numpy.ndarray
    attrs
    """

    @staticmethod
    def _validate(dtype):
        def valid(instance, attribute, value):
            if not isinstance(value, np.ndarray):
                raise ValueError("%s must be of type np.ndarray" % attribute.name)
            if not value.dtype == dtype:
                raise ValueError("%s must have dtype %s" % (attribute.name, str(dtype)))
        return valid

    @classmethod
    def np_attr(cls, dtype):

        """ Represents a component of a Element that is a np.ndarray """
        return attr.ib(kw_only=True,
                        validator=cls._validate(dtype),
                        metadata={"np_attr" : True})

    @classmethod
    def reduce(cls, element_list):
        """ Fetches np_element's from each BufferType in the list
        and splits into a BufferElement containg th

            Args:
                element_list: list of BufferElement's to reduce

            Returns:
                reduced Element

            Raises:
                ValueError: for missing attrs or fails to stack

        """
        if not hasattr(cls, "make_element_from_dict"):
            raise ValueError("cls must have attribute 'make_element_from_dict'. \
                Sugggestion: subclass Element")

        np_attrs = filter(lambda x: ("np_attr" in x.metadata), list(cls.__attrs_attrs__))
        np_attrs_dict = {x.name: [] for x in np_attrs}

        for element in element_list:
            for name, name_list in np_attrs_dict.items():
                name_list.append(getattr(element, name))

        np_attrs_dict_stacked = {k: np.vstack(v) for k, v in np_attrs_dict.items()} # TODO: ,aybe wrap this exception (fail to stack)

        return cls.make_element_from_dict(np_attrs_dict_stacked)



class NormalizingElementMixin(object):
        """Mixin for a Element that allows for Normalization of some or all
        of it's attributes. A lot RL algorithms utilizing Approximators require
        Normalization of their inputs
        """

        @staticmethod
        def running_variance(sums, sums_sqr, num):
            """Computes running variance from running sum of data, running sum of squares of data and data count

                Args:
                    sums: running sum of data
                    sums_sqr: running sum of square of data
                    num: running count of data

                Returns:
                    running variance
            """

            return (1 / num) * (sums_sqr - ( (sums ** 2) / num) )


        @classmethod
        def make_stats(cls, normalize_attrs=(), **kwargs_to_make_element_zero):
            """ Returns a dictionary of attrs to keep track of for
            normalizing. Used internally by this mixin
                Args:
                    normalize_attrs: tuple of str representing which attrs to normalize
                    kwargs_to_make_element_zero: args to pass to make_element_zero

                Returns:
                    normalizing: dict containg attrs to keep track of

                Raises:
                    ValueError: missing attrs, or bad attrs
            """

            if not normalize_attrs:
                raise ValueError("normalize_attrs expects some attrs to normalize")

            if not hasattr(cls, "make_element_zero"):
                raise ValueError("Mixin expects subclass to contain attribute make_element_zero, \
                    should subclass Element")



            if not hasattr(cls, "__attrs_attrs__"):
                raise ValueError("Object returned from make_element_zero must have attribute __attrs_attrs__, \
                    should subclass Element and decorated with BufferElement.element")

            normalizing = {}

            attrs_to_normalize = filter(lambda x: x.name in normalize_attrs, cls.__attrs_attrs__)

            for attribute in attrs_to_normalize:
                attr_name = attribute.name
                normalizing[attr_name] = (getattr(cls.make_element_zero(**kwargs_to_make_element_zero), attr_name),
                                            getattr(cls.make_element_zero(**kwargs_to_make_element_zero), attr_name))

            return normalizing

        @staticmethod
        def update_normalize_stats(stats, element):
            """ Updates a stats object with attrs from a new element

                    Args:
                        stats: as returned from make_stats
                        element: element to aggregate

                    mutates stats
            """

            for attribute_name, stat_tup in stats.items():
                sums, sums_sqr = stat_tup

                value = getattr(element, attribute_name)
                sums_sqr += value ** 2
                sums += value

        @classmethod
        def normalize_element(cls, stats, number_elements, element, eps=0.01):
            """ Normalizes an element based on stats.
                    To take advantage of numpy broadcasting, look for
                    reduce method in mixins

                    Args:
                        stats: as returned from make_stats
                        number_elements: number of elements used to update stats
                        element: element to normalize
                        eps: divide-by-zero protection threshold
            """
            if not hasattr(cls, "make_element_from_dict"):
                raise ValueError("Mixin expects subclass to contain attribute make_element_from_dict, \
                    should subclass Element")

            new_attrs_dict = {attribute.name: getattr(element, attribute.name) for attribute in element.__attrs_attrs__}

            for attr_name, stats_tup in stats.items():
                sums, sums_sqr = stats_tup

                variance = cls.running_variance(sums, sums_sqr, number_elements)
                mean = (1 / number_elements) * sums

                value = getattr(element, attr_name)

                normalized_attr = (value - mean) / np.sqrt(np.maximum(variance, eps))

                new_attrs_dict[attr_name] = normalized_attr

            return cls.make_element_from_dict(new_attrs_dict)




class Element(object, metaclass=ABCMeta):
    """ Represents a type to be placed in a Replay buffer (however, they can be used
    for other purposes as well, and do not require a buffer)
    There purpose is to serve as keepers of steps taken by the agent
    and the associated changes in the environment.
    """

    @staticmethod
    def element(fn):
        """ Wrap attr in Element """
        return attr.s(fn, frozen=True)

    def unzip(self):
        """ Puts attributes to tuple """
        return tuple((getattr(self, attribute.name) for attribute in self.__attrs_attrs__))

    @classmethod
    def make_element_from_dict(cls, dictionary):
        return cls.make_element(**dictionary)

    @abstractmethod
    def make_element_zero(cls,**kwargs):
        """ Factory: Instantiates a 'zeroed' out Element
                **kwargs:
                    arguments specifying extra info
                Returns:
                    instance of 'cls' with 'zeroed' out elements
                        up to the subclass to define what 'zero' means
        """
        raise NotImplementedError()

    @abstractmethod
    def make_element(cls, **kwargs):
        """ Factory: Instantiates a Element of subclass 'cls'
                Up to the subclass how it wants to handle data-sources.
                There just must be at least one 'make' method
                Args:
                    kwargs: specify attrs needed to instantiate

                Returns:
                    instance of 'cls'
        """
        raise NotImplementedError()
