# {{header1}}
# {{header2}}

from __future__ import print_function, division
import numbers
import cython
cimport cython

import numpy as np

cimport cpython.array
import array

from math import log10, floor

cimport Storage
import Storage
from lua cimport *
from nnWrapper cimport *
cimport PyTorch
# import Storage
# from Storage cimport *

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# from http://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

cdef extern from "THRandom.h":
    cdef struct THGenerator
    void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_)

def manualSeed(long seed):
    THRandom_manualSeed(globalState.generator, seed)

cdef floatToString(float floatValue):
    return '%.6g'% floatValue

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
_{{Real}}Storage = Storage._{{Real}}Storage
{% endfor %}

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
cdef extern from "THTensor.h":
    cdef struct TH{{Real}}Tensor
    TH{{Real}}Tensor *TH{{Real}}Tensor_new()
    TH{{Real}}Tensor *TH{{Real}}Tensor_newClone(TH{{Real}}Tensor *self)
    {{real}} *TH{{Real}}Tensor_data(TH{{Real}}Tensor *self)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newContiguous(TH{{Real}}Tensor *self)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize1d(long size0)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize2d(long size0, long size1)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize3d(long size0, long size1, long size2)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithSize4d(long size0, long size1, long size2, long size3)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newWithStorage(Storage.TH{{Real}}Storage *storage_, long storageOffset_, Storage.THLongStorage *size_, Storage.THLongStorage *stride_)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage1d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage2d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0, long size1, long stride1)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage3d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0, long size1, long stride1,
        long size2, long stride2)
    TH{{Real}}Tensor* TH{{Real}}Tensor_newWithStorage4d(Storage.TH{{Real}}Storage *storage, long storageOffset, long size0, long stride0, long size1, long stride1,
        long size2, long stride2, long size3, long stride3)
    void TH{{Real}}Tensor_retain(TH{{Real}}Tensor *self)
    void TH{{Real}}Tensor_free(TH{{Real}}Tensor *self)

    int TH{{Real}}Tensor_nDimension(TH{{Real}}Tensor *tensor)
    void TH{{Real}}Tensor_resizeAs(TH{{Real}}Tensor *self, TH{{Real}}Tensor *model)
    void TH{{Real}}Tensor_resize1d(TH{{Real}}Tensor *self, long size0)
    void TH{{Real}}Tensor_resize2d(TH{{Real}}Tensor *self, long size0, long size1)
    void TH{{Real}}Tensor_resize3d(TH{{Real}}Tensor *self, long size0, long size1, long size2)
    void TH{{Real}}Tensor_resize4d(TH{{Real}}Tensor *self, long size0, long size1, long size2, long size3)
    long TH{{Real}}Tensor_size(const TH{{Real}}Tensor *self, int dim)
    long TH{{Real}}Tensor_nElement(TH{{Real}}Tensor *self)
    long TH{{Real}}Tensor_stride(const TH{{Real}}Tensor *self, int dim)
    int TH{{Real}}Tensor_isContiguous(const TH{{Real}}Tensor *self)

    void TH{{Real}}Tensor_set1d(const TH{{Real}}Tensor *tensor, long x0, float value)
    void TH{{Real}}Tensor_set2d(const TH{{Real}}Tensor *tensor, long x0, long x1, float value)
    {{real}} TH{{Real}}Tensor_get1d(const TH{{Real}}Tensor *tensor, long x0)
    {{real}} TH{{Real}}Tensor_get2d(const TH{{Real}}Tensor *tensor, long x0, long x1)

    void TH{{Real}}Tensor_fill(TH{{Real}}Tensor *self, {{real}} value)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newSelect(TH{{Real}}Tensor *self, int dimension, int sliceIndex)
    TH{{Real}}Tensor *TH{{Real}}Tensor_newNarrow(TH{{Real}}Tensor *self, int dimension, long firstIndex, long size)
    Storage.TH{{Real}}Storage *TH{{Real}}Tensor_storage(TH{{Real}}Tensor *self)

    void TH{{Real}}Tensor_tanh(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t)
    void TH{{Real}}Tensor_sigmoid(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t)
    void TH{{Real}}Tensor_cinv(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t)
    void TH{{Real}}Tensor_neg(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t)
    void TH{{Real}}Tensor_abs(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t)

    void TH{{Real}}Tensor_eqTensor(THByteTensor *r_, TH{{Real}}Tensor *ta, TH{{Real}}Tensor *tb)

    void TH{{Real}}Tensor_add(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_div(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_mul(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value)

    void TH{{Real}}Tensor_add(TH{{Real}}Tensor *tensorSelf, TH{{Real}}Tensor *tensorOne, {{real}} value)

    void TH{{Real}}Tensor_cadd(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, {{real}} value, TH{{Real}}Tensor *second)
    void TH{{Real}}Tensor_cmul(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, TH{{Real}}Tensor *src)
    void TH{{Real}}Tensor_cdiv(TH{{Real}}Tensor *r_, TH{{Real}}Tensor *t, TH{{Real}}Tensor *src)

    void TH{{Real}}Tensor_cmaxValue(TH{{Real}}Tensor *r, TH{{Real}}Tensor *t, {{real}} value)
    void TH{{Real}}Tensor_cminValue(TH{{Real}}Tensor *r, TH{{Real}}Tensor *t, {{real}} value)

    {{real}} TH{{Real}}Tensor_minall(TH{{Real}}Tensor *t)
    {{real}} TH{{Real}}Tensor_maxall(TH{{Real}}Tensor *t)
    {{real}} TH{{Real}}Tensor_sumall(TH{{Real}}Tensor *t)

    void TH{{Real}}Tensor_geometric(TH{{Real}}Tensor *self, THGenerator *_generator, double p)
    void TH{{Real}}Tensor_bernoulli(TH{{Real}}Tensor *self, THGenerator *_generator, double p)

    {% if Real in ['Float', 'Double'] %}
    void TH{{Real}}Tensor_addmm(TH{{Real}}Tensor *tensorSelf, double beta, TH{{Real}}Tensor *tensorOne, double alpha, TH{{Real}}Tensor *mat1, TH{{Real}}Tensor *mat2)

    void TH{{Real}}Tensor_uniform(TH{{Real}}Tensor *self, THGenerator *_generator, double a, double b)
    void TH{{Real}}Tensor_normal(TH{{Real}}Tensor *self, THGenerator *_generator, double mean, double stdv)
    void TH{{Real}}Tensor_exponential(TH{{Real}}Tensor *self, THGenerator *_generator, double _lambda);
    void TH{{Real}}Tensor_cauchy(TH{{Real}}Tensor *self, THGenerator *_generator, double median, double sigma)
    void TH{{Real}}Tensor_logNormal(TH{{Real}}Tensor *self, THGenerator *_generator, double mean, double stdv)
    {% endif %}
{% endfor %}

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
cdef class _{{Real}}Tensor(object):
    # properties are in the PyTorch.pxd file

#    def __cinit__(Tensor self, THFloatTensor *tensorC = NULL):
#        self.thFloatTensor = tensorC

    def __cinit__(self, *args, _allocate=True):
#        cdef _{{Real}}Tensor childobject
        cdef TH{{Real}}Tensor *newTensorC
        cdef _{{Real}}Tensor templateObject
        logger.debug('{{Real}}Tensor.__cinit__')
#        cdef TH{{Real}}Storage *storageC
#        cdef long addr
#        if len(kwargs) > 0:
#            raise Exception('cannot provide arguments to initializer')
        if _allocate:
            if len(args) == 1 and isinstance(args[0], _LongStorage):  # it's a size tensor
               self.native = TH{{Real}}Tensor_new()
               self.resize(args[0])
               return
            if len(args) == 1 and isinstance(args[0], _{{Real}}Tensor):
               templateObject = args[0]
               newTensorC = TH{{Real}}Tensor_newClone(templateObject.native)
               self.native = newTensorC
               return
            for arg in args:
                if not isinstance(arg, int):
                    raise Exception('cannot provide arguments to initializer')
            if len(args) == 0:
                # print('no args, calling TH{{Real}}Tensor_new()')
                self.native = TH{{Real}}Tensor_new()
            elif len(args) == 1:
                # print('new tensor 1d length', args[0])
                self.native = TH{{Real}}Tensor_newWithSize1d(args[0])
            elif len(args) == 2:
                # print('args=2')
                self.native = TH{{Real}}Tensor_newWithSize2d(args[0], args[1])
            elif len(args) == 3:
                # print('new tensor 1d length', args[0])
                self.native = TH{{Real}}Tensor_newWithSize3d(args[0], args[1], args[2])
            elif len(args) == 4:
                # print('new tensor 1d length', args[0])
                self.native = TH{{Real}}Tensor_newWithSize4d(args[0], args[1], args[2], args[3])
            else:
                logger.error('Raising exception...')
                raise Exception('Not implemented, len(args)=' + str(len(args)))
#        else:
#            if len(args) > 0:
#                if len(args) > 1:
#                    raise Exception('args for allocate=false must be lenght 1 or 0')
#                if isinstance(args[0], _{{Real}}Tensor):
#                    childobject = args[0]
#                    self.native = childobject.native
#                else:
#                    raise Exception('arg for allocate=0 must be tensor, but was ' + type(args[0]))

#    def __cinit__(self, THFloatTensor *tensorC, Storage storage):
#        self.thFloatTensor = tensorC
#        self.storage = storage

#    def __cinit__(self, Storage storage, offset, size0, stride0, size1, stride1):
#        self.thFloatTensor = THFloatTensor_newWithStorage2d(storage.thFloatStorage, offset, size0, stride0, size1, stride1)
#        self.storage = storage

    def __dealloc__(self):
        cdef int refCount
#        cdef int dims
#        cdef int size
#        cdef int i
#        cdef THFloatStorage *storage
#        logger.debug('__dealloc__ native %s', <long>(self.native) != 0)
        if <long>(self.native) != 0:
            refCount = TH{{Real}}Tensor_getRefCount(self.native)
   #         print('{{Real}}Tensor.dealloc old refcount', refCount)
   #        storage = THFloatTensor_storage(self.thFloatTensor)
   #        if storage == NULL:
   #            # print('   dealloc, storage NULL')
   #        else:
   #            # print('   dealloc, storage ', hex(<long>(storage)))
   #        dims = THFloatTensor_nDimension(self.thFloatTensor)
   #        # print('   dims of dealloc', dims)
   #        for i in range(dims):
   #            # print('   size[', i, ']', THFloatTensor_size(self.thFloatTensor, i))
            if refCount < 1:
                raise Exception('Unallocated an already deallocated tensor... :-O')  # Hmmm, seems this exceptoin wont go anywhere useful... :-P
            TH{{Real}}Tensor_free(self.native)
        else:
            logger.debug('__dealloc__ tensor never allocated')

    def nElement(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_nElement(self.native)

    def asNumpyTensor(_{{Real}}Tensor self):
        cdef Storage._{{Real}}Storage storage
        cdef {{real}} *data
        cdef _{{Real}}Tensor contig
        size = self.size()
        dims = len(size)
        dtype = None
        {% if Real == 'Double' %}dtype=np.float64{% endif %}
        {% if Real == 'Float' %}dtype=np.float32{% endif %}
        {% if Real == 'Byte' %}dtype=np.uint8{% endif %}
        if dtype is None:
          raise Exception("not implemented for {{Real}}")
#        print('dtype', dtype)
        if dims >= 1:
            totalSize = 1
            for d in range(dims - 1, -1, -1):
                totalSize *= size[d]
            myarray = np.zeros(totalSize, dtype=dtype)
            contig = self.contiguous()
            data = contig.data()
            for i in range(totalSize):
                myarray[i] = data[i]
            shape = []
            for d in range(dims):
                shape.append(size[d])
            return myarray.reshape(shape)
        else:
            raise Exception('Not implemented for dims = {dims}'.format(dims=dims))

    @property
    def refCount(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_getRefCount(self.native)

    cdef {{real}} *data(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_data(self.native)

    cpdef int dims(self):
        return TH{{Real}}Tensor_nDimension(self.native)

    cpdef set1d(self, int x0, {{real}} value):
        TH{{Real}}Tensor_set1d(self.native, x0, value)

    cpdef set2d(self, int x0, int x1, {{real}} value):
        TH{{Real}}Tensor_set2d(self.native, x0, x1, value)

    cpdef {{real}} get1d(self, int x0):
        return TH{{Real}}Tensor_get1d(self.native, x0)

    cpdef {{real}} get2d(self, int x0, int x1):
        return TH{{Real}}Tensor_get2d(self.native, x0, x1)

    cpdef int isContiguous(_{{Real}}Tensor self):
        return TH{{Real}}Tensor_isContiguous(self.native)

    cpdef {{real}} max(self):
        return TH{{Real}}Tensor_maxall(self.native)

    cpdef {{real}} min(self):
        return TH{{Real}}Tensor_minall(self.native)

    def __repr__(_{{Real}}Tensor self):
        return self.as_string(self)

    def as_string(_{{Real}}Tensor self, show_size=True):
        # assume 2d matrix for now
        cdef int size0
        cdef int size1
        dims = self.dims()
        if dims == 0:
            return '[torch.{{Real}}Tensor with no dimension]\n'
        elif dims == 2:
            size0 = TH{{Real}}Tensor_size(self.native, 0)
            size1 = TH{{Real}}Tensor_size(self.native, 1)
            res = ''
            for r in range(size0):
                thisline = ''
                for c in range(size1):
                    if c > 0:
                        thisline += ' '
                    {% if Real in ['Float'] %}
                    thisline += floatToString(self.get2d(r,c),)
                    {% else %}
                    thisline += str(self.get2d(r,c),)
                    {% endif %}
                res += thisline + '\n'
            if show_size:
                res += '[torch.{{Real}}Tensor of size ' + ('%.0f' % size0) + 'x' + str(size1) + ']\n'
            return res
        elif dims == 1:
            size0 = TH{{Real}}Tensor_size(self.native, 0)
            res = ''
            thisline = ''
            for c in range(size0):
                if c > 0:
                    thisline += ' '
                {% if Real in ['Float'] %}
                thisline += floatToString(self.get1d(c))
                {% else %}
                thisline += str(self.get1d(c))
                {% endif %}
            res += thisline + '\n'
            if show_size:
                res += '[torch.{{Real}}Tensor of size ' + str(size0) + ']\n'
            return res
        elif dims == 3:
            res = ''
            for d in range(self.size()[0]):
                res += '(' + str(d) + ',.,.) =\n'
                res += self[d].as_string(show_size=False)
            res += '\ntorch.{{Real}}Tensor of size '
            first = True
            for d in self.size():
               if not first:
                  res += 'x'
               res += str(d)
               first = False
            res += ']'
            return res
        else:
            raise Exception("Not implemented: dims > 2")

    def __getitem__(_{{Real}}Tensor self, int index):
        if self.dims() == 1:
            return self.get1d(index)
        cdef TH{{Real}}Tensor *res = TH{{Real}}Tensor_newSelect(self.native, 0, index)
        return _{{Real}}Tensor_fromNative(res, False)

    def __setitem__(_{{Real}}Tensor self, int index, {{real}} value):
        if self.dims() == 1:
            self.set1d(index, value)
        else:
            raise Exception("not implemented")

    def fill(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_fill(self.native, value)
        return self

    def sum(_{{Real}}Tensor self):
        cdef {{real}} result = TH{{Real}}Tensor_sumall(self.native)
        return result

    {% if Real in ['Float', 'Double'] %}

    def itanh(_{{Real}}Tensor self):
        TH{{Real}}Tensor_tanh(self.native, self.native)
        return self

    def isigmoid(_{{Real}}Tensor self):
        TH{{Real}}Tensor_sigmoid(self.native, self.native)
        return self

    def icinv(_{{Real}}Tensor self):
        TH{{Real}}Tensor_cinv(self.native, self.native)
        return self


    def tanh(_{{Real}}Tensor self):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        TH{{Real}}Tensor_tanh(res.native, self.native)
        return res

    def sigmoid(_{{Real}}Tensor self):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        TH{{Real}}Tensor_sigmoid(res.native, self.native)
        return res

    def cinv(_{{Real}}Tensor self):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        TH{{Real}}Tensor_cinv(res.native, self.native)
        return res

    def neg(_{{Real}}Tensor self):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        TH{{Real}}Tensor_neg(res.native, self.native)
        return res

    def ineg(_{{Real}}Tensor self):
        TH{{Real}}Tensor_neg(self.native, self.native)
        return self

    {% endif %}

    {% if Real != 'Byte' %}
    def abs(_{{Real}}Tensor self):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        TH{{Real}}Tensor_abs(res.native, self.native)
        return res

    def iabs(_{{Real}}Tensor self):
        TH{{Real}}Tensor_abs(self.native, self.native)
        return self

    {% endif %}

    def size(_{{Real}}Tensor self):
        cdef int dims = self.dims()
#        cdef LongStorage size
        if dims > 0:
            size = _LongStorage(dims)
            for d in range(dims):
                size[d] = TH{{Real}}Tensor_size(self.native, d)
            return size
        else:
            return None  # not sure how to handle this yet

    @staticmethod
    def new():
#        # print('allocate tensor')
        return _{{Real}}Tensor()
#        return _FloatTensor_fromNative(newTensorC, False)

    def narrow(_{{Real}}Tensor self, int dimension, long firstIndex, long size):
        cdef TH{{Real}}Tensor *narrowedC = TH{{Real}}Tensor_newNarrow(self.native, dimension, firstIndex, size)
        return _{{Real}}Tensor_fromNative(narrowedC, retain=False)


    def contiguous(_{{Real}}Tensor self):
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newContiguous(self.native)
        return _{{Real}}Tensor_fromNative(newTensorC, retain=False)

    def resize1d(_{{Real}}Tensor self, int size0):
        TH{{Real}}Tensor_resize1d(self.native, size0)
        return self

    def resize2d(_{{Real}}Tensor self, int size0, int size1):
        TH{{Real}}Tensor_resize2d(self.native, size0, size1)
        return self

    def resize3d(_{{Real}}Tensor self, int size0, int size1, int size2):
        TH{{Real}}Tensor_resize3d(self.native, size0, size1, size2)
        return self

    def resize4d(_{{Real}}Tensor self, int size0, int size1, int size2, int size3):
        TH{{Real}}Tensor_resize4d(self.native, size0, size1, size2, size3)
        return self

    def resizeAs(_{{Real}}Tensor self, _{{Real}}Tensor model):
        TH{{Real}}Tensor_resizeAs(self.native, model.native)
        return self
    
    def resize(_{{Real}}Tensor self, Storage._LongStorage size):
#        # print('_FloatTensor.resize size:', size)
        if len(size) == 0:
            return self
        cdef int dims = len(size)
#        # print('_FloatTensor.resize dims:', dims)
        if dims == 1:
            TH{{Real}}Tensor_resize1d(self.native, size[0])
        elif dims == 2:
            TH{{Real}}Tensor_resize2d(self.native, size[0], size[1])
        elif dims == 3:
            TH{{Real}}Tensor_resize3d(self.native, size[0], size[1], size[2])
        elif dims == 4:
            TH{{Real}}Tensor_resize4d(self.native, size[0], size[1], size[2], size[3])
        else:
            raise Exception('Not implemented for dims=' + str(dims))
        return self

    @staticmethod
    def newWithStorage(Storage._{{Real}}Storage storage, offset, Storage._LongStorage size, Storage._LongStorage stride):
#        # print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage(storage.native, offset, size.native, stride.native)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage1d(Storage._{{Real}}Storage storage, offset, size0, stride0):
#        # print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage1d(storage.native, offset, size0, stride0)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage2d(Storage._{{Real}}Storage storage, offset, size0, stride0, size1, stride1):
#        # print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage2d(storage.native, offset, size0, stride0, size1, stride1)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage3d(Storage._{{Real}}Storage storage, offset, size0, stride0, size1, stride1, size2, stride2):
#        # print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage3d(storage.native, offset, size0, stride0, size1, stride1,
            size2, stride2)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    @staticmethod
    def newWithStorage4d(Storage._{{Real}}Storage storage, offset, size0, stride0, size1, stride1, size2, stride2,
            size3, stride3):
#        # print('allocate tensor')
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newWithStorage4d(storage.native, offset, size0, stride0, size1, stride1,
            size2, stride2, size3, stride3)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    def clone(_{{Real}}Tensor self):
        cdef TH{{Real}}Tensor *newTensorC = TH{{Real}}Tensor_newClone(self.native)
        return _{{Real}}Tensor_fromNative(newTensorC, False)

    def storage(_{{Real}}Tensor self):
        cdef Storage.TH{{Real}}Storage *storageC = TH{{Real}}Tensor_storage(self.native)
        if storageC == NULL:
            return None
        return Storage._{{Real}}Storage_fromNative(storageC)

    def __add__(_{{Real}}Tensor self, second):
        # assume 2d matrix for now?
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_add(res.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cadd(res.native, self.native, 1, secondTensor.native)
        return res

    def cmul(_{{Real}}Tensor self, second):
#        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        cdef _{{Real}}Tensor secondTensor
        secondTensor = second
        TH{{Real}}Tensor_cmul(self.native, self.native, secondTensor.native)
        return self

    def __sub__(_{{Real}}Tensor self, second):
        # assume 2d matrix for now?
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_add(res.native, self.native, -second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cadd(res.native, self.native, -1, secondTensor.native)
        return res

    def eq(_{{Real}}Tensor self, _{{Real}}Tensor second):
        cdef _ByteTensor res = _ByteTensor.new()
        TH{{Real}}Tensor_eqTensor(res.native, self.native, second.native);
        return res

    def icmin(_{{Real}}Tensor self, second):
      TH{{Real}}Tensor_cminValue(self.native, self.native, second)
      return self

    def icmax(_{{Real}}Tensor self, second):
      TH{{Real}}Tensor_cmaxValue(self.native, self.native, second)
      return self

    {% if Real in ['Float', 'Double'] %}
    def __truediv__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_div(res.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cdiv(res.native, self.native, secondTensor.native)
        return res

    def __itruediv__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_div(self.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cdiv(self.native, self.native, secondTensor.native)
        return self
    {% else %}
    def __floordiv__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor res = _{{Real}}Tensor.new()
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_div(res.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cdiv(res.native, self.native, secondTensor.native)
        return res

    def __ifloordiv__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_div(self.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cdiv(self.native, self.native, secondTensor.native)
        return self
    {% endif %}

    def __iadd__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_add(self.native, self.native, second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cadd(self.native, self.native, 1, secondTensor.native)
        return self

    def __isub__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor secondTensor
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_add(self.native, self.native, -second)
        else:
            secondTensor = second
            TH{{Real}}Tensor_cadd(self.native, self.native, -1, secondTensor.native)
        return self

    def __imul__(_{{Real}}Tensor self, {{real}} value):
        TH{{Real}}Tensor_mul(self.native, self.native, value)
        return self

#    def __mul__(_{{Real}}Tensor self, _{{Real}}Tensor M2):
    def __mul__(_{{Real}}Tensor self, second):
        cdef _{{Real}}Tensor M2
        cdef _{{Real}}Tensor T
        cdef _{{Real}}Tensor res
        cdef int resRows
        cdef int resCols

        res = _{{Real}}Tensor.new()
        if isinstance(second, numbers.Number):
            TH{{Real}}Tensor_mul(res.native, self.native, second)
            return res
        else:
        {% if Real in ['Float', 'Double'] %}
            M2 = second
            T = _{{Real}}Tensor.new()
            resRows = TH{{Real}}Tensor_size(self.native, 0)
            resCols = TH{{Real}}Tensor_size(M2.native, 1)
            res.resize2d(resRows, resCols)
            T.resize2d(resRows, resCols)
            TH{{Real}}Tensor_addmm(res.native, 0, T.native, 1, self.native, M2.native)
            return res
        {% else %}
            raise Exception('Invalid arg type for second: ' + str(type(second)))
        {% endif %}

    # ========== random ===============================

    def bernoulli(_{{Real}}Tensor self, float p=0.5):
        TH{{Real}}Tensor_bernoulli(self.native, globalState.generator, p)
        return self

    def geometric(_{{Real}}Tensor self, float p=0.5):
        TH{{Real}}Tensor_geometric(self.native, globalState.generator, p)
        return self

{% if Real in ['Float', 'Double'] %}
    def normal(_{{Real}}Tensor self, {{real}} mean=0, {{real}} stdv=1):
        TH{{Real}}Tensor_normal(self.native, globalState.generator, mean, stdv)
        return self

    def exponential(_{{Real}}Tensor self, {{real}} _lambda=1):
        TH{{Real}}Tensor_exponential(self.native, globalState.generator, _lambda)
        return self

    def cauchy(_{{Real}}Tensor self, {{real}} median=0, {{real}} sigma=1):
        TH{{Real}}Tensor_cauchy(self.native, globalState.generator, median, sigma)
        return self

    def logNormal(_{{Real}}Tensor self, {{real}} mean=1, {{real}} stdv=2):
        TH{{Real}}Tensor_logNormal(self.native, globalState.generator, mean, stdv)
        return self

    def uniform(_{{Real}}Tensor self, {{real}} a=0, {{real}} b=1):
        TH{{Real}}Tensor_uniform(self.native, globalState.generator, a, b)
        return self
{% endif %}

#    @staticmethod
cdef _{{Real}}Tensor_fromNative(TH{{Real}}Tensor *tensorC, retain=True):
    if retain:
        TH{{Real}}Tensor_retain(tensorC)
    tensor = _{{Real}}Tensor(_allocate=False)
    tensor.native = tensorC
    return tensor

{% if Real in ['Float', 'Double', 'Byte'] %}
def _as{{Real}}Tensor(myarray):
    cdef {{real}}[:] myarraymv
    cdef Storage._{{Real}}Storage storage
    if str(type(myarray)) in ["<type 'numpy.ndarray'>", "<class 'numpy.ndarray'>"]:
        dims = len(myarray.shape)
        if dims >= 1:
            totalSize = 1
            size = Storage._LongStorage.newWithSize(dims)
            stride = Storage._LongStorage.newWithSize(dims)
            strideSoFar = 1
            for d in range(dims - 1, -1, -1):
                totalSize *= myarray.shape[d]
                size[d] = myarray.shape[d]
                stride[d] = strideSoFar
                strideSoFar *= size[d]
            myarraymv = myarray.reshape(totalSize)
            storage = Storage._{{Real}}Storage.newWithData(myarraymv)
            Storage.TH{{Real}}Storage_retain(storage.native) # since newWithData takes ownership

            tensor = _{{Real}}Tensor.newWithStorage(storage, 0, size, stride)
            return tensor
        else:
            raise Exception('dims == {dims} not implemented; please raise an issue'.format(
                dims=dims))
    elif isinstance(myarray, array.array):
        myarraymv = myarray
        storage = Storage._{{Real}}Storage.newWithData(myarraymv)
        Storage.TH{{Real}}Storage_retain(storage.native) # since newWithData takes ownership
        tensor = _{{Real}}Tensor.newWithStorage1d(storage, 0, len(myarray), 1)
        return tensor        
    else:
        raise Exception("not implemented")
{% endif %}

{% endfor %}

cdef class GlobalState(object):
    def __cinit__(GlobalState self):
        pass

    def __dealloc__(self):
        pass

    def getLua(self):
        return LuaState_fromNative(self.L)

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
{% if Real in ['Double', 'Float', 'Byte'] %}
def _pop{{Real}}Tensor():
    global globalState
    cdef TH{{Real}}Tensor *tensorC = pop{{Real}}Tensor(globalState.L)
    return _{{Real}}Tensor_fromNative(tensorC)

def _push{{Real}}Tensor(_{{Real}}Tensor tensor):
    global globalState
    push{{Real}}Tensor(globalState.L, tensor.native)
{% endif %}
{% endfor %}

# there's probably an official Torch way of doing this
{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
{% if Real in ['Double', 'Float'] %}
cpdef int get{{Real}}Prediction(_{{Real}}Tensor output):
    cdef int prediction = 0
    cdef {{real}} maxSoFar = output[0]
    cdef {{real}} thisValue = 0
    cdef int i = 0
    for i in range(TH{{Real}}Tensor_size(output.native, 0)):
        thisValue = TH{{Real}}Tensor_get1d(output.native, i)
        if thisValue > maxSoFar:
            maxSoFar = thisValue
            prediction = i
    return prediction + 1
{% endif %}
{% endfor %}

cdef GlobalState globalState

def getGlobalState():
    global globalState
    return globalState

def require(libName):
    global globalState
    cdef lua_State *L
    L = globalState.L
    luaRequire(L, libName.encode('utf-8'))

def getGlobal(name):
    global globalState
    cdef lua_State *L
    L = globalState.L

def init():
    global globalState
    # print('initializing PyTorch...')
    globalState = GlobalState()
    globalState.L = luaInit()
    globalState.generator = <THGenerator *>(getGlobal2(globalState.L, 'torch', '_gen'))
    # print('generator null:', globalState.generator == NULL)
    # print(' ... PyTorch initialized')

init()

from floattensor import *

# ==== Nn ==================================
cdef class Nn(object):  # just used to provide the `nn.` syntax
    def collectgarbage(self):
        collectGarbage(globalState.L)

#    def Linear(self, inputSize, outputSize):
#        return Linear(inputSize, outputSize)

