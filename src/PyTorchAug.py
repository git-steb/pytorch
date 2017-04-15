from __future__ import print_function
from collections import OrderedDict
import threading
import PyTorch
import PyTorchLua
from PyTorchHelpers import *

nextObjectId = 1
luaClasses = {}
luaClassesReverse = {}

# this is so we can ctrl-c lua functions.  we have to run them in a separate thread
# for this, so that the python event log can continue ('event loop' might not be quite
# the right technical term, but the concept is approximately right.  I think)
def interruptableCall(function, args):
    mythread = threading.Thread(target=function, args=args)
    mythread.daemon = True
    mythread.start()
    while mythread.isAlive():
        mythread.join(0.1)
        # print('join timed out')


def getNextObjectId():
    global nextObjectId
    res = nextObjectId
    nextObjectId += 1
    return res


def pushGlobal(lua, name1, name2=None, name3=None):
    lua.getGlobal(name1)
    if name2 is None:
        return
    lua.getField(-1, name2)
    lua.remove(-2)
    if name3 is None:
        return
    lua.getField(-1, name3)
    lua.remove(-2)


def pushGlobalFromList(lua, nameList):
    lua.getGlobal(nameList[0])
    for name in nameList[1:]:
        lua.getField(-1, name)
        lua.remove(-2)


def popString(lua):
    res = lua.toString(-1)
    lua.remove(-1)
    return res.decode('utf-8')


def registerObject(lua, myobject):
    lua.pushNumber(myobject.__objectId)
    lua.insert(-2)
    lua.setRegistry()


def unregisterObject(lua, myobject):
    lua.pushNumber(myobject.__objectId)
    lua.pushNil()
    lua.setRegistry()


def pushObject(lua, myobject):
    lua.pushNumber(myobject.__objectId)
    lua.getRegistry()


def torchType(lua, pos):
    lua.pushValue(-1)
    pushGlobal(lua, "torch", "type")
    lua.insert(-2)
    lua.call(1, 1)
    return popString(lua)


def pushSomething(lua, something):
    if isinstance(something, int):
        lua.pushNumber(something)
        return

    if isinstance(something, float):
        lua.pushNumber(something)
        return

    if isinstance(something, str):
        lua.pushString(something)
        return

    if isinstance(something, dict):
        pushTable(lua, something)
        return

    if isinstance(something, (list, tuple)):
        pushTable(lua, OrderedDict(zip(range(1, len(something) + 1), something)))
        return

    for pythonClass in pushFunctionByPythonClass:
        if isinstance(something, pythonClass):
            pushFunctionByPythonClass[pythonClass](something)
            return

    if type(something) in luaClassesReverse:
        pushObject(lua, something)
        return

    typestring = str(type(something))
    if typestring in ["<class 'numpy.ndarray'>", "<type 'numpy.ndarray'>"]:
        dtypestr = str(something.dtype)
        if dtypestr == 'float32':
            pushSomething(lua, PyTorch._asFloatTensor(something))
            return
        if dtypestr == 'float64':
            pushSomething(lua, PyTorch._asDoubleTensor(something))
            return
        if dtypestr == 'uint8':
            pushSomething(lua, PyTorch._asByteTensor(something))
            return
        raise Exception('pushing numpy array with elements of type ' + dtypestr + ' it not currently implemented')

    raise Exception('pushing type ' + str(type(something)) + ' not implemented, value ', something)


def popSomething(lua, self=None, name=None):
    lua.pushValue(-1)
    pushGlobal(lua, 'torch', 'type')
    lua.insert(-2)
    lua.call(1, 1)
    typestring = popString(lua)

    if typestring in cythonClasses:
        popFunction = cythonClasses[typestring]['popFunction']
        res = popFunction()
        return res

    if typestring == 'number':
        res = lua.toNumber(-1)
        lua.remove(-1)
        return res

    if typestring == 'string':
        res = popString(lua)
        return res

    if typestring == 'table':
        return popTable(lua)

    if typestring in luaClasses:
        returnobject = luaClasses[typestring]('__FROMLUA__')
        registerObject(lua, returnobject)
        return returnobject

    if typestring == 'function':
        def mymethod(*args):
            topStart = lua.getTop()
            pushObject(lua, self)
            lua.getField(-1, name)
            lua.insert(-2)
            for arg in args:
                pushSomething(lua, arg)
            res = lua.pcall(len(args) + 1, 1, 1)   # +1 for self
            if res != 0:
                errorMessage = popString(lua)
                raise Exception(errorMessage)
            res = popSomething(lua)
            topEnd = lua.getTop()
            assert topStart == topEnd
            return res
        lua.remove(-1)
        return mymethod

    if typestring == 'boolean':
        res = lua.toBoolean(-1)
        lua.remove(-1)
        return bool(res)

    if typestring == 'userdata':
        pushGlobal(lua,'swig_wrap')
        lua.insert(-2)
        lua.call(1, 1)
        return popSomething(lua, self, name)

    if typestring == 'nil':
        lua.remove(-1)
        return None

    raise Exception('pop type ' + str(typestring) + ' not implemented')


def pushTable(lua, table):
    lua.newTable()
    for k, v in table.items():
        pushSomething(lua, k)
        pushSomething(lua, v)
        lua.setTable(-3)


def popTable(lua):
    res = {}
    lua.pushNil()
    while lua.next(-2) != 0:
        value = popSomething(lua)
        lua.pushValue(-1)
        key = popSomething(lua)
        res[key] = value
    lua.remove(-1)
    return res


def save(filepath, target):
    lua = PyTorch.getGlobalState().getLua()

    topStart = lua.getTop()

    pushGlobal(lua, 'torch', 'saveobj')
    pushSomething(lua, filepath)
    pushSomething(lua, target)
    res = lua.pcall(2, 0, 1)
    if res != 0:
        errorMessage = popString(lua)
        raise Exception(errorMessage)

    topEnd = lua.getTop()
    assert topStart == topEnd


def load(filepath):
    lua = PyTorch.getGlobalState().getLua()
    topStart = lua.getTop()

    pushGlobal(lua, 'torch', 'loadobj')
    pushSomething(lua, filepath)

    res = lua.pcall(1, 1, 1)
    if res != 0:
        errorMessage = popString(lua)
        raise Exception(errorMessage)

    res = popSomething(lua)

    topEnd = lua.getTop()
    assert topStart == topEnd

    return res

class LuaClass(object):
    def __init__(self, nameList, *args):
        lua = PyTorch.getGlobalState().getLua()
        self.__dict__['__objectId'] = getNextObjectId()
        topStart = lua.getTop()
        #print('nameList', nameList)
        #print('args', args)
        pushGlobalFromList(lua, nameList)
        for arg in args:
            pushSomething(lua, arg)
        res = lua.pcall(len(args), 1)
        if res != 0:
            errorMessage = popString(lua)
            raise Exception(errorMessage)
#        lua.call(len(args), 1)
        registerObject(lua, self)

        topEnd = lua.getTop()
        assert topStart == topEnd

    # def __del__(self):
        # name = self.__class__.__name__

    def __repr__(self):
        topStart = lua.getTop()
        luaClass = self.luaclass
        if luaClass == 'table':
            return 'table'
        splitLuaClass = luaClass.split('.')
        if len(splitLuaClass) == 1:
            pushGlobal(lua, splitLuaClass[0], '__tostring')
        elif len(splitLuaClass) == 2:
            pushGlobal(lua, splitLuaClass[0], splitLuaClass[1], '__tostring')
        else:
            raise Exception('not implemented: luaclass with more than 2 parts ' + luaClass)
        pushObject(lua, self)
        lua.call(1, 1)
        res = popString(lua)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return res

    def __dir__(self):
        topStart = lua.getTop()
        attributes = []
        pushObject(lua, self)
        lua.pushNil()
        while(lua.next(-2)) != 0:
            keyname = lua.toString(-2)
            attributes.append(keyname)
            lua.remove(-1)
        lua.remove(-1)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return attributes

    def __getattr__(self, name):
        if name == '__objectId':
            return self.__dict__['__objectId']
        topStart = lua.getTop()
        pushObject(lua, self)
        lua.getField(-1, name)
        lua.remove(-2)
        res = popSomething(lua, self, name)
        topEnd = lua.getTop()
        assert topStart == topEnd
        return res

def loadModuleClass(module,lua_classname,load_module=True,makeLookupKey = lambda m,c: m+'.'+c):
    if load_module:
        PyTorch.require(module)
    class LuaWrapper(LuaClass):
        def __init__(self, *args):
            _fromLua = False
            if len(args) >= 1:
                if args[0] == '__FROMLUA__':
                   _fromLua = True
                   args = args[1:]
            #print('LuaWrapper.__init__', lua_classname, 'fromLua', _fromLua, 'args', args)
            #print([module,lua_classname])
            self.luaclass = makeLookupKey(module,lua_classname)
            if not _fromLua:
                splitNames = self.luaclass.split('.')
                LuaClass.__init__(self, splitNames, *args)                
            else:
                self.__dict__['__objectId'] = getNextObjectId()
    renamedClass = PyTorchLua.renameClass(LuaWrapper, module, lua_classname)
    return renamedClass

def setupModuleClass(moduleName,
                     moduleClassName,
                     makeLookupKey = lambda m,c: m+'.'+c):
    moduleClass = loadModuleClass(moduleName,moduleClassName,False,makeLookupKey)
    globals()[moduleClassName] = moduleClass
    lookupKey = makeLookupKey(moduleName, moduleClassName)
    moduleClass.luaclass = lookupKey
    luaClasses[lookupKey] = moduleClass
    luaClassesReverse[moduleClass] = lookupKey
    return moduleClass

def setupGlobalModuleClass(moduleName,
                           moduleClassName):
    return setupModuleClass(moduleName,
                            moduleClassName,
                            lambda m,c: c)

class Module(object):
    def __init__(self,moduleName,load_module=True):
        if load_module:
            PyTorch.require(moduleName)
        self.classes = {}
        self.moduleName = moduleName

    setupModuleClass = staticmethod(setupModuleClass)

    def __getattr__(self, name):
        if name not in self.classes:
            self.classes[name] = self.setupModuleClass(self.moduleName,name)
        thisClass = self.classes[name]
        return thisClass

class ModuleGlobal(Module):
    setupModuleClass = staticmethod(setupGlobalModuleClass)

def populateLuaClassesReverse():
    luaClassesReverse.clear()
    for name in luaClasses:
        classtype = luaClasses[name]
        luaClassesReverse[classtype] = name

lua = PyTorch.getGlobalState().getLua()
ret = lua.loadBufferAndCall("""
local SwigWrap = torch.class('SwigWrap')
function swig_wrap(rv)
   if torch.type(rv) == 'userdata' then
      return SwigWrap.new(rv)
   else
      return rv
   end
end
function SwigWrap:__init(ud)
   self.__userdata__ = ud
   self:registerSwigMetatable(getmetatable(ud))
end
function SwigWrap:registerSwigMetatable(mt)
   for k,v in pairs(mt['.bases']) do
      self:registerSwigMetatable(v)
   end
   for k,v in pairs(mt['.fn']) do
      self[k] = function(b,...) return swig(v(b.__userdata__,...)) end
   end
   for k,v in pairs(mt['.get']) do
      self['get_'+tostring(k)] = function(b) return swig(v(b.__userdata__)) end
   end
   for k,v in pairs(mt['.set']) do
      self['set_'+tostring(k)] = function(b,...) return swig(v(b.__userdata__,...)) end
   end
end
""","swigwrap")
assert(ret == 0)

swigwrap = ModuleGlobal('swigwrap',False)
swigwrap.SwigWrap

nn = Module('nn')

cythonClasses = {}
cythonClasses['torch.FloatTensor'] = {'popFunction': PyTorch._popFloatTensor}
cythonClasses['torch.DoubleTensor'] = {'popFunction': PyTorch._popDoubleTensor}
cythonClasses['torch.ByteTensor'] = {'popFunction': PyTorch._popByteTensor}

pushFunctionByPythonClass = {}
pushFunctionByPythonClass[PyTorch._FloatTensor] = PyTorch._pushFloatTensor
pushFunctionByPythonClass[PyTorch._DoubleTensor] = PyTorch._pushDoubleTensor
pushFunctionByPythonClass[PyTorch._ByteTensor] = PyTorch._pushByteTensor
