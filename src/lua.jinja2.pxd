# {{header1}}
# {{header2}}

from PyTorch cimport *

cdef extern from "LuaHelper.h":
    void *getGlobal2(lua_State *L, const char *name1, const char *name2);
    void luaRequire(lua_State *L, const char *name)
    void *getGlobal1(lua_State *L, const char *name1);
    int getLuaRegistryIndex()

{% for typedict in types %}
{% set Real = typedict['Real'] %}
{% set real = typedict['real'] %}
{% if Real in ['Double', 'Float', 'Byte'] %}
cdef extern from "LuaHelper.h":
    TH{{Real}}Tensor *pop{{Real}}Tensor(lua_State *L)
    void push{{Real}}Tensor(lua_State *L, TH{{Real}}Tensor *tensor)
{% endif %}
{% endfor %}

cdef extern from "lua_externc.h":
    struct lua_State
    void lua_pushnumber(lua_State *L, float number)
    float lua_tonumber(lua_State *L, int index)
    int lua_toboolean(lua_State *L, int index)
    void lua_pushstring(lua_State *L, const char *value)
    const char *lua_tostring(lua_State *L, int index)
    void lua_call(lua_State *L, int argsIn, int argsOut) nogil
    int lua_pcall(lua_State *L, int nargs, int nresults, int errfunc) nogil
    void lua_remove(lua_State *L, int index)
    void lua_insert(lua_State *L, int index)
    void lua_getglobal(lua_State *L, const char *name)
    void lua_setglobal(lua_State *L, const char *name)
    void lua_newtable(lua_State *L)
    void lua_settable(lua_State *L, int index)
    void lua_gettable(lua_State *L, int index)
    void lua_getfield(lua_State *L, int index, const char *name)
    void lua_pushnil(lua_State *L)
    int lua_type(lua_State *L, int index)
    const char *lua_typename(lua_State *L, int tp)
    void lua_pushvalue(lua_State *L, int index)
    int lua_next(lua_State *L, int index)
    int lua_gettop(lua_State *L)
    int lua_isuserdata(lua_State *L, int index)

cdef class LuaState(object):
    cdef lua_State *L

cdef LuaState_fromNative(lua_State *L)

