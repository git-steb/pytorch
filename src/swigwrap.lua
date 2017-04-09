require('torch')

-- Wrap any C++ instance (type userdata in Lua) that originates from
-- an interface that was produced by Swig,
-- to create a SwigWrap Lua object with all the member functions
-- of the original C++ class.

-- Using a single Lua class import PyTorchAug.setupModuleClass('swigwrap','SwigWrap')
-- all Lua classes that are based on Swig-type userdata, can then be used in PyTorch,
-- with auto-completion of members on any instantiated object.

-- TODO: PyTorchAug.popSomething could automatically wrap userdata in Lua
-- and then pop it as a SwigWrap instance.
-- Until then, use swigwrap.swig(ob) inside your lua code to create instances
-- that PyTorch can pop and convert into Python objects.

local SwigWrap = torch.class('SwigWrap')

function swig(rv)
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
