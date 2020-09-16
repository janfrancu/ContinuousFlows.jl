module ContinuousFlows

include("utils.jl")
export inv_flow

include("maf/maf.jl")
include("maf/made.jl")
export MAF, MADE

include("real_nvp.jl")
export RealNVP

end
