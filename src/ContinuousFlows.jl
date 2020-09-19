module ContinuousFlows

include("utils.jl")
export inv_flow

include("maf/made.jl")
include("maf/maf.jl")
export MaskedAutoregressiveFlow, MADE

include("real_nvp.jl")
export RealNVP

end
