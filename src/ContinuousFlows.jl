module ContinuousFlows

abstract type AbstractContinuousFlow end

include("utils.jl")
export inv_flow

include("nets/made.jl")
export MADE

include("maf.jl")
export MaskedAutoregressiveFlow

include("real_nvp.jl")
export RealNVP

end
