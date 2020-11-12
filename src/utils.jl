using Flux

function inv_flow(c::Flux.Chain, x)
    inv_flow_chain(reverse(c.layers), x)
end

inv_flow_chain(::Tuple{}, x) = x
inv_flow_chain(fs::Tuple, x) = inv_flow_chain(Base.tail(fs), inv_flow(first(fs), x))

function inv_flow(bn::Flux.BatchNorm, yl)
    Y, logJ = yl
    X = bn.μ .+ sqrt.(bn.σ² .+ bn.ϵ).*(Y .- bn.β) ./ bn.γ
    X, logJ .- sum(log.(bn.γ)) .+ 0.5*sum(log.(bn.σ² .+ bn.ϵ))
end
