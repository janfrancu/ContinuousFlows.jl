using Flux

function inv_flow(c::Flux.Chain, x)
    inv_flow_chain(reverse(c.layers), x)
end

inv_flow_chain(::Tuple{}, x) = x
inv_flow_chain(fs::Tuple, x) = inv_flow_chain(Base.tail(fs), inv_flow(first(fs), x))
