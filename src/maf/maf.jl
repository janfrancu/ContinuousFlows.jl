struct MAF{A,B}
	fμ::A 
	fα::B
end

Flux.@functor(MAF)

function MAF(in::Integer, hs, out::Integer, nat_ord::Bool = false, num_masks = 1, rs = time_ns())
	MAF(MADE(in, hs, out, nat_ord, num_masks, rs), MADE(in, hs, out, nat_ord, num_masks, rs))
end

function (m::MAF)(xx::Tuple)
	x, logdet = xx
	μ = m.fμ(x)
	α = m.fα(x)
	# α = min.(α, 25f0)
	# α = max.(α, -25f0)
	# @show (minimum(α),maximum(α))
	u = exp.(0.5 .* α) .* (x - μ)
	u, logdet .+ 0.5  * sum(α, dims = 1)
end
