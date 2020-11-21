using Flux

struct MaskedAutoregressiveFlow <: AbstractContinuousFlow
	cα::MADE
	cβ::MADE
	bn::Union{BatchNorm, Nothing}
end

Flux.@functor(MaskedAutoregressiveFlow)

function MaskedAutoregressiveFlow(
		isize::Integer, 
		hsize::Integer, 
		nlayers::Integer, 
		osize::Integer,
		activations,
		ordering::String="sequential";
		lastlayer::String="linear",
		use_batchnorm::Bool=true,
		lastzero::Bool=true,
		seed=time_ns())
	m = MaskedAutoregressiveFlow(
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering,
			ftype=activations.α,
			ptype=(lastlayer == "linear") ? identity : activations.α,
			rs=seed), 
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering, 
			ftype=activations.β,
			ptype=(lastlayer == "linear") ? identity : activations.β,
			rs=seed),
		use_batchnorm ? BatchNorm(isize; momentum=1.0f0) : nothing)
	if lastzero
		m.cα.net[end].W .*= 0.0f0
		m.cβ.net[end].W .*= 0.0f0
	end
	m
end

function (maf::MaskedAutoregressiveFlow)(xl::Tuple)
	X, logJ = xl
	α, β = maf.cα(X), maf.cβ(X)
	# Y = exp.( 0.5 .* β) .* X .+ α
	Y = exp.(-0.5 .* β) .*(X .- α) # inv
	# logJy = logJ .+ 0.5 .* sum(β, dims = 1)
	logJy = logJ .- 0.5 .* sum(β, dims = 1) # inv
	
	if maf.bn !== nothing
		bn = maf.bn
		Z = bn(Y)
		# @info("", X, Y, Z, α, β, exp.(0.5 .* β), hcat(bn.μ, bn.β), hcat(bn.σ², bn.γ))
		logJz = logJy .+ sum(log.(bn.γ)) .- 0.5*sum(log.(bn.σ² .+ bn.ϵ))
		return Z, logJz
	end
	Y, logJy
end

function inv_flow(maf::MaskedAutoregressiveFlow, yl)
	Y, logJ = (maf.bn !== nothing) ? inv_flow(maf.bn, yl) : yl
	D, N = size(Y)
	perm = maf.cα.m[0]
	
	X = zeros(eltype(Y), 0, N)
	for (d, pd) in zip(1:D, perm)
		X_cond = vcat(X, zeros(eltype(Y), D - d + 1, N))[perm, :]
		α, β = maf.cα(X_cond)[pd:pd, :], maf.cβ(X_cond)[pd:pd, :]
		# X = vcat(X, exp.(-0.5 .* β) .* (Y[pd:pd, :] .- α))
		X = vcat(X,  exp.(0.5 .* β) .* Y[pd:pd, :] .+ α) # inv
		# logJ .-= 0.5 .* β
		logJ .+= 0.5 .* β # inv
	end
	
	X[perm, :], logJ
end

function Base.show(io::IO, maf::MaskedAutoregressiveFlow)
	print(io, "MaskedAutoregressiveFlow(cα=")
	print(io, maf.cα)
	print(io, ", cβ=")
	print(io, maf.cβ)
	(maf.bn !== nothing) && print(io, ", ", maf.bn)
	print(io, ")")
end