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
		use_batchnorm::Bool=true,
		seed=time_ns())
	MaskedAutoregressiveFlow(
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering,
			ftype=activations.α,
			rs=seed), 
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering, 
			ftype=activations.β,
			rs=seed),
		use_batchnorm ? BatchNorm(isize) : nothing)
end

function (maf::MaskedAutoregressiveFlow)(xl::Tuple)
	X, logJ = xl
	α, β = maf.cα(X), maf.cβ(X)
	Y = α .+ exp.(0.5 .* β) .* X
	logJy = logJ .+ 0.5 .* sum(β, dims = 1)
	
	if maf.bn !== nothing
		bn = maf.bn
		Z = bn(Y)
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
		X = vcat(X, (Y[pd:pd, :] .- α) ./ exp.(0.5 .* β))
		logJ .-= 0.5 .* β
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