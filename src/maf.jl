using Flux

struct MaskedAutoregressiveFlow <: AbstractContinuousFlow
	cα::MADE
	cβ::MADE
end

Flux.@functor(MaskedAutoregressiveFlow)

function MaskedAutoregressiveFlow(
		isize::Integer, 
		hsize::Integer, 
		nlayers::Integer, 
		osize::Integer,
		ordering::String = "sequential")
	seed = time_ns()
	MaskedAutoregressiveFlow(
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering,
			ftype=relu,
			rs=seed), 
		MADE(
			isize, 
			fill(hsize, nlayers-1), 
			osize, 
			ordering, 
			ftype=relu,
			ptype=exp,
			rs=seed),
		)
end

function (maf::MaskedAutoregressiveFlow)(xl::Tuple)
	X, logJ = xl
	α, β = maf.cα(X), maf.cβ(X)
	Y = α .+ β .* X
	Y, logJ .+ sum(log.(abs.(β)), dims = 1)
end

function inv_flow(maf::MaskedAutoregressiveFlow, yl)
	Y, logJ = yl
	D, N = size(Y)
	perm = maf.cα.m[0]
	
	X = zeros(eltype(Y), 0, N)
	for (d, pd) in zip(1:D, perm)
		X_cond = vcat(X, zeros(eltype(Y), D - d + 1, N))[perm, :]
		α, β = maf.cα(X_cond)[pd:pd, :], maf.cβ(X_cond)[pd:pd, :]
		X = vcat(X, (Y[pd:pd, :] .- α) ./ β)
		logJ .-= log.(abs.(β))
	end
	
	X[perm, :], logJ
end