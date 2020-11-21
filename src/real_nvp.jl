using Flux, Flux.Zygote

struct RealNVP{A,B,Vb<:AbstractArray{Bool,1}} <: AbstractContinuousFlow
	cα::A  # location conditioner
	cβ::B  # scale conditioner
	mask::Vb
	pβ::Union{Vector{Float32}, Nothing}
	bn::Union{BatchNorm, Nothing}
end

function RealNVP(isize::Int, 
				conditioner_builder, 
				even=true; 
				use_batchnorm::Bool=true,
				tanh_scaling::Bool=true,
				lastzero::Bool=true)
	mask = even ? (mod.(1:isize, 2) .== 0) : (mod.(1:isize, 2) .== 1)
	d = sum(mask)
	cα = conditioner_builder.α(d, isize - d)
	cβ = conditioner_builder.β(d, isize - d)
	if lastzero
		cα[end].W .*= 0.0f0
		cβ[end].W .*= 0.0f0
	end
	RealNVP(
		cα, cβ, mask, 
		tanh_scaling ? [0.0f0, 1.0f0] : nothing,
		use_batchnorm ? BatchNorm(isize; momentum=1.0f0) : nothing)
end

function (nvp::RealNVP)(xl)
	X, logJ = xl
	X_cond = X[nvp.mask,:]
	α, β = nvp.cα(X_cond), nvp.cβ(X_cond)
	βₜ = (nvp.pβ !== nothing) ? nvp.pβ[2] .* tanh.(β) .+ nvp.pβ[1] : β
	Y = exp.(-0.5 .* βₜ) .* (X[.~nvp.mask,:] .- α) # inv
	# Y = α .+ exp.(0.5 .* βₜ) .* X[.~nvp.mask,:]
	Z = _cat_with_mask(X_cond, Y, nvp.mask)
	logJz = logJ .- 0.5 .* sum(βₜ, dims=1) # inv
	# logJz = logJ .+ 0.5 .* sum(βₜ, dims=1)

	if nvp.bn !== nothing
		bn = nvp.bn
		ZZ = bn(Z)
		# @info("", X, Y, Z, α, βₜ, exp.(-0.5 .* βₜ), bn.μ, bn.σ²)
		logJzz = logJz .+ sum(log.(bn.γ)) .- 0.5*sum(log.(bn.σ² .+ bn.ϵ))
		return ZZ, logJzz
	end
	Z, logJz
end

function inv_flow(nvp::RealNVP, yl)
	Y, logJ = (nvp.bn !== nothing) ? inv_flow(nvp.bn, yl) : yl
	Y_cond = Y[nvp.mask,:]
	α, β = nvp.cα(Y_cond), nvp.cβ(Y_cond)
	βₜ = (nvp.pβ !== nothing) ? nvp.pβ[2] .* tanh.(β) .+ nvp.pβ[1] : β
	X = exp.(0.5 .* βₜ) .* Y[.~nvp.mask,:] .+ α # inv
	# X = exp.(-0.5 .* βₜ) .* (Y[.~nvp.mask,:] .- α)
	_cat_with_mask(Y_cond, X, nvp.mask), logJ .+ 0.5 .* sum(βₜ, dims=1) # inv
	# _cat_with_mask(Y_cond, X, nvp.mask), logJ .- 0.5 .* sum(βₜ, dims=1)
end

Flux.@functor RealNVP
Flux.trainable(nvp::RealNVP) = (nvp.cα, nvp.cβ, nvp.bn, nvp.pβ, )

function _cat_with_mask(x1, x2, mask)
	M1, N = size(x1)
	M2, _ = size(x2)
	Y = similar(x1, M1+M2, N)
	Y[mask,:] .= x1
	Y[.~mask, :] .= x2
	Y
end

Zygote.@adjoint _cat_with_mask(x1, x2, mask) = _cat_with_mask(x1, x2, mask), Δ -> begin
	(Δ[mask,:], Δ[.~mask,:], nothing)
end

function Base.show(io::IO, nvp::RealNVP)
	print(io, "RealNVP(cα=", nvp.cα)
	print(io, ", cβ=", nvp.cβ)
	(nvp.bn !== nothing) && print(io, ", ", nvp.bn)
	print(io, ")")
end