using Flux, Flux.Zygote

struct RealNVP{A,B,Vb<:AbstractArray{Bool,1}} <: AbstractContinuousFlow
	cα::A  # location conditioner
	cβ::B  # scale conditioner
	mask::Vb
end

function RealNVP(isize::Int, conditioner_builder::Function, even=true)
	mask = even ? (mod.(1:isize, 2) .== 0) : (mod.(1:isize, 2) .== 1)
	d = sum(mask)
	cα = conditioner_builder(d, isize - d, "relu", "identity")
	cβ = conditioner_builder(d, isize - d, "tanh", "identity")
	return RealNVP(cα, cβ, mask)
end

function (nvp::RealNVP)(xl)
	X, logJ = xl
	X_cond = X[nvp.mask,:]
	α, β = nvp.cα(X_cond), nvp.cβ(X_cond)
	Y = α .+ exp.(0.5 .* β) .* X[.~nvp.mask,:]
	_cat_with_mask(X_cond, Y, nvp.mask), logJ .+ 0.5 .* sum(β, dims=1)
end

function inv_flow(nvp::RealNVP, yl)
	Y, logJ = yl
	Y_cond = Y[nvp.mask,:]
	α, β = nvp.cα(Y_cond), nvp.cβ(Y_cond)
	X = (Y[.~nvp.mask,:] .- α) ./ exp.(0.5 .* β)
	_cat_with_mask(Y_cond, X, nvp.mask), logJ .- 0.5 .* sum(β, dims=1)
end

Flux.trainable(nvp::RealNVP) = (nvp.cα, nvp.cβ, )

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
