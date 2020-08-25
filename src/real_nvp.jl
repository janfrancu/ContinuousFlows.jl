using Flux, Flux.Zygote

struct RealNVP
	c # some conditioner
	mask
end

function RealNVP(isize, even=true)
	mask = even ? (mod.(1:isize, 2) .== 0) : (mod.(1:isize, 2) .== 1)
	d = sum(mask)
	RealNVP(Dense(d, 2*(isize - d)), mask)
end

function (nvp::RealNVP)(xl)
	X, logJ = xl
	X_cond = X[nvp.mask,:]
	α, β = nvp.c(X_cond)
	Y = α .+ β .* X[.~nvp.mask,:]
	_cat_with_mask(X_cond, Y, nvp.mask), sum(log.(abs.(β)))
end

Flux.trainable(nvp::RealNVP) = (nvp.c, )

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
