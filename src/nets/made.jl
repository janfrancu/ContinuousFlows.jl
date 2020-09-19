#=
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Re-Re-implementation by Tejan Karmali using Flux.jl ;)
Simplification for use in MaskedAutoregressiveFlows by Jan Francu.
=#

using Flux, Random
using Flux: glorot_uniform
# ------------------------------------------------------------------------------

add_dims_r(a) = reshape(a, size(a)..., 1)
add_dims_l(a) = reshape(a, 1, size(a)...)

function funnel(c::Chain)
    isize = size(c[1].W, 2)
    bsize = [length(l.b) for l in c]
    osize = bsize[end]
    isize, bsize[1:end-1], osize
end

# ------------------------------------------------------------------------------

struct MaskedDense{F,S,T,M}
  # same as Linear except has a configurable mask on the weights
  W::S
  b::T
  mask::M
  σ::F
end

function MaskedDense(in::Integer, out::Integer, σ = identity)
  return MaskedDense(glorot_uniform(out, in), zeros(out), ones(out, in), σ)
end

Flux.@functor MaskedDense
Flux.trainable(m::MaskedDense) = (m.W, m.b)

function (a::MaskedDense)(x)
  a.σ.(a.mask .* a.W * x .+ a.b)
end

# ------------------------------------------------------------------------------

mutable struct MADE
  net::Chain
  m::Dict{Int, Vector{Int}}

  function MADE(
        isize::Int, 
        hsizes, 
        osize::Int, 
        ordering::String="sequential";
        ftype=relu,
        ptype=identity,
        rs=time_ns())
    
    # define a simple MLP neural net
    hsizes = push!([isize], hsizes...)
    layers = [MaskedDense(hsizes[i], hsizes[i + 1], ftype) for i = 1:length(hsizes) - 1]
    net = Chain(layers..., MaskedDense(hsizes[end], osize, ptype))

    made = new(net, Dict{Int, Vector{Int}}())
    update_masks!(made, ordering, rs)
    made
  end
end

Flux.@functor MADE
Flux.trainable(m::MADE) = (m.net, )

function update_masks!(made::MADE, ordering, seed)
    isize, hsizes, osize = funnel(made.net)
    L = length(hsizes)

    # make masks reproducible
    rng = MersenneTwister(seed)

    # sample the order of the inputs and the connectivity of all neurons
    made.m[0] = (ordering == "sequential") ? collect(1:isize) : (
        ordering == "reversed" ? collect(isize:-1:1) : randperm(rng, isize))
    for l = 1:L
        made.m[l] = rand(rng, minimum(made.m[l - 1]):isize - 1, hsizes[l])
    end

    # construct the mask matrices
    masks = [add_dims_r(made.m[l - 1]) .<= add_dims_l(made.m[l]) for l = 1:L]
    push!(masks, add_dims_r(made.m[L]) .< add_dims_l(made.m[0]))

    # this assumes that there are no other layers than MaskedDense
    for (l, m) in zip(made.net, masks)
        l.mask .= permutedims(m, [2, 1])
    end
end

function (made::MADE)(x)
  made.net(x)
end

# ------------------------------------------------------------------------------
