#=
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
Re-Re-implementation by Tejan Karmali using Flux.jl ;)
=#

using Flux, Random
using Flux: glorot_uniform
# ------------------------------------------------------------------------------

add_dims_r(a) = reshape(a, size(a)..., 1)
add_dims_l(a) = reshape(a, 1, size(a)...)

# ------------------------------------------------------------------------------

struct MaskedDense{F,S,T,M}
  # same as Linear except has a configurable mask on the weights
  W::S
  b::T
  mask::M
  σ::F
end

function MaskedDense(in::Integer, out::Integer; σ = identity)
  return MaskedDense(glorot_uniform(out, in), zeros(out), ones(out, in), σ)
end

Flux.@functor MaskedDense
Flux.trainable(m::MaskedDense) = (m.W, m.b)

function (a::MaskedDense)(x)
  a.σ.(a.mask .* a.W * x .+ a.b)
end

# ------------------------------------------------------------------------------

mutable struct MADE
  nin::Int
  nout::Int
  hidden_sizes::Vector{Int}
  net::Chain

  # seeds for orders/connectivities of the model ensemble
  natural_ordering::Bool
  num_masks::Int
  seed::UInt  # for cycling through num_masks orderings

  m::Dict{Int, Vector{Int}}

  function MADE(in::Integer, hs, out::Integer, nat_ord::Bool, num_masks = 1, rs = time_ns())
    # define a simple MLP neural net
    hs = push!([in], hs...)
    layers = [MaskedDense(hs[i], hs[i + 1]; σ = tanh) for i = 1:length(hs) - 1]

    net = Chain(layers..., MaskedDense(hs[end], out))

    made = new(in, out, hs[2:end], net, nat_ord, num_masks, rs, Dict{Int, Vector{Int}}())
    update_masks(made)
    return made
  end
end

Flux.@functor MADE
Flux.trainable(m::MADE) = (m.net, )

function update_masks(made::MADE)
  if made.m != Dict() && made.num_masks == 1
    return # only a single seed, skip for efficiency
  end

  L = length(made.hidden_sizes)

  # fetch the next seed and construct a random stream
  rng = MersenneTwister(made.seed)
  made.seed = (made.seed + 1) % made.num_masks

  # sample the order of the inputs and the connectivity of all neurons
  made.m[0] = made.natural_ordering ? collect(1:made.nin) : randperm(rng, made.nin)
  for l = 1:L
    made.m[l] = rand(rng, minimum(made.m[l - 1]):made.nin - 2, made.hidden_sizes[l])
  end

  # construct the mask matrices
  masks = [add_dims_r(made.m[l - 1]) .<= add_dims_l(made.m[l]) for l = 1:L]
  push!(masks, add_dims_r(made.m[L]) .< add_dims_l(made.m[0]))

  # handle the case where nout = nin * k, for integer k > 1
  if made.nout > made.nin
    k = div(made.nout, made.nin)
    # replicate the mask across the other outputs
    masks[end] = hcat(tuple((masks[end] for i=1:k)...))
  end

  # set the masks in all MaskedLinear layers
  for (l, m) in zip(made.net, masks)
    isa(l, MaskedDense) ? l.mask .= permutedims(m, [2, 1]) : continue
  end
end

function (made::MADE)(x)
  made.net(x)
end

# ------------------------------------------------------------------------------
