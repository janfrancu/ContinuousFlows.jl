using Revise

using Plots; gr()
using ContinuousFlows
using ContinuousFlows.Flux
using Distributions

using ToyProblems
using SumDenseProduct

build_mlp(ks::Vector{Int}, fs::Vector) = Flux.Chain(map(i -> Dense(i[2],i[3],i[1]), zip(fs,ks[1:end-1],ks[2:end]))...)

build_mlp(isize::Int, hsize::Int, osize::Int, nlayers::Int; kwargs...) =
    build_mlp(vcat(isize, fill(hsize, nlayers-1)..., osize); kwargs...)

function build_mlp(ks::Vector{Int}; ftype::String = "relu", lastlayer::String = "")
    ftype = (ftype == "linear") ? "identity" : ftype
    fs = Array{Any}(fill(eval(:($(Symbol(ftype)))), length(ks) - 1))
    if !isempty(lastlayer)
        fs[end] = (lastlayer == "linear") ? identity : eval(:($(Symbol(lastlayer))))
    end
    build_mlp(ks, fs)
end

function buildmodel(isize, p)
    Chain([
        RealNVP(
            isize, 
            (d, o, ftype, postprocess) -> build_mlp(d, p.hsize, o, p.num_layers, ftype=ftype, lastlayer=postprocess),
            mod(i,2) == 0) 
        for i in 1:p.num_flows]...)
end

x = onemoon(1000)
scatter(x[1,:], x[2,:])

# x = vcat(onemoon(1000), onemoon(1000))

p = (
    batchsize = 100,
    epochs = 100,
    num_flows = 2,
    num_layers = 2,
    hsize = 5,)

isize = size(x, 1)

model = buildmodel(isize, p);
base = MvNormal(isize, 1.0f0)

ps = Flux.params(model);
length(ps)

opt = ADAM();
data = Flux.Data.DataLoader(x, batchsize=p.batchsize)


_init_logJ(data) = zeros(eltype(data), 1, size(data, 2))
function loss(x, model, base) 
    z, logJ = model((x, _init_logJ(x)))
    -sum(logpdf(base, z) .+ logJ)/size(x, 2)
end

one_batch = first(data)
# z, logJ = model((one_batch, _init_logJ(one_batch)))

# nvp = model[1]
# X, logJ = one_batch,  _init_logJ(one_batch)
# X_cond = X[nvp.mask,:]
# α, β = nvp.cα(X_cond), nvp.cβ(X_cond)
# Y = α .+ β .* X[.~nvp.mask,:]
# logJ .+ sum(log.(abs.(β)), dims=1)

loss(one_batch, model, base)
gs = gradient(ps) do
    loss(one_batch, model, base)
end
Flux.update!(opt, ps, gs)


train_steps = 0
Flux.@epochs p.epochs for batch in data
    global train_steps
    l = 0.0f0
    gs = gradient(() -> begin l = loss(batch, model, base) end, ps)
    Flux.update!(opt, ps, gs)
    if mod(train_steps, 10) == 0
        @info("$(train_steps) - loss: $(l)")
    end
    train_steps += 1
end


xy, _ = model((x, _init_logJ(x)))
scatter!(xy[1,:], xy[2,:], size=(800,800))


base_samples = rand(base, 1000)
yx = inv_flow(model, (base_samples, _init_logJ(base_samples)))[1]

scatter(base_samples[1,:], base_samples[2,:], size=(800,800))
scatter!(x[1,:], x[2,:], size=(800,800))
scatter!(yx[1,:], yx[2,:], ylim=(-6.0, 6.0), xlim=(-6.0,6.0) , size=(800,800))
savefig("./RealNVP_4_flows_2_layer_relu_tanh.png")

# heatmap(P_base, title="p_base", aspect_ratio=:equal)


# check the inversion
x
xy = model((x, _init_logJ(base_samples)))[1]
yx = inv_flow(model, (xy, _init_logJ(base_samples)))[1]

isapprox(x, yx, atol=1e-3)
