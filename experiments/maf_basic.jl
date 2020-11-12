using Revise

using Plots; gr()
using Random
using DrWatson
using Distributions
using DistributionsAD

using ContinuousFlows
using ContinuousFlows.Flux

using ToyProblems


function buildmodel(isize, p)
    Random.seed!(p.seed)

    m = Chain([
        MaskedAutoregressiveFlow(
            isize, 
            p.hsize,
            p.num_layers, 
            isize, 
            (α=eval(:($(Symbol(p.act_loc)))), β=eval(:($(Symbol(p.act_scl))))),
            (p.ordering == "natural") ? (
                (mod(i, 2) == 0) ? "reversed" : "sequential"
              ) : "random";
            seed=rand(UInt),
            use_batchnorm=p.bn,
            lastlayer=p.lastlayer
        ) 
        for i in 1:p.num_flows]...)
    Random.seed!()
    m
end

p = (
    batchsize = 100,
    epochs = 100,
    num_flows = 4,
    num_layers = 2,
    act_loc = "relu",
    act_scl = "tanh",
    bn = false,
    hsize = 5,
    ordering = "natural",
    seed = 42,
    wreg = 0.0,
    lr = 1e-3,
    lastlayer = "linear"
)

Random.seed!(p.seed)
x = Float32.(onemoon(1000))
scatter(x[1,:], x[2,:])
Random.seed!()

isize = size(x, 1)
base = MvNormal(isize, 1.0f0)
model = buildmodel(isize, p);
ps = Flux.params(model);
opt = (p.wreg > 0) ? ADAMW(p.lr, (0.9, 0.999), p.wreg) : Flux.ADAM(p.lr)
length(ps)

data = Flux.Data.DataLoader(x, batchsize=p.batchsize)


_init_logJ(data) = zeros(eltype(data), 1, size(data, 2))
function loss(x, model, base) 
    z, logJ = model((x, _init_logJ(x)))
    -sum(logpdf(base, z) .+ logJ)/size(x, 2)
end

one_batch = first(data)
z, logJ = model((one_batch, _init_logJ(one_batch)))

loss(one_batch, model, base)
gs = gradient(ps) do
    loss(one_batch, model, base)
end
Flux.update!(opt, ps, gs)


train_steps = 1
for e in 1:p.epochs 
    for batch in data
        l = 0.0f0
        gs = gradient(() -> begin l = loss(batch, model, base) end, ps)
        Flux.update!(opt, ps, gs)
        if mod(train_steps, 10) == 0
            @info("$(train_steps) - loss: $(l)")
            # @info("$(train_steps) - loss: $(l)\n\t\t\t $(model[1].bn.γ) $(model[1].bn.β) \n\t\t\t $(model[2].bn.γ) $(model[2].bn.β) \n\t\t\t $(model[3].bn.γ) $(model[3].bn.β) \n\t\t\t $(model[4].bn.γ) $(model[4].bn.β)")
            # @info("$(train_steps) - loss: $(l)\n\t\t\t $(model[1].bn.σ²) $(model[1].bn.μ) \n\t\t\t $(model[2].bn.σ²) $(model[2].bn.μ) \n\t\t\t $(model[3].bn.σ²) $(model[3].bn.μ) \n\t\t\t $(model[4].bn.σ²) $(model[4].bn.β)")
        end
        train_steps += 1
    end
end

testmode!(model, true);
savepath = datadir("MAF")
!isdir(savepath) && mkdir(savepath)
filename = joinpath(savepath, savename(p, digits=6))

# xy, _ = model((x, _init_logJ(x)))
# scatter(x[1,:], x[2,:])
# scatter!(xy[1,:], xy[2,:], size=(800,800))


base_samples = rand(base, 1000)
yx = inv_flow(model, (base_samples, _init_logJ(base_samples)))[1]

scatter(base_samples[1,:], base_samples[2,:], size=(800,800))
scatter!(x[1,:], x[2,:], size=(800,800))
scatter!(yx[1,:], yx[2,:], ylim=(-6.0, 6.0), xlim=(-6.0,6.0) , size=(800,800))
savefig(filename * ".png")

# check the inversion
x, logJ = x, _init_logJ(x)
xy, logJxy = model((x, _init_logJ(base_samples)))
yx, logJyx = inv_flow(model, (xy, _init_logJ(base_samples)))

isapprox(x, yx, atol=1e-3)
isapprox(logJxy, -logJyx, atol=1e-3)

