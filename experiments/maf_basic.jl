using Revise

using Plots; gr()
using Distributions
using DistributionsAD

using ContinuousFlows
using ContinuousFlows.Flux

using ToyProblems


function buildmodel(isize, p)
    Chain([
        MaskedAutoregressiveFlow(
            isize, 
            p.hsize,
            p.num_layers, 
            isize, 
            (p.ordering == "natural") ? (
                (mod(i, 2) == 0) ? "reversed" : "sequential"
              ) : "random"
            ) 
        for i in 1:p.num_flows]...)
end

x = onemoon(1000)
scatter(x[1,:], x[2,:])

p = (
    batchsize = 100,
    epochs = 200,
    num_flows = 4,
    num_layers = 2,
    hsize = 10,
    ordering = "natural"
    )


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
z, logJ = model((one_batch, _init_logJ(one_batch)))

loss(one_batch, model, base)
gs = gradient(ps) do
    loss(one_batch, model, base)
end
Flux.update!(opt, ps, gs)


train_steps = 1
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
scatter(x[1,:], x[2,:])
scatter!(xy[1,:], xy[2,:], size=(800,800))


base_samples = rand(base, 1000)
yx = inv_flow(model, (base_samples, _init_logJ(base_samples)))[1]

scatter(base_samples[1,:], base_samples[2,:], size=(800,800))
scatter!(x[1,:], x[2,:], size=(800,800))
scatter!(yx[1,:], yx[2,:], ylim=(-6.0, 6.0), xlim=(-6.0,6.0) , size=(800,800))
savefig("./MAF_4_flows_2_layer_relu-relu_2000.png")

# check the inversion
x
xy = model((x, _init_logJ(x)))[1]
yx = inv_flow(model, (xy, _init_logJ(xy)))[1]

isapprox(x, yx, atol=1e-3)

