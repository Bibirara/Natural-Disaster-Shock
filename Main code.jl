using LinearAlgebra, Statistics, Distributions
using LaTeXStrings, Parameters, QuantEcon, DataFrames, Random
##############################################################################################################
#### Parameters
##############################################################################################################
β = .953                # time discount rate
γ = 2          # risk aversion (fixed)
r = 0.017        # US 1 year T-bill interest rate from 2008 to 2022 (fixed)
ρ = .945         # persistence in output
η = 0.025       # st dev of output shock
θ = 0.282        # prob of regaining access (fixed)
phi = 0.969      # output punishment
ny = 21        # number of points in y grid (fixed)
nB = 251
##############################################################################################################
#### Grids Setup
##############################################################################################################
# Bond grids
Bgrid = collect(range(-.4, .4, length = nB))
# Tauchen method starts for output grids
ymax=3*(η^2/(1-ρ^2))^(1/2)
ymin=-3*(η^2/(1-ρ^2))^(1/2)
ygap=(ymax-ymin)/(ny-1)
ygrid=zeros(ny)
for i in 2:ny       # or you can use: ygrid = range(ymin, stop=ymax, length=ny)
    ygrid[1]=ymin
    ygrid[i]=ymin+ygap*(i-1)
end
ygrid=exp.(ygrid)
ydefgrid = min.(phi * mean(ygrid), ygrid)  # Done with setting up the grids (state space) of output 
Π=zeros(ny,ny)
for i in 1:ny       
    Π[i,1]=cdf.(Normal(0,1), (ygrid[1]+ygap/2-ρ*ygrid[i])/η)
end
for i in 1:ny       
    Π[i,ny]=1-cdf.(Normal(0,1), (ygrid[ny]-ygap/2-ρ*ygrid[i])/η)
end
for i in 1:ny
    for j in 2:ny-1       
        Π[i,j]=abs(cdf.(Normal(0,1), (ygrid[j]-ygap/2-ρ*ygrid[i])/η)-cdf.(Normal(0,1), (ygrid[j]+ygap/2-ρ*ygrid[i])/η))
    end
end # Done with setting up the transition matrix (Markov matrix)
##############################################################################################################
#### Initial value functions, default decisions, bond price setup
##############################################################################################################
vf = zeros(nB, ny)                  # Value Function after Considering Default or Not
vd = zeros(1, ny)                   # Value Function of Default
vc = zeros(nB, ny)                  # Value Function of Not Default
policy = zeros(nB, ny)
q = ones(nB, ny) .* (1 / (1 + r))
defprob = zeros(nB, ny)
##############################################################################################################
#### Iteration
##############################################################################################################
Πt = Π'
V_upd = similar(vf)
copyto!(V_upd, vf)
EV = vf * Πt
EVd = vd * Πt
EVc = vc * Πt

zero_ind = searchsortedfirst(Bgrid, 0.)
for iy in 1:ny
    y = ygrid[iy]
    ydef = ydefgrid[iy]

    # value of being in default with income y
    defval = (ydef^(1 - γ)) / (1 - γ) + β * (θ * EV[zero_ind, iy] + (1-θ) * EVd[1, iy])
    vd[1, iy] = defval

    for ib in 1:nB
        B = Bgrid[ib]

        current_max = -1e14
        pol_ind = 0
        for ib_next=1:nB
            c = max(y - q[ib_next, iy]*Bgrid[ib_next] + B, 1e-14)
            m = (c^(1 - γ)) / (1 - γ) + β * EV[ib_next, iy]

            if m > current_max
                current_max = m
                pol_ind = ib_next
            end

        end

        # update value and policy functions
        vc[ib, iy] = current_max
        policy[ib, iy] = pol_ind
        vf[ib, iy] = defval > current_max ? defval : current_max
    end
end

vd_compat = repeat(vd, nB)
default_states = vd_compat .> vc
copyto!(defprob, default_states * Π')
copyto!(q, (1 .- defprob) / (1 + r))

dist = maximum(abs(x - y) for (x, y) in zip(V_upd, vf))