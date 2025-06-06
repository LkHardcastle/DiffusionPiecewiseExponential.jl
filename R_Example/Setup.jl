# Setup.jl for R example in README
dat = init_data(y, cens, covar, breaks);
x0, v0, s0 = init_params(p, dat);
v0 = v0./norm(v0);
t0 = 0.0;
state0 = ECMC2(x0, v0, s0, collect(.!s0), breaks, t0, length(breaks), true, findall(s0));
nits = 10_000;
nsmp = 10;
settings = Splitting(nits, nsmp, 1_000_000, 1.0, 5.0, 0.1, false, true, 0.01, 50.0);
test_times = collect(0.2:0.2:1.0)
burn_in = 1_000
