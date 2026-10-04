# CLAUDE.md

Julia package for **diffusion piecewise exponential models** (Hardcastle, Livingstone & Baio, [arXiv:2505.05932](https://arxiv.org/abs/2505.05932); the `.tex`/`.pdf` are in the repo root; the paper's research code is [PEM_extrap](https://github.com/LkHardcastle/PEM_extrap)). The model has a piecewise-constant log-hazard, a skew-symmetric discretised-diffusion prior on its increments and a Poisson-process prior on the knots. It is sampled with sticky forward event-chain PDMPs using a splitting scheme. Users are HTA analysts, often calling it from R via JuliaCall (`R_Example/Setup.jl`).

## Checking changes
- There are no tests. Run the README example (`nits = 1_000` takes seconds) and compare R-hat/ESS. For refactors, compare outputs from a fixed seed.
- For gradient changes, compare `∇U` (active x, then log σ) with finite differences of
  U = Σ[exp(η)W - δη] + Σ_k x_k1²/(2σ0²) + Σ_{k, j≥2 active}[x_kj²/2 - log(1 + tanh(μ(α_k,j-1)σx_kj))] + aσ - log σ,
  where α_k = cumsum([x_k1, σx_k2, …]) and η = UQ'α.
- For knot changes, check that `grid_update!` keeps every active (row, `s_loc[j-1]`, x) jump and the log-likelihood unchanged. For knot-prior changes, run without data (y ≈ 0, all censored) and compare knot counts with Poisson(ωΓ(max_time - breaks[1])).

## Architecture
- `pem_fit` runs `pem_sample` twice and returns `(out1, out2, rhat_, ess_)`. The only sampler is `Splitting` with an `ECMC2` state.
- `split_inner!` (Sampler.jl):
  - It starts with `grid_update!` if `h_rate > 0`, then takes `thin` steps of size `δ`.
  - Each step computes `∇U` once. It then bounces with probability `1 - exp(-δλ)` via `flip!(state, priors, U_grad)`, which passes the gradient on to `ortho_update!`. Then it calls `update!` (a linear move with sticky split/merge, SplitMerge.jl) and possibly `refresh!`.
- `grid_update!` (HyperUpdates.jl) redraws the inactive knots from PPP((1-ω)Γ) on (`dyn.min_time`, `max_time`), where `dyn.min_time` is the initial `breaks[1]`, and rebuilds `dyn.W`/`dyn.δ`. It must not change the likelihood: inactive columns go with their left ends, and each new candidate starts a zero column at its location.
- The output Dict holds `"Sk_*"` arrays, one entry per stored iteration, `Inf`-padded beyond each iteration's knots. `cumsum(Sk_θ, dims = 2)` is the log-hazard.
- `get_llhood(out, dat, burn_in)` gives the pointwise log-likelihood (n × iterations, in `dat`'s sorted order), and `get_looic` runs PSIS-LOO on it with ParetoSmooth.

## Notation (mirrors the paper)
- `state.x` (p × J): column 1 is the initial log-hazard α₀ ~ N(0, σ0²); columns 2:J are increments scaled by σ. Row 1 is the baseline (`covar` includes an intercept row), and the other rows are covariate effects.
- `s` marks active entries and `g` inactive candidate knots. `active = findall(s)`.
- `s_loc[j]` is the right end of column j's interval [`s_loc[j-1]`, `s_loc[j]`), and the last column runs to ∞. So the knot for column j > 1 sits at `s_loc[j-1]`, and the drift for its increment is evaluated there.
- `priors.σ.σ` is σ, and log σ moves with velocity `priors.v`. `priors.ω.ω` is the slab weight. `priors.grid.Γ` is the dominating PPP intensity, so γ = ωΓ. `CtsNB(α, β, Γ, …)` puts Γ ~ Gamma(α, rate β) on it.
- `dyn.W`/`dyn.δ` (L × J) are exposure and event counts per unique covariate pattern (`dat.UQ`).
- `cens` is the **event** indicator (1 = event observed).

## Gotchas
- Sampling mutates `priors`, so `pem_sample` works on `deepcopy(priors0)` to keep chains independent.
- `Splitting(max_ind, h_rate, r_rate, verbose, δ, thin)`: `h_rate` is only a flag. If positive, the inactive knots are redrawn every iteration.
- `σ0` is on the log-hazard scale of the data's time units. σ0 = 1 suits years; widen it for months or days.
- `GaussLangevin` calls μ(t) on the whole `s_loc` vector, so μ must broadcast; σ(t) must return a scalar.
- `init_data` needs the last break above every observation.
- The `Fixed` grid fails in `store_state!` because it has no `Γ`/`γ` fields.
- Keep `p_split = 1`. The unsticking rate carries `p_split` twice (inside `split_rate` and again in `update!`), so other values shrink the knot count by a factor `p_split`.

## Coding style
- Code uses the paper's notation, with Unicode Greek for maths quantities and even function names (`∇U`, `∇σ`, `μθ`, `Σθ`, `Γ`, `ω`).
- Extend by multiple dispatch over the small abstract hierarchies (`State`, `Prior`, `Variance`, `Weight`, `Grid`/`Cts`, `Diffusion`, `Settings`).
  - To dispatch on a component, pass it again as an argument: `grid_update!(state, dyn, dat, priors, priors.grid)`.
  - Variants with nothing to do get empty methods.
- Types are `mutable struct`s with concrete field types (`Float64`, `Int64` written out, `Matrix{Float64}`) and positional constructors; no `@kwdef`. `BasicPrior` is parametric so that `priors.σ.σ` stays type-stable.
- Layout: snake_case functions with `!` when they mutate, long-form `function … end` with an explicit `return`, 4-space indent, `dims = 2` spacing, mostly unspaced arithmetic (`state.v*t`), and `1_000_000`.
- Broadcast and array code reads like the maths; clarity beats micro-optimisation. Randomness uses `rand(Distribution(…))` with the global RNG.
- Comments are sparse: `###`/`##` step headers, `#####` banners and short maths notes. There are no docstrings; errors are short `error("...")` calls and progress uses `println`.
- Each `export` sits next to the `include` that defines it.
- Keep the R route working: plain arrays and Dicts in and out.
- Leave commented-out alternatives alone unless asked.
