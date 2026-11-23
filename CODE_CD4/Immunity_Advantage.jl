using DataFrames
using Statistics
using StatsBase
using Random
using PlotlyJS

Random.seed!(2026)

############################################################
# RAW PoS MEASURED DAYS:
# ONE PLOT, COLOURED BY PAIR
#
# CUSTOM "BOX + WHISKER":
#   middle marker/line = median
#   box bottom/top     = Q1 / Q3
#   whisker ends       = 2.5% / 97.5%
#
# SPAGHETTI:
#   15 sampled replicate trajectories per pair
############################################################

measured_days = [90.0, 150.0 ,300.0, 450.0, 600.0,750.0, 900.0,]
#measured_days = [90.0, 420.0]
NRUNS_MEASURED = 100
N_SPAGHETTI_PER_PAIR = 10

pair_colors = Dict(
    "50 > 60" => "#AA4499",
    "50 > 70" => "#117733",
    "60 > 70" => "#332288"
)

############################################################
# RECOMPUTE RAW PoS AT EXACT MEASURED DAYS
############################################################

measured_raw_rows = DataFrame(
    Run = Int[],
    Pair = String[],
    Young = Int[],
    Old = Int[],
    Time = Float64[],
    P = Float64[],
    N_young = Int[],
    N_old = Int[]
)

for run_id in 1:NRUNS_MEASURED
    println("Measured-day raw PoS replicate $run_id / $NRUNS_MEASURED")

    final_vp_rep = Dict{Int,DataFrame}()

    for age in ages
        final_vp_rep[age] = build_final_pool_for_age(
            cases[age],
            fits[age],
            df,
            age;
            rule = final_pool_rule
        )
    end

    for (young, old) in pairs_age
        pair_label = "$young > $old"

        for t in measured_days
            xraw = raw_values_at_time(final_vp_rep[young], cases[young], t)
            yraw = raw_values_at_time(final_vp_rep[old], cases[old], t)

            push!(measured_raw_rows, (
                Run = run_id,
                Pair = pair_label,
                Young = young,
                Old = old,
                Time = t,
                P = fast_pos(xraw, yraw),
                N_young = length(xraw),
                N_old = length(yraw)
            ))
        end
    end
end

############################################################
# SUMMARY: MEDIAN, IQR, 95% CI
############################################################

measured_raw_summary = combine(
    groupby(measured_raw_rows, [:Pair, :Young, :Old, :Time]),
    :P => median => :P_median,
    :P => (x -> quantile(x, 0.25)) => :Q1,
    :P => (x -> quantile(x, 0.75)) => :Q3,
    :P => (x -> quantile(x, 0.025)) => :CI_low,
    :P => (x -> quantile(x, 0.975)) => :CI_high,
    nrow => :NRuns
)

sort!(measured_raw_summary, [:Young, :Old, :Time])

############################################################
# CUSTOM PLOT
############################################################

function plot_custom_ci_box_spaghetti(
    measured_raw_rows::DataFrame,
    measured_raw_summary::DataFrame;
    n_spaghetti_per_pair = 15,
    title_str = "Anti-gE Antibody titre immune advantage probability",
    ypad = 0.04,
    box_half_width = 10.0,
    cap_half_width = 16.0
)

    traces = PlotlyJS.GenericTrace[]

    pair_order = unique(measured_raw_summary.Pair)

    ymin = max(0.0, minimum(measured_raw_summary.CI_low) - ypad)
    ymax = min(1.0, maximum(measured_raw_summary.CI_high) + ypad)

    ########################################################
    # SPAGHETTI: sampled replicate trajectories
    ########################################################

    for pr in pair_order
        col = pair_colors[pr]

        sub_reps = copy(measured_raw_rows[measured_raw_rows.Pair .== pr, :])
        run_ids = unique(sub_reps.Run)

        if length(run_ids) > n_spaghetti_per_pair
            run_ids = sample(run_ids, n_spaghetti_per_pair; replace=false)
        end

        for rid in run_ids
            sr = copy(sub_reps[sub_reps.Run .== rid, :])
            sort!(sr, :Time)

            push!(traces,
                scatter(
                    x = sr.Time,
                    y = sr.P,
                    mode = "lines",
                    line = attr(color = col, width = 1.5),
                    opacity = 0.45,
                    showlegend = false,
                    legendgroup = pr,
                    name = "$pr spaghetti"
                )
            )
        end
    end

    ########################################################
    # CUSTOM BOXES + 95% CI WHISKERS
    ########################################################

    for pr in pair_order
        col = pair_colors[pr]

        sub = copy(measured_raw_summary[measured_raw_summary.Pair .== pr, :])
        sort!(sub, :Time)

        for r in eachrow(sub)
            x = r.Time

            # whisker vertical line: 2.5% to 97.5%
            push!(traces,
                scatter(
                    x = [x, x],
                    y = [r.CI_low, r.CI_high],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            # bottom cap
            push!(traces,
                scatter(
                    x = [x - cap_half_width, x + cap_half_width],
                    y = [r.CI_low, r.CI_low],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            # top cap
            push!(traces,
                scatter(
                    x = [x - cap_half_width, x + cap_half_width],
                    y = [r.CI_high, r.CI_high],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            # IQR box polygon: Q1 to Q3
            push!(traces,
                scatter(
                    x = [
                        x - box_half_width,
                        x + box_half_width,
                        x + box_half_width,
                        x - box_half_width,
                        x - box_half_width
                    ],
                    y = [
                        r.Q1,
                        r.Q1,
                        r.Q3,
                        r.Q3,
                        r.Q1
                    ],
                    mode = "lines",
                    fill = "toself",
                    fillcolor = "rgba(255,255,255,0)",
                    line = attr(color = col, width = 2.5),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            # median horizontal line inside box
            push!(traces,
                scatter(
                    x = [x - box_half_width, x + box_half_width],
                    y = [r.P_median, r.P_median],
                    mode = "lines",
                    line = attr(color = col, width = 4),
                    showlegend = false,
                    legendgroup = pr
                )
            )
        end

        # median trajectory
        push!(traces,
            scatter(
                x = sub.Time,
                y = sub.P_median,
                mode = "lines+markers",
                line = attr(color = col, width = 3),
                marker = attr(color = col, size = 8),
                name = "$pr median",
                legendgroup = pr
            )
        )
    end

    ########################################################
    # NULL LINE
    ########################################################

    push!(traces,
        scatter(
            x = measured_days,
            y = fill(0.5, length(measured_days)),
            mode = "lines",
            line = attr(color = "black", dash = "dash", width = 2),
            showlegend = false,
            name = "null"
        )
    )

    layout = Layout(
        title = attr(
    	text = title_str,
   	 x = 0.5,
    	xanchor = "center"
	),
        xaxis = attr(
            title = "Measured day",
            tickmode = "array",
            tickvals = measured_days,
            ticktext = string.(Int.(measured_days)),
            showline = true,
            linecolor = "black"
        ),
        yaxis = attr(
            title = "Probability",
            range = [ymin, ymax],
            showline = true,
            linecolor = "black"
        ),
        width = 1100,
        height = 650,
        plot_bgcolor = "white",
        legend = true
    )

    return plot(traces, layout)
end

p_measured_raw = plot_custom_ci_box_spaghetti(
    measured_raw_rows,
    measured_raw_summary;
    n_spaghetti_per_pair = N_SPAGHETTI_PER_PAIR,
    title_str = "Anti-gE antibody titre immune advantage probability"
)

#display(p_measured_raw)

############################################################
# Shingrix CD4 persistence model — SINGLE EXPONENTIAL
#
# FIXED VERSION:
#   a is drawn only within observed TIME=0 min/max
#   c is drawn only within observed TIME=90 min/max
#
# Therefore:
#   no simulated day 0 value can lie outside observed day 0 min/max
#   no simulated day 90 value can lie outside observed day 90 min/max
#
# Single-exponential model:
#   y(t) = a + (c - a) * exp(-u * (t - t0))
############################################################

using CSV, DataFrames
using Statistics, StatsBase
using Distributions
using Random
using Dates
using Printf

Random.seed!(9876)

# ==========================================================
# USER INPUTS
# ==========================================================
csv_path = raw"C:\Users\Doc\Desktop\gsk data\GSKCD4 - GSKCD4.csv"
#csv_path = raw"C:\Users\Doc\Desktop\gsk data\GSK_Shingrix_Statistics - Sheet1AB - GSK_Shingrix_Statistics - Sheet1AB.csv"
#csv_path = raw"C:\Users\Doc\Downloads\CD4_stats_from_C1 - CD4_stats_from_C1.csv"

t0 = 90.0
tukey_k = 1.5

# MCMC
N_iter = 150_000
burn_in_iter = 5_000
N_sim = 300

# Proposal widths
SD_mu_u = 0.06
SD_logsigma_u = 0.06

# Bounds for individual rate
u_lo = log(2) / 5000.0
u_hi = log(2) / 7.0

# Bounds for population SD on log scale
sigma_u_lo = 0.05
sigma_u_hi = 1.25

# ABC settings
abc_epsilon = Inf
abc_kernel_scale = 0.10
support_weight = 2.0
max_saved_trajs_per_accept = 40

# Robust initialization
max_init_tries = 500

# ==========================================================
# MODEL
# ==========================================================
function simulate_persistence(times::Vector{Float64}, a, c, u, t0)
    Δ = times .- t0
    return a .+ (c - a) .* exp.(-u .* Δ)
end

# ==========================================================
# LOGNORMAL RECONSTRUCTION FROM QUARTILES
# ==========================================================
function lognormal_from_q1_median_q3(q1::Float64, med::Float64, q3::Float64)
    q1 > 0 || error("Q1 must be > 0")
    med > 0 || error("Median must be > 0")
    q3 > 0 || error("Q3 must be > 0")
    q3 >= q1 || error("Q3 must be >= Q1")

    mu = log(med)
    sigma = q3 == q1 ? 1e-6 : max((log(q3) - log(q1)) / 1.3489795, 1e-6)
    return LogNormal(mu, sigma), mu, sigma
end

function pooled_lognormal_from_rows(rows::DataFrame)
    Ns = Float64.(rows.N)
    wts = Ns ./ sum(Ns)

    pooled_log_q1 = sum(wts .* log.(Float64.(rows.Q1)))
    pooled_log_med = sum(wts .* log.(Float64.(rows.Median)))
    pooled_log_q3 = sum(wts .* log.(Float64.(rows.Q3)))

    dist, mu, sigma = lognormal_from_q1_median_q3(
        exp(pooled_log_q1),
        exp(pooled_log_med),
        exp(pooled_log_q3)
    )

    return (
        dist = dist,
        mu = mu,
        sigma = sigma,
        q1 = exp(pooled_log_q1),
        median = exp(pooled_log_med),
        q3 = exp(pooled_log_q3)
    )
end

# ==========================================================
# SUPPORT HELPERS
# ==========================================================
function tukey_bounds_logscale(q1::Float64, q3::Float64; k=1.5)
    q1 > 0 || error("Q1 must be > 0")
    q3 > 0 || error("Q3 must be > 0")
    q3 >= q1 || error("Q3 must be >= Q1")

    lq1 = log(q1)
    lq3 = log(q3)
    liqr = lq3 - lq1

    lo = exp(lq1 - k * liqr)
    hi = exp(lq3 + k * liqr)
    return lo, hi
end

# ==========================================================
# HELPERS
# ==========================================================
function summarize_sample(x::AbstractVector{<:Real})
    xx = collect(x)
    return (
        mean = mean(xx),
        sd = length(xx) > 1 ? std(xx, corrected=true) : 0.0,
        q1 = quantile(xx, 0.25),
        median = quantile(xx, 0.50),
        q3 = quantile(xx, 0.75)
    )
end

function summarize_matrix_by_row(Y::Matrix{Float64})
    nt = size(Y, 1)
    means = zeros(nt)
    sds = zeros(nt)
    q1s = zeros(nt)
    meds = zeros(nt)
    q3s = zeros(nt)

    for i in 1:nt
        s = summarize_sample(view(Y, i, :))
        means[i] = s.mean
        sds[i] = s.sd
        q1s[i] = s.q1
        meds[i] = s.median
        q3s[i] = s.q3
    end

    return (mean = means, sd = sds, q1 = q1s, median = meds, q3 = q3s)
end

# ==========================================================
# DATA PREP
# ==========================================================
function prep_ab(df::DataFrame)
    return df[df.Type .== "CD4", :] #change to AB when doing AB
end

function make_age_case(df_ab::DataFrame, age::Int; t0=90.0, tukey_k=1.5)
    d_age = df_ab[df_ab.AGE .== age, :]
    nrow(d_age) > 0 || error("No CD4 rows found for AGE=$age")

    d_base = d_age[d_age.TIME .== 0, :]
    nrow(d_base) > 0 || error("No CD4 row at TIME=0 for AGE=$age")
    base_fit = pooled_lognormal_from_rows(d_base)

    d_peak = d_age[d_age.TIME .== t0, :]
    nrow(d_peak) > 0 || error("No CD4 row at TIME=t0=$t0 for AGE=$age")
    peak_fit = pooled_lognormal_from_rows(d_peak)

    # strict observed bounds for TIME=0 and TIME=90
    base_min = minimum(Float64.(d_base.Min))
    base_max = maximum(Float64.(d_base.Max))
    peak_min = minimum(Float64.(d_peak.Min))
    peak_max = maximum(Float64.(d_peak.Max))

    d_persist = d_age[d_age.TIME .>= t0, :]
    times = sort(unique(Float64.(d_persist.TIME)))

    target_mean = zeros(length(times))
    target_sd = zeros(length(times))
    target_q1 = zeros(length(times))
    target_median = zeros(length(times))
    target_q3 = zeros(length(times))
    target_log_q1 = zeros(length(times))
    target_log_median = zeros(length(times))
    target_log_q3 = zeros(length(times))
    support_lo = zeros(length(times))
    support_hi = zeros(length(times))
    target_n = zeros(Int, length(times))

    for (j, tj) in enumerate(times)
        rows = d_persist[d_persist.TIME .== tj, :]
        Ns = Float64.(rows.N)
        wts = Ns ./ sum(Ns)

        target_n[j] = round(Int, sum(Ns))
        target_mean[j] = sum(wts .* Float64.(rows.Mean))
        target_sd[j] = sum(wts .* Float64.(rows.SD))
        target_q1[j] = sum(wts .* Float64.(rows.Q1))
        target_median[j] = sum(wts .* Float64.(rows.Median))
        target_q3[j] = sum(wts .* Float64.(rows.Q3))

        target_log_q1[j] = sum(wts .* log.(Float64.(rows.Q1)))
        target_log_median[j] = sum(wts .* log.(Float64.(rows.Median)))
        target_log_q3[j] = sum(wts .* log.(Float64.(rows.Q3)))

        obs_min = minimum(Float64.(rows.Min))
        obs_max = maximum(Float64.(rows.Max))
        tuk_lo, tuk_hi = tukey_bounds_logscale(target_q1[j], target_q3[j]; k=tukey_k)

        support_lo[j] = min(obs_min, tuk_lo)
        support_hi[j] = max(obs_max, tuk_hi)
    end

    return (
        age = age,
        times = times,
        t0 = t0,

        baseline_dist = base_fit.dist,
        c_dist = peak_fit.dist,

        baseline_mu = base_fit.mu,
        baseline_sigma = base_fit.sigma,
        peak_mu = peak_fit.mu,
        peak_sigma = peak_fit.sigma,

        base_min = base_min,
        base_max = base_max,
        peak_min = peak_min,
        peak_max = peak_max,

        target_n = target_n,
        target_mean = target_mean,
        target_sd = target_sd,
        target_q1 = target_q1,
        target_median = target_median,
        target_q3 = target_q3,
        target_log_q1 = target_log_q1,
        target_log_median = target_log_median,
        target_log_q3 = target_log_q3,
        support_lo = support_lo,
        support_hi = support_hi
    )
end

# ==========================================================
# INITIAL FIT TO OBSERVED MEDIANS
# ==========================================================
function fit_initial_mu_u_from_medians(case; u_lo, u_hi)
    times = case.times
    meds = case.target_median

    idx_t0 = findfirst(==(case.t0), times)
    idx_t0 === nothing && return log(0.005)

    a0 = exp(case.baseline_mu)
    c0 = meds[idx_t0]

    c0 > a0 || return log(0.005)

    u_estimates = Float64[]

    for j in eachindex(times)
        t = times[j]
        y = meds[j]

        t > case.t0 || continue
        y > a0 || continue
        y < c0 || continue

        Δ = t - case.t0
        uhat = -log((y - a0) / (c0 - a0)) / Δ

        if isfinite(uhat) && u_lo <= uhat <= u_hi
            push!(u_estimates, uhat)
        end
    end

    isempty(u_estimates) && return log(0.005)

    return log(median(u_estimates))
end

# ==========================================================
# DRAW SUBJECT-LEVEL RATES
# ==========================================================
function draw_subject_rates(N::Int, mu_u, sigma_u)
    dist = LogNormal(mu_u, sigma_u)
    return rand(dist, N)
end

# ==========================================================
# DRAW a AND c WITH STRICT TIME-0 / TIME-90 BOUNDS
# ==========================================================
function draw_a_c_strict(case, N::Int)
    # Truncate the reconstructed lognormal distributions so:
    #   a ∈ [observed TIME=0 min, observed TIME=0 max]
    #   c ∈ [observed TIME=90 min, observed TIME=90 max]
    a_dist = truncated(case.baseline_dist, case.base_min, case.base_max)
    c_dist = truncated(case.c_dist, case.peak_min, case.peak_max)

    a = rand(a_dist, N)
    c = rand(c_dist, N)

    return a, c
end

# ==========================================================
# SIMULATE PSEUDO-POPULATION
# ==========================================================
function simulate_population(case, mu_u, sigma_u; N_sim=300)
    times = case.times
    nt = length(times)

    a, c = draw_a_c_strict(case, N_sim)

    # still enforce c >= a
    bad_ac = findall(c .< a)
    tries = 0
    while !isempty(bad_ac) && tries < 500
        a_new, c_new = draw_a_c_strict(case, length(bad_ac))
        a[bad_ac] .= a_new
        c[bad_ac] .= c_new
        bad_ac = findall(c .< a)
        tries += 1
    end
    isempty(bad_ac) || return nothing

    u = draw_subject_rates(N_sim, mu_u, sigma_u)

    Y = Matrix{Float64}(undef, nt, N_sim)
    for i in 1:N_sim
        Y[:, i] = simulate_persistence(times, a[i], c[i], u[i], case.t0)
    end

    return (Y=Y, a=a, c=c, u=u)
end

# ==========================================================
# SOFT SUPPORT PENALTY
# ==========================================================
function support_penalty(case, Y::Matrix{Float64})
    pen = 0.0

    for j in eachindex(case.times)
        y = view(Y, j, :)
        lo = case.support_lo[j]
        hi = case.support_hi[j]

        below = y[y .< lo]
        above = y[y .> hi]

        if !isempty(below)
            pen += mean(((lo .- below) ./ max(lo, 1e-12)).^2)
        end

        if !isempty(above)
            pen += mean(((above .- hi) ./ max(hi, 1e-12)).^2)
        end
    end

    return pen
end

# ==========================================================
# ABC DISTANCE
# ==========================================================
function abc_distance(case, Y::Matrix{Float64}; support_weight=1.0)
    summ = summarize_matrix_by_row(Y)

    sim_log_q1 = log.(max.(summ.q1, 1e-12))
    sim_log_median = log.(max.(summ.median, 1e-12))
    sim_log_q3 = log.(max.(summ.q3, 1e-12))

    d_q1 = mean((sim_log_q1 .- case.target_log_q1).^2)
    d_med = mean((sim_log_median .- case.target_log_median).^2)
    d_q3 = mean((sim_log_q3 .- case.target_log_q3).^2)
    d_sup = support_penalty(case, Y)

    d_total = d_q1 + d_med + d_q3 + support_weight * d_sup

    return d_total, (
        mean = summ.mean,
        sd = summ.sd,
        q1 = summ.q1,
        median = summ.median,
        q3 = summ.q3
    )
end

function abc_log_kernel(distance::Float64, scale::Float64)
    return -distance / (2 * scale^2)
end

# ==========================================================
# PRIOR
# ==========================================================
function log_prior_hyper(mu_u, sigma_u;
    u_lo, u_hi, sigma_u_lo, sigma_u_hi)

    if !(log(u_lo) <= mu_u <= log(u_hi))
        return -Inf
    end
    if !(sigma_u_lo <= sigma_u <= sigma_u_hi)
        return -Inf
    end

    return 0.0
end

# ==========================================================
# ABC-MCMC
# ==========================================================
function run_abc_mcmc_for_age(case;
    N_iter = 40_000,
    burn_in_iter = 5_000,
    N_sim = 300,
    SD_mu_u = 0.05,
    SD_logsigma_u = 0.05,
    u_lo = log(2)/5000.0,
    u_hi = log(2)/7.0,
    sigma_u_lo = 0.05,
    sigma_u_hi = 1.25,
    abc_epsilon = Inf,
    abc_kernel_scale = 0.10,
    support_weight = 2.0,
    max_saved_trajs_per_accept = 40,
    max_init_tries = 500
)
    cur_mu_u = fit_initial_mu_u_from_medians(case; u_lo=u_lo, u_hi=u_hi)
    cur_sigma_u = 0.30

    cur_lp = log_prior_hyper(
        cur_mu_u, cur_sigma_u;
        u_lo=u_lo, u_hi=u_hi,
        sigma_u_lo=sigma_u_lo, sigma_u_hi=sigma_u_hi
    )
    isfinite(cur_lp) || error("Initial hyperparameters outside support")

    cur_pop = nothing
    for _ in 1:max_init_tries
        trial_pop = simulate_population(case, cur_mu_u, cur_sigma_u; N_sim=N_sim)
        if trial_pop !== nothing
            cur_pop = trial_pop
            break
        end
    end
    cur_pop === nothing && error("Could not generate an initial pseudo-population")

    cur_dist, cur_summ = abc_distance(case, cur_pop.Y; support_weight=support_weight)

    mu_u_chain = zeros(N_iter)
    sigma_u_chain = zeros(N_iter)
    dist_chain = fill(NaN, N_iter)
    accepted_flag = falses(N_iter)

    mu_u_chain[1] = cur_mu_u
    sigma_u_chain[1] = cur_sigma_u
    dist_chain[1] = cur_dist

    accepted_samples = NamedTuple[]
    accepted_pop_summaries = NamedTuple[]
    accepted_subject_params = NamedTuple[]
    accepted_trajs = Vector{Vector{Float64}}()

    start_time = now()

    for i in 2:N_iter
        prop_mu_u = rand(Normal(cur_mu_u, SD_mu_u))
        prop_logsigma_u = rand(Normal(log(cur_sigma_u), SD_logsigma_u))
        prop_sigma_u = exp(prop_logsigma_u)

        prop_lp = log_prior_hyper(
            prop_mu_u, prop_sigma_u;
            u_lo=u_lo, u_hi=u_hi,
            sigma_u_lo=sigma_u_lo, sigma_u_hi=sigma_u_hi
        )

        if !isfinite(prop_lp)
            mu_u_chain[i] = cur_mu_u
            sigma_u_chain[i] = cur_sigma_u
            dist_chain[i] = cur_dist
            continue
        end

        prop_pop = simulate_population(case, prop_mu_u, prop_sigma_u; N_sim=N_sim)
        if prop_pop === nothing
            mu_u_chain[i] = cur_mu_u
            sigma_u_chain[i] = cur_sigma_u
            dist_chain[i] = cur_dist
            continue
        end

        prop_dist, prop_summ = abc_distance(case, prop_pop.Y; support_weight=support_weight)

        if prop_dist > abc_epsilon
            mu_u_chain[i] = cur_mu_u
            sigma_u_chain[i] = cur_sigma_u
            dist_chain[i] = cur_dist
            continue
        end

        log_alpha =
            (prop_lp - cur_lp) +
            (abc_log_kernel(prop_dist, abc_kernel_scale) -
             abc_log_kernel(cur_dist, abc_kernel_scale))

        if log(rand()) < min(0.0, log_alpha)
            cur_mu_u = prop_mu_u
            cur_sigma_u = prop_sigma_u
            cur_lp = prop_lp
            cur_pop = prop_pop
            cur_dist = prop_dist
            cur_summ = prop_summ
            accepted_flag[i] = true

            if i > burn_in_iter
                push!(accepted_samples, (
                    iter = i,
                    mu_u = cur_mu_u,
                    sigma_u = cur_sigma_u,
                    distance = cur_dist
                ))

                push!(accepted_pop_summaries, (
                    iter = i,
                    distance = cur_dist,
                    mean = cur_summ.mean,
                    sd = cur_summ.sd,
                    q1 = cur_summ.q1,
                    median = cur_summ.median,
                    q3 = cur_summ.q3
                ))

                keep = min(length(cur_pop.u), max_saved_trajs_per_accept)
                idx = sample(1:length(cur_pop.u), keep; replace=false)

                for k in idx
                    push!(accepted_subject_params, (
                        iter = i,
                        a = cur_pop.a[k],
                        c = cur_pop.c[k],
                        u = cur_pop.u[k]
                    ))
                    push!(accepted_trajs, vec(cur_pop.Y[:, k]))
                end
            end
        end

        mu_u_chain[i] = cur_mu_u
        sigma_u_chain[i] = cur_sigma_u
        dist_chain[i] = cur_dist
    end

    elapsed = now() - start_time
    post_accept = count(accepted_flag[(burn_in_iter+1):end])
    acc_rate = post_accept / max(N_iter - burn_in_iter, 1)

    println("AGE=$(case.age) | elapsed = $elapsed")
    println("AGE=$(case.age) | post-burn accepted = $post_accept")
    println("AGE=$(case.age) | post-burn acceptance rate = $(round(acc_rate, digits=4))")
    println("AGE=$(case.age) | final distance = $(round(cur_dist, digits=5))")

    return (
        mu_u_chain = mu_u_chain,
        sigma_u_chain = sigma_u_chain,
        dist_chain = dist_chain,
        accepted_flag = accepted_flag,
        accepted_samples = accepted_samples,
        accepted_pop_summaries = accepted_pop_summaries,
        accepted_subject_params = accepted_subject_params,
        accepted_trajs = accepted_trajs,
        acceptance_rate = acc_rate
    )
end

# ==========================================================
# POSTERIOR SUMMARY
# ==========================================================
function summarize_accepted_populations(case, fit)
    pops = fit.accepted_pop_summaries
    isempty(pops) && return nothing

    nt = length(case.times)
    np = length(pops)

    M_mean = zeros(nt, np)
    M_sd = zeros(nt, np)
    M_q1 = zeros(nt, np)
    M_med = zeros(nt, np)
    M_q3 = zeros(nt, np)
    dvec = zeros(np)

    for k in 1:np
        pk = pops[k]
        M_mean[:, k] .= pk.mean
        M_sd[:, k] .= pk.sd
        M_q1[:, k] .= pk.q1
        M_med[:, k] .= pk.median
        M_q3[:, k] .= pk.q3
        dvec[k] = pk.distance
    end

    return (
        mean = [quantile(view(M_mean, i, :), 0.50) for i in 1:nt],
        sd = [quantile(view(M_sd, i, :), 0.50) for i in 1:nt],
        q1 = [quantile(view(M_q1, i, :), 0.50) for i in 1:nt],
        median = [quantile(view(M_med, i, :), 0.50) for i in 1:nt],
        q3 = [quantile(view(M_q3, i, :), 0.50) for i in 1:nt],
        distance_median = quantile(dvec, 0.50)
    )
end

function comparison_table(case, fit)
    summ = summarize_accepted_populations(case, fit)
    summ === nothing && return DataFrame()

    return DataFrame(
        TIME = Int.(case.times),
        ObsMean = round.(case.target_mean, digits=2),
        VPMean = round.(summ.mean, digits=2),
        ObsSD = round.(case.target_sd, digits=2),
        VPSD = round.(summ.sd, digits=2),
        ObsQ1 = round.(case.target_q1, digits=2),
        VPQ1 = round.(summ.q1, digits=2),
        ObsMedian = round.(case.target_median, digits=2),
        VPMedian = round.(summ.median, digits=2),
        ObsQ3 = round.(case.target_q3, digits=2),
        VPQ3 = round.(summ.q3, digits=2),
    )
end

# ==========================================================
# MAIN
# ==========================================================
df = CSV.read(csv_path, DataFrame)

needed = ["AGE","TIME","N","Mean","SD","Min","Q1","Median","Q3","Max","Type"]
missing = setdiff(needed, names(df))
isempty(missing) || error("Missing required columns: $(missing)")

d_ab = prep_ab(df)
ages = sort(unique(Int.(d_ab.AGE)))
println("CD4 ages available: ", ages)

cases = Dict{Int,Any}()
fits = Dict{Int,Any}()

for age in ages
    cases[age] = make_age_case(d_ab, age; t0=t0, tukey_k=tukey_k)
    println("AGE=$age times: ", cases[age].times)
    @printf("AGE=%d | baseline LogNormal(mu=%.4f, sigma=%.4f)\n", age, cases[age].baseline_mu, cases[age].baseline_sigma)
    @printf("AGE=%d | peak     LogNormal(mu=%.4f, sigma=%.4f)\n", age, cases[age].peak_mu, cases[age].peak_sigma)
    @printf("AGE=%d | strict TIME=0 bounds   = [%.4f, %.4f]\n", age, cases[age].base_min, cases[age].base_max)
    @printf("AGE=%d | strict TIME=90 bounds  = [%.4f, %.4f]\n", age, cases[age].peak_min, cases[age].peak_max)
    @printf("AGE=%d | support_lo = %s\n", age, string(round.(cases[age].support_lo, digits=2)))
    @printf("AGE=%d | support_hi = %s\n", age, string(round.(cases[age].support_hi, digits=2)))
end

for age in ages
    println("\n=== CD4 | AGE=$age ===")
    fits[age] = run_abc_mcmc_for_age(
        cases[age];
        N_iter = N_iter,
        burn_in_iter = burn_in_iter,
        N_sim = N_sim,
        SD_mu_u = SD_mu_u,
        SD_logsigma_u = SD_logsigma_u,
        u_lo = u_lo,
        u_hi = u_hi,
        sigma_u_lo = sigma_u_lo,
        sigma_u_hi = sigma_u_hi,
        abc_epsilon = abc_epsilon,
        abc_kernel_scale = abc_kernel_scale,
        support_weight = support_weight,
        max_saved_trajs_per_accept = max_saved_trajs_per_accept,
        max_init_tries = max_init_tries
    )

    println(comparison_table(cases[age], fits[age]))
end

# ==========================================================
# OPTIONAL: inspect saved subject-level draws
# ==========================================================
for age in ages
    if !haskey(fits, age)
        println("AGE=$age | no fit stored")

        continue
    end

    sparams = fits[age].accepted_subject_params
    if isempty(sparams)
        println("AGE=$age | no accepted subject-level parameter draws saved")
        continue
    end

    df_subj = DataFrame(
        iter = [x.iter for x in sparams],
        a = [x.a for x in sparams],
        c = [x.c for x in sparams],
        u = [x.u for x in sparams]
    )

    println("\nAGE=$age | saved heterogeneous subject draws:")
    println(first(df_subj, min(10, nrow(df_subj))))
end
