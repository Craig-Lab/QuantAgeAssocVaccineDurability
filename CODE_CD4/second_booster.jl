############################################################
# BOOSTER THRESHOLD ANALYSIS
# 
#
# No CSV saving.
#
# Requires in memory:
#   ages, cases, fits, df
############################################################

using DataFrames
using Statistics
using StatsBase
using Random
using PlotlyJS

Random.seed!(2027)

############################################################
# SETTINGS
############################################################

boost_age = 70

last_dose_day_young = 60.0
days_after_last_dose = 1000.0
young_eval_time = last_dose_day_young + days_after_last_dose  # 1060

booster_times = [180.0, 365.0]

alpha_grid = collect(0.0:0.025:1.0)
alpha_step = 0.05

pairs_age = [(50,70), (60,70)]

NRUNS = 100
final_pool_rule = "t90"

target_advantage = 0.50

pair_colors_booster = Dict(
    "50 > boosted 70" => "#0072B2",
    "60 > boosted 70" => "#D55E00"
)

############################################################
# MODEL FUNCTIONS
############################################################

function base_persistence(t, a, c, u, t0)
    return a + (c - a) * exp(-u * (t - t0))
end

function boosted_persistence(t, a, c, u, t0, tb, alpha)
    if t < tb
        return base_persistence(t, a, c, u, t0)
    end

    y_before = base_persistence(tb, a, c, u, t0)
    B3 = alpha * (c - a)
    y_after = y_before + B3

    return a + (y_after - a) * exp(-u * (t - tb))
end

############################################################
# FINAL VPOP SAMPLING
############################################################

function get_final_pool_size(case, df, age; rule="t90")
    rows = df[(df.AGE .== age) .& (df.TIME .>= case.t0), :]

    nrow(rows) > 0 || error("No persistence rows found for AGE=$age")

    if rule == "t90"
        r = rows[rows.TIME .== case.t0, :]
        nrow(r) == 0 && error("No TIME=$(case.t0) row for AGE=$age")
        return Int(round(r.N[1]))
    elseif rule == "min"
        return Int(minimum(rows.N))
    elseif rule isa Integer
        return rule
    else
        error("Unknown final_pool_rule = $rule")
    end
end

function build_final_pool_for_age(case, fit, df, age; rule="t90")
    subj = fit.accepted_subject_params
    isempty(subj) && error("No accepted subject params for AGE=$age")

    n_final = get_final_pool_size(case, df, age; rule=rule)

    if length(subj) < n_final
        error("Not enough saved accepted subjects for AGE=$age: have $(length(subj)), need $n_final")
    end

    idx = sample(1:length(subj), n_final; replace=false)

    rows = NamedTuple[]

    for k in idx
        s = subj[k]
        push!(rows, (
            a = Float64(s.a),
            c = Float64(s.c),
            u = Float64(s.u)
        ))
    end

    return DataFrame(rows)
end

############################################################
# PROBABILITY OF SUPERIORITY
############################################################

function fast_pos(x::Vector{Float64}, y::Vector{Float64})
    xs = sort(x)
    ys = sort(y)

    nx = length(xs)
    ny = length(ys)

    j = 1
    wins = 0

    for xi in xs
        while j <= ny && ys[j] < xi
            j += 1
        end
        wins += j - 1
    end

    return wins / (nx * ny)
end

############################################################
# VALUE EXTRACTION
############################################################

function values_at_time(vp::DataFrame, case, t)
    return Float64[
        base_persistence(t, r.a, r.c, r.u, case.t0)
        for r in eachrow(vp)
    ]
end

function boosted_values_at_time(vp::DataFrame, case, t, tb, alpha)
    return Float64[
        boosted_persistence(t, r.a, r.c, r.u, case.t0, tb, alpha)
        for r in eachrow(vp)
    ]
end

############################################################
# SINGLE FAIR REPLICATE
############################################################

function booster_advantage_fair_single_run(
    final_vp,
    cases,
    pairs_age,
    booster_times,
    alpha_grid;
    run_id = 1,
    young_eval_time = 1060.0,
    days_after_booster = 1000.0
)
    rows = NamedTuple[]

    for tb in booster_times
        old_eval_time = tb + days_after_booster

        for alpha in alpha_grid
            for (young, old) in pairs_age
                pair_label = "$young > boosted $old"

                y_young = values_at_time(
                    final_vp[young],
                    cases[young],
                    young_eval_time
                )

                y_old = boosted_values_at_time(
                    final_vp[old],
                    cases[old],
                    old_eval_time,
                    tb,
                    alpha
                )

                p = fast_pos(y_young, y_old)

                push!(rows, (
                    Run = Int(run_id),
                    BoosterTime = Float64(tb),
                    Alpha = Float64(alpha),
                    Pair = pair_label,
                    Young = Int(young),
                    Old = Int(old),
                    YoungEvalTime = Float64(young_eval_time),
                    OldEvalTime = Float64(old_eval_time),
                    DaysAfterLastDose = Float64(days_after_booster),
                    Advantage = Float64(p),
                    N_young = Int(length(y_young)),
                    N_old = Int(length(y_old))
                ))
            end
        end
    end

    return DataFrame(rows)
end

############################################################
# SUMMARY
############################################################

function summarize_booster_fair_results(df::DataFrame)
    g = groupby(df, [
        :BoosterTime,
        :Alpha,
        :Pair,
        :Young,
        :Old,
        :YoungEvalTime,
        :OldEvalTime,
        :DaysAfterLastDose
    ])

    out = combine(
        g,
        :Advantage => median => :Adv_median,
        :Advantage => mean => :Adv_mean,
        :Advantage => std => :Adv_sd,
        :Advantage => (x -> quantile(x, 0.25)) => :Adv_q1,
        :Advantage => (x -> quantile(x, 0.75)) => :Adv_q3,
        :Advantage => (x -> quantile(x, 0.025)) => :PI_low,
        :Advantage => (x -> quantile(x, 0.975)) => :PI_high,
        nrow => :NRuns
    )

    sort!(out, [:BoosterTime, :Pair, :Alpha])
    return out
end

############################################################
# PLOTTING HELPERS
############################################################

function on_alpha_grid(a; step=0.05, atol=1e-8)
    return isapprox(a, round(a / step) * step; atol=atol)
end

function first_crossing_alpha_fair(
    summary_df::DataFrame;
    booster_time,
    target=0.50,
    alpha_step=0.05
)
    sub = summary_df[summary_df.BoosterTime .== booster_time, :]

    sub = sub[
        [on_alpha_grid(a; step=alpha_step) for a in sub.Alpha], :
    ]

    nrow(sub) > 0 || error("No data found for booster_time=$booster_time")

    alpha_crossings = Float64[]

    for pr in unique(sub.Pair)
        s = copy(sub[sub.Pair .== pr, :])
        sort!(s, :Alpha)

        hit = s[s.Adv_median .<= target, :]

        if nrow(hit) > 0
            push!(alpha_crossings, Float64(minimum(hit.Alpha)))
        end
    end

    isempty(alpha_crossings) && return Float64(maximum(sub.Alpha))

    return Float64(maximum(alpha_crossings))
end

############################################################
# BOXPLOT UNTIL FIRST CROSSING
############################################################

function plot_booster_fair_box_until_crossing(
    summary_df::DataFrame;
    booster_time = 180.0,
    target = 0.50,
    alpha_step = 0.05,
    box_half_width = 0.004,
    cap_half_width = 0.007,
    ypad = 0.04
)
    alpha_stop = first_crossing_alpha_fair(
        summary_df;
        booster_time = booster_time,
        target = target,
        alpha_step = alpha_step
    )

    sub_all = summary_df[
        (summary_df.BoosterTime .== booster_time) .&
        (summary_df.Alpha .<= alpha_stop), :
    ]

    sub_all = sub_all[
        [on_alpha_grid(a; step=alpha_step) for a in sub_all.Alpha], :
    ]

    nrow(sub_all) > 0 || error("No data found after filtering")

    traces = PlotlyJS.GenericTrace[]
    pair_order = unique(sub_all.Pair)

    ymin = max(0.0, minimum(sub_all.PI_low) - ypad)
    ymax = min(1.0, maximum(sub_all.PI_high) + ypad)

    for pr in pair_order
        col = get(pair_colors_booster, pr, "gray")

        push!(traces,
            scatter(
                x = [nothing],
                y = [nothing],
                mode = "lines",
                line = attr(color = col, width = 4),
                name = pr,
                legendgroup = pr,
                showlegend = true
            )
        )

        sub = copy(sub_all[sub_all.Pair .== pr, :])
        sort!(sub, :Alpha)

        for r in eachrow(sub)
            x = r.Alpha

            push!(traces,
                scatter(
                    x = [x, x],
                    y = [r.PI_low, r.PI_high],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            push!(traces,
                scatter(
                    x = [x - cap_half_width, x + cap_half_width],
                    y = [r.PI_low, r.PI_low],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            push!(traces,
                scatter(
                    x = [x - cap_half_width, x + cap_half_width],
                    y = [r.PI_high, r.PI_high],
                    mode = "lines",
                    line = attr(color = col, width = 2),
                    showlegend = false,
                    legendgroup = pr
                )
            )

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
                        r.Adv_q1,
                        r.Adv_q1,
                        r.Adv_q3,
                        r.Adv_q3,
                        r.Adv_q1
                    ],
                    mode = "lines",
                    fill = "toself",
                    fillcolor = "rgba(255,255,255,0)",
                    line = attr(color = col, width = 2.5),
                    showlegend = false,
                    legendgroup = pr
                )
            )

            push!(traces,
                scatter(
                    x = [x - box_half_width, x + box_half_width],
                    y = [r.Adv_median, r.Adv_median],
                    mode = "lines",
                    line = attr(color = col, width = 4),
                    showlegend = false,
                    legendgroup = pr
                )
            )
        end
    end

    push!(traces,
        scatter(
            x = [minimum(sub_all.Alpha), maximum(sub_all.Alpha)],
            y = [target, target],
            mode = "lines",
            line = attr(color = "black", dash = "dash", width = 2),
            showlegend = false
        )
    )

    alpha_ticks = sort(unique(sub_all.Alpha))

    old_eval_time = unique(sub_all.OldEvalTime)[1]
    young_eval_time_plot = unique(sub_all.YoungEvalTime)[1]

    layout = Layout(
        font = attr(family="Helvetica", size=18),
        title = attr(text = "", x = 0.5, xanchor = "center"),
        xaxis = attr(
            title = "Booster strength relative to two-dose response ",
            tickmode = "array",
            tickvals = alpha_ticks,
            ticktext = string.(round.(alpha_ticks, digits=2)),
            showline = true,
            linecolor = "black"
        ),
        yaxis = attr(
            title = "Humoral Immunity Advantage Probability",
            range = [ymin, ymax],
            showline = true,
            linecolor = "black"
        ),
        annotations = [
            attr(
                text = "",
                xref = "paper",
                yref = "paper",
                x = 0.5,
                y = 1.08,
                showarrow = false,
                font = attr(size = 15)
            )
        ],
        width = 1000,
        height = 600,
        plot_bgcolor = "white",
        legend = true
    )

    return plot(traces, layout)
end

############################################################
# MAIN LOOP
############################################################

all_rows = DataFrame()

for run_id in 1:NRUNS
    println("Fair booster scenario replicate $run_id / $NRUNS")

    final_vp = Dict{Int,DataFrame}()

    for age in ages
        final_vp[age] = build_final_pool_for_age(
            cases[age],
            fits[age],
            df,
            age;
            rule = final_pool_rule
        )
    end

    rep_df = booster_advantage_fair_single_run(
        final_vp,
        cases,
        pairs_age,
        booster_times,
        alpha_grid;
        run_id = run_id,
        young_eval_time = young_eval_time,
        days_after_booster = days_after_last_dose
    )

    append!(all_rows, rep_df)
end

############################################################
# SUMMARIZE
############################################################

booster_fair_summary = summarize_booster_fair_results(all_rows)

println("\nFair comparison complete.")
println("50/60 evaluated at day $(Int(young_eval_time))")
println("70+ booster at day 180 evaluated at day 1180")
println("70+ booster at day 365 evaluated at day 1365")

println("\nSummary:")
println(booster_fair_summary)

############################################################
# MAKE PLOTS
############################################################

p_boost_180_fair = plot_booster_fair_box_until_crossing(
    booster_fair_summary;
    booster_time = 180.0,
    target = target_advantage,
    alpha_step = alpha_step
)

display(p_boost_180_fair)

p_boost_365_fair = plot_booster_fair_box_until_crossing(
    booster_fair_summary;
    booster_time = 365.0,
    target = target_advantage,
    alpha_step = alpha_step
)

display(p_boost_365_fair)
