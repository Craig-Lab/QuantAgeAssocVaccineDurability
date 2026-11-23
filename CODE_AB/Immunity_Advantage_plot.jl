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

############################################################

measured_days = [90.0, 150 ,300, 450.0, 600.0,750.0, 900.0]
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

function plot_custom_ci_box_spaghetti_annotated(
    measured_raw_rows::DataFrame,
    measured_raw_summary::DataFrame;
    n_spaghetti_per_pair = 15,
    title_str = "Anti-gE Antibody titre immune advantage probability",
    ypad = 0.04,
    box_half_width = 10.0,
    cap_half_width = 16.0,
    spline_smoothing = 1.3
)

    traces = PlotlyJS.GenericTrace[]
    pair_order = unique(measured_raw_summary.Pair)

    ymin = max(0.0, minimum(measured_raw_summary.CI_low) - ypad)
    ymax = min(1.0, maximum(measured_raw_summary.CI_high) + ypad)

    ########################################################
    # SMOOTH SPAGHETTI LINES
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
                    line = attr(
                        color = col,
                        width = 1.5,
                        shape = "spline",
                        smoothing = spline_smoothing
                    ),
                    opacity = 0.0,
                    showlegend = false,
                    legendgroup = pr,
                    name = "$pr spaghetti"
                )
            )
        end
    end

    ########################################################
    # LEGEND ENTRIES FOR PAIR COLORS
    ########################################################

    for pr in pair_order
        col = pair_colors[pr]

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
    end

    ########################################################
    # SMOOTH MEDIAN LINES CONNECTING BOX MEDIANS
    ########################################################

    for pr in pair_order
        col = pair_colors[pr]

        sub = copy(measured_raw_summary[measured_raw_summary.Pair .== pr, :])
        sort!(sub, :Time)

        push!(traces,
            scatter(
                x = sub.Time,
                y = sub.P_median,
                mode = "lines",
                line = attr(
                    color = col,
                    width = 4,
                    shape = "spline",
                    smoothing = spline_smoothing
                ),
                showlegend = false,
                legendgroup = pr,
                name = "$pr median curve"
            )
        )
    end

    ########################################################
    # CUSTOM BOXES + WHISKERS
    ########################################################

    for pr in pair_order
        col = pair_colors[pr]

        sub = copy(measured_raw_summary[measured_raw_summary.Pair .== pr, :])
        sort!(sub, :Time)

        for r in eachrow(sub)
            x = r.Time

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

    ########################################################
    # LAYOUT
    ########################################################

    layout = Layout(
       font = attr(            family = "Helvetica",
            size = 18,
            color = "black"),
        title = attr(
            text = title_str,
            x = 0.5,
            xanchor = "center"
        ),
        xaxis = attr(
            title = "Time (Days)",
            tickmode = "array",
            tickvals = measured_days,
            ticktext = string.(Int.(measured_days)),
            showline = true,
            linecolor = "black"
        ),
        yaxis = attr(
            title = "Humoral Immunity Advantage Probability",
            range = [ymin, ymax],
            showline = true,
            linecolor = "black"
        ),

        width = 1400,
        height = 650,
        plot_bgcolor = "white",
        legend = true
    )

    return plot(traces, layout)
end

p_measured_raw_annotated = plot_custom_ci_box_spaghetti_annotated(
    measured_raw_rows,
    measured_raw_summary;
    n_spaghetti_per_pair = N_SPAGHETTI_PER_PAIR,
    title_str = "",
    spline_smoothing = 1.3
)
