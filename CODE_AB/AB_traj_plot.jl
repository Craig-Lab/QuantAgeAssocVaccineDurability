using PlotlyJS
using StatsBase
using DataFrames

########################################################
# ONE COMBINED LARGE VP PLOT
# TRAJECTORIES ONLY
# COLOUR CODED BY AGE
# WITH LEGEND
########################################################

max_trajs_per_age = 60
dense_dt = 2.0
plot_all = false

age_colors = Dict(
    50 => "#785EF0",
    60 => "#009E73",
    70 => "#D55E00"
)

function simulate_persistence(times::AbstractVector, a, c, u, t0)
    Δ = times .- t0
    return a .+ (c - a) .* exp.(-u .* Δ)
end

large_vp = Dict{Int,DataFrame}()

for age in ages
    sparams = fits[age].accepted_subject_params
    large_vp[age] = DataFrame(
        a = [x.a for x in sparams],
        c = [x.c for x in sparams],
        u = [x.u for x in sparams]
    )
end

function plot_combined_trajectories_only(
    cases,
    large_vp;
    title = "Anti-gE Antibody titre sample trajectories by age",
    max_trajs_per_age = 60,
    dense_dt = 2.0,
    plot_all = false
)

    traces = PlotlyJS.GenericTrace[]

    for age in ages
        case = cases[age]
        vp = large_vp[age]
        col = age_colors[age]

        times = Float64.(case.times)
        t_dense = collect(minimum(times):dense_dt:maximum(times))

        n_available = nrow(vp)

        idx = plot_all ? collect(1:n_available) :
            sample(1:n_available, min(n_available, max_trajs_per_age); replace=false)

        first_trace_for_age = true

        for k in idx
            s = vp[k, :]
            y = simulate_persistence(t_dense, s.a, s.c, s.u, case.t0)

            push!(traces,
                scatter(
                    x = t_dense,
                    y = y,
                    mode = "lines",
                    line = attr(color = col, width = 1.2),
                    opacity = 0.22,
                    showlegend = first_trace_for_age,
                    name = "Age $age",
                    legendgroup = "Age $age"
                )
            )

            first_trace_for_age = false
        end
    end

    layout = Layout(
        title = title,
        xaxis = attr(
            title = "Time (days)",
            showline = true,
            linecolor = "black"
        ),
        yaxis = attr(
            title = "Anti-gE antibody titre",
            type = "linear",
            showline = true,
            linecolor = "black"
        ),
        plot_bgcolor = "white",
        width = 1100,
        height = 750,
        font = attr(family = "Arial", size = 18),
        showlegend = true,
        legend = attr(
            x = 0.82,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0.7)",
            bordercolor = "black",
            borderwidth = 1
        )
    )

    return plot(traces, layout)
end

p_combined_traj = plot_combined_trajectories_only(
    cases,
    large_vp;
    title = "Anti-gE Antibody titre sample trajectories by age",
    max_trajs_per_age = max_trajs_per_age,
    dense_dt = dense_dt,
    plot_all = false
)

#display(p_combined_traj)

function plot_combined_trajectories_only_better_linear(
    cases,
    large_vp;
    title = "Anti-gE antibody titre VPOP sample trajectories by age",
    max_trajs_per_age = 60,
    dense_dt = 2.0,
    plot_all = false
)

    traces = PlotlyJS.GenericTrace[]

    for age in ages
        case = cases[age]
        vp = large_vp[age]
        col = age_colors[age]

        times = Float64.(case.times)
        t_dense = collect(minimum(times):dense_dt:maximum(times))

        n_available = nrow(vp)

        idx = plot_all ? collect(1:n_available) :
            sample(
                1:n_available,
                min(n_available, max_trajs_per_age);
                replace = false
            )

        for k in idx
            s = vp[k, :]

            y_raw = simulate_persistence(
                t_dense,
                s.a,
                s.c,
                s.u,
                case.t0
            )

            y = log10.(y_raw)

            push!(
                traces,
                scatter(
                    x = t_dense,
                    y = y,
                    mode = "lines",
                    line = attr(
                        color = col,
                        width = 1.4
                    ),
                    opacity = 0.22,
                    showlegend = false,
                    hoverinfo = "skip",
                    legendgroup = "Age $age"
                )
            )
        end
    end

    n_ages = length(ages)
    age_index = Dict(age => i for (i, age) in enumerate(ages))

    box_offset_step = 10.0
    z97 = 2.1700903775845606

    for age in ages
        case = cases[age]
        col = age_colors[age]

        times = Float64.(case.times)

        offset = (age_index[age] - (n_ages + 1) / 2) * box_offset_step
        x_box = times .+ offset

        q1_raw = Float64.(case.target_q1)
        med_raw = Float64.(case.target_median)
        q3_raw = Float64.(case.target_q3)

        mu_log = log.(med_raw)

        sigma_log = (
            log.(q3_raw) .- log.(q1_raw)
        ) ./ 1.3489795

        lower97_raw = exp.(mu_log .- z97 .* sigma_log)
        upper97_raw = exp.(mu_log .+ z97 .* sigma_log)

        push!(
            traces,
            box(
                x = x_box,

                q1 = log10.(q1_raw),
                median = log10.(med_raw),
                q3 = log10.(q3_raw),

                lowerfence = log10.(lower97_raw),
                upperfence = log10.(upper97_raw),

                name = "Age $age observed",
                legendgroup = "Age $age",

                width = 13.0,
                whiskerwidth = 0.75,

                fillcolor = col,
                opacity = 0.55,

                line = attr(
                    color = col,
                    width = 2.5
                ),

                marker = attr(
                    color = col,
                    size = 6
                ),

                boxpoints = false,
                showlegend = false
            )
        )

        push!(
            traces,
            scatter(
                x = x_box,
                y = log10.(med_raw),
                mode = "markers",
                marker = attr(
                    color = "black",
                    size = 10,
                    symbol = "line-ew",
                    line = attr(
                        color = "black",
                        width = 2.5
                    )
                ),
                showlegend = false,
                hoverinfo = "skip",
                legendgroup = "Age $age"
            )
        )
    end

    for age in ages
        col = age_colors[age]

        push!(
            traces,
            scatter(
                x = [nothing],
                y = [nothing],
                mode = "lines",
                line = attr(
                    color = col,
                    width = 5
                ),
                name = "Age $age",
                legendgroup = "Age $age",
                showlegend = true
            )
        )
    end

    layout = Layout(
        title = attr(
            text = title,
            x = 0.5,
            xanchor = "center"
        ),

        xaxis = attr(
            title = "Time (Days)",

            tickmode = "array",
            tickvals = [90, 420, 780, 1140],
            ticktext = ["90", "420", "780", "1140"],

            showline = true,
            linecolor = "black",
            linewidth = 2,
            mirror = false,
            zeroline = false
        ),

        yaxis = attr(
            title = "Anti-gE antibody titre (Log10)",
            tickformat = ".0f",
            exponentformat = "none",
            separatethousands = false,

            showline = true,
            linecolor = "black",
            linewidth = 2,
            mirror = false,
            zeroline = false
        ),

        plot_bgcolor = "white",
        paper_bgcolor = "white",

        width = 1400,
        height = 750,

        font = attr(
            family = "Helvetica",
            size = 18,
            color = "black"
        ),

        boxmode = "overlay",

        showlegend = true,

        legend = attr(
            x = 0.82,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0)",
            borderwidth = 0
        )
    )

    return plot(traces, layout)
end


p_combined_traj_better_linear =
    plot_combined_trajectories_only_better_linear(
        cases,
        large_vp;
        title = "",
        max_trajs_per_age = max_trajs_per_age,
        dense_dt = dense_dt,
        plot_all = false
    )

display(p_combined_traj_better_linear)

