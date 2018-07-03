using JuMP
using Clp
using MathProgBase
include(joinpath(pwd(), "Simplex", "Simplex.jl" ))
include(joinpath(pwd(), "interiorpoints", "interiorpoints.jl" ))
using DataFrames
# using Plots
# using PyPlot
# using Plotly
using GR
using Rsvg

function quantile_regression()
    quantiles = [0.25, 0.5, 0.75, 0.9]    
    n_amostras = collect(1:10:40)#collect(1:10:60)
    τ = 0.25
    n_amostra = 10

    # Health
    df = readtable("quantile_health.csv")
    names(df)
    Y = df[:totexp]
    X = df[:age]


    # Eolica
    # df = readtable("hrrenwne0.csv")
    # n_amostras = 100#size(df)[1]
    # names(df)
    # Y = df[:totexp]
    # X = df[:age]
    results = Dict()

    # plotly() # Switch to using the PyPlot.jl backend
    # Plots.plot()
    for τ in quantiles
        partial = Dict()
        for n_amostra in n_amostras

            m = quantile_problem(τ, Y, X, n_amostra)
            partial[n_amostra] = benchmark(m)
        end
        plot_time(partial, τ)
        results[τ] = partial
    end

    return 
end

function build_problem(τ, Y, X)
    # 50 o simplex ja fica lento
    # 10 ok
    # 20 ok

    # 30 ok interior
    # 40 ok interior
    # 50 ok interior
    # 80 ok
    # 100
    n_amostras = 20
    m = quantile_problem(τ, Y, X)
    
    return m 
end

function QAR_problem(τ, Y, X)
    q = τ
    
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    n_amostras = size(Y)[1]
    n_AR = 1
    
    m = Model(solver = ClpSolver())
    @variable(m,  β[1:(n_AR + 1)] )

    # positive epsilon
    @variable(m,  u[1:n_amostras] >= 0 )
    # negative epsilon
    @variable(m,  v[1:n_amostras] >= 0 )
    @constraint(m,  [i = 1:n_amostras], u[i] - v[i] == Y[i] - X[i] * β[2] - β[1]) #sum(x[i,j]* β[j] for j in 1:n_AR)

    
    @objective(m, Min, sum( q*u[i] + (1-q)*v[i] for i in 1:n_amostras))
    
    status = solve(m)
    
    println("β = ", getvalue(β))
    println("age = ", getvalue(β)[4])
    println("u = ", getvalue(u))
    println("v = ", getvalue(v))
    
    #print(m)
    JuMP.build(m)

    return m 
end

function quantile_problem(τ, Y, X, n_amostras, n_AR = 1)
    q = τ
    
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    
    m = Model(solver = ClpSolver())
    @variable(m,  β[1:n_AR] )

    # positive epsilon
    @variable(m,  u[1:n_amostras] >= 0 )
    # negative epsilon
    @variable(m,  v[1:n_amostras] >= 0 )
    @constraint(m,  [i = 1:n_amostras], u[i] - v[i] == Y[i] - X[i] * β[1]) #sum(x[i,j]* β[j] for j in 1:n_AR)
    
    
    @objective(m, Min, sum( q*u[i] + (1-q)*v[i] for i in 1:n_amostras))
    
    JuMP.build(m)
    
    return m 
end

function MRE_problem(GF, GSF, Ger_disp, P_vend, Q_vend, Q_disp, PLD, P_fixo, CVU)
    # Premissas
    # GSF : varia por estagio
    # GF : cte
    # Ger_disp : varia por estagio e por agente
    # P_vend : cte
    # Q_vend : cte
    # Q_disp : varia por estagio e por agente
    # PLD : varia por estagio
    # P_fixo : varia por estagio e por agente
    # CVU : varia por agente

    # numero de agentes
    AG = size(GSF)[2]
    
    # numero de etapas
    T = size(GSF)[1]
    
    # numero de contratos
    #C = size(Q_disp)[2]


    m = Model(solver = ClpSolver())
    
    # variaveis de decisao 
    @variable(m, 0 <= a_disp[t=1:T, ag=1:AG] <= 0.05 )
    @variable(m, 0 <= C_red <= 0.05 * Q_vend)
    
    # restricoes

    # montante disponivel
    @constraint(m,  MontDisp[t=1:T], 0 <= sum(a_disp[t, :]) <= 0.1)
    
    # balanco do contrato
    @constraint(m,  BalCont[t=1:T] , Q_vend - C_red <= GF[ag] +  sum(a_disp[t, ag] * Q_disp[t, ag] for ag in 1:AG))

    @objective(m, Max, (sum((GSF[t] * GF - Q_vend)* PLD[t] for t in 1:T) + sum(Ger_disp[t, ag] * a_disp[t, ag]* PLD[t] for ag in 1:AG, t in 1:T) + P_vend * (Q_vend - C_red) - sum(P_fixo[ag] * a_disp[t, ag] * Q_disp[t, ag] for ag in 1:AG, t in t:T) - sum(CVU[ag] * a_disp[t, ag] * Ger_disp[t, ag] for t in 1:T, ag in 1:AG)))


    JuMP.build(m)

    return m 
end

function check_function(Y, x, q)
    if Y - x < 0
        return (1 - q)
    else
        return q
    end
end 

function extract_problem(m::JuMP.Model)
    
    # get JuMP internal model
    internal_model = JuMP.internalmodel(m)
    
    # get matrix
    A = Matrix(MathProgBase.getconstrmatrix(internal_model))
    b = MathProgBase.getconstrLB(internal_model)
    c = MathProgBase.getobj(internal_model)

    return A, b, c
end

struct Result
    z::Float64
    x::Array{Float64}
    status::Int
    time::Float64
    it::Int
    name::String
end

struct BenchResult
    jmp::Result
    smp::Result
    int::Result
end

function benchmark(m::JuMP.Model)
    A, b, c = extract_problem(m)

    # solve with JuMP
   jmp_result = solve_jump(m)

    # solve with Simplex
    smp_result  = solve_simplex(A, b, c) # Simplex works only with max

    # solve with Interior Points
    int_result = solve_interior_points(A, b, c)

    return BenchResult(jmp_result, smp_result, int_result)
end

function solve_jump(m::JuMP.Model)
    # solve with JuMP
    tic()
    status = solve(m)
    time = toc()

    z = getobjectivevalue(m)

    # simplex_it = getsimplexiter(m)

    # barrier_it = getbarrieriter(m)

    x_jmp = getvalue(getvariable(m, :β))
    u_jmp = getvalue(getvariable(m, :u))
    v_jmp = getvalue(getvariable(m, :v))
    x_jmp2= [x_jmp; u_jmp; v_jmp]
    
    if status == :Optimal
        status = 1
    elseif status == :Unbounded
        status = -1
    else
        status = -2
    end
    jmp_result = Result(z, x_jmp2, status, time, 0, "Clp")

    return jmp_result#z, status, time #, simplex_it, barrier_it
end

function solve_simplex(A, b, c)
    tic()
    # A = [A eye(size(A)[1])]
    # c = -[c; zeros(size(A)[1])]
    x, z, status, it = Simplex(A, b, -c, false)
    time = toc()

    smp_result = Result(z, x, status, time, it, "Simplex")

    return smp_result #z, x, status, time, it 
end

function solve_interior_points(A, b, c)

    tic()
    x, p, s, status, it = interior_points(A, b, c, false)
    time = toc()

    int_result = Result(c'*x, x, status, time, it, "Interior Points")

    return int_result #c'*x, x, status, time, it 
end

function plot_result()

end

function plot_benchmark(results, quantiles, n_amostras)
    Plots.pyplot()

    draws = Array{Vector}(6)
    titles = Array{String}(1, 6)
    for i in n_amostras
        m, s = 2*(rand() - 0.5), rand() + 1
        d = Normal(m, s)
        draws[i] = rand(d, 100)
        t = string(L"$\mu = $", round(m, 2), L", $\sigma = $", round(s, 2))
        titles[i] = t
    end
end

function plot_time(results, τ)
    # pyplot() # Switch to using the PyPlot.jl backend
    # plotly() # Switch to using the PyPlot.jl backend
    # Plots.plot()
    # plotlyjs()
    y = [[] for i in 1:3]
    x = [[] for i in 1:3]
    labels = []
    for n_amostra in keys(results)
        result = results[n_amostra]
        push!(y[1], result.jmp.time)
        push!(y[2], result.smp.time)
        push!(y[3], result.int.time)
        push!(x[1], n_amostra)
        push!(x[2], n_amostra)
        push!(x[3], n_amostra)
        # v = [result.jmp.time, result.smp.time, result.int.time]
        labels = [result.jmp.name  result.smp.name result.int.name]

    end

    graph_name = "Tempo x Numero de Amostras (τ = $(string(τ)))"
    trace1 = scatter(;x=x[1], y=y[1], mode="lines+markers", name = "Clp")
    trace2 = scatter(;x=x[2], y=y[2], mode="lines+markers", name = "Simplex")
    trace3 = scatter(;x=x[3], y=y[3], mode="lines+markers", name = "Pontos Interiores")
    layout = Layout(;title=graph_name, xaxis=attr(title="GDP per Capital", showgrid=false, zeroline=false),
    yaxis=attr(title="Percent", zeroline=false))
    p=plot([trace1, trace2, trace3], layout)
    # p = PlotlyJS.scatter(x=x, y=y, title=graph_name, label = labels, markersize = 5, shape = [:circle, :circle, :circle])

    PlotlyJS.savefig(p, graph_name*".png")
end


function exemple_problem()
    m = Model(solver = ClpSolver())


    @variable(m,  x >= 0 )
    @variable(m,  y >= 0 )

    @objective(m, Min, -5x - 3*y )
    @constraint(m, 1x + 5y >= 3.0 )
    
    JuMP.build(m)
    
    # print(m)

    # internal_model = internalmodel(m)
    # c = MathProgBase.getobj(internal_model)
    # A = Matrix(MathProgBase.getconstrmatrix(internal_model))

    status = solve(m)

    println("Objective value: ", getobjectivevalue(m))
    println("x = ", getvalue(x))
    println("y = ", getvalue(y))
end

function test_jq()
    q = 0.25
    
    n_AR = 1
    X = collect(1:20)
    Y = 10*X + rand()
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    n_amostras = size(Y)[1]
    
    m = Model(solver = ClpSolver())
    @variable(m,  β[1:n_AR] )

    # positive epsilon
    @variable(m,  u[1:n_amostras] >= 0 )
    # negative epsilon
    @variable(m,  v[1:n_amostras] >= 0 )
    @constraint(m,  [i = 1:n_amostras], u[i] - v[i] == Y[i] - X[i] * β[1]) #sum(x[i,j]* β[j] for j in 1:n_AR)

    
    @objective(m, Min, sum( q*u[i] + (1-q)*v[i] for i in 1:n_amostras))
    
    #print(m)
    JuMP.build(m)

    z_jmp, status_jmp, time_jmp = solve_jump(m)
    x_jmp = getvalue(getvariable(m, :β))
    u_jmp = getvalue(getvariable(m, :u))
    v_jmp = getvalue(getvariable(m, :v))
    x_jmp2= [x_jmp; u_jmp; v_jmp]


    A, b, c = extract_problem(m)
    z_smp, x_smp, status_smp, time_smp, it_smp  = solve_simplex(A, b, c) 
    z_int, x_int, status_int, time_int, it_int  = solve_interior_points(A, b, c)

end

function test_quantile()
    df = readtable("quantile_health.csv")
    n_amostras = size(df)[1]
    names(df)
    Y = df[:totexp]
    X = df[:age]
    x_anos = X
    q = 0.25
    
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    n_usinas = 1
    n_AR = 1
    
    m = Model(solver = ClpSolver())
    @variable(m,  β[1:n_AR] )

    # positive epsilon
    @variable(m,  u[1:n_amostras] >= 0 )
    # negative epsilon
    @variable(m,  v[1:n_amostras] >= 0 )
    @constraint(m,  [i = 1:n_amostras], u[i] - v[i] == Y[i] - X[i] * β[1]) #sum(x[i,j]* β[j] for j in 1:n_AR)

    
    @objective(m, Min, sum( q*u[i] + (1-q)*v[i] for i in 1:n_amostras))
    
    status = solve(m)
    
    println("β = ", getvalue(β))
    println("age = ", getvalue(β)[4])
    println("u = ", getvalue(u))
    println("v = ", getvalue(v))
    
    print(m)
    # JuMP.build(m)
end 

function test_quantile_bobo()
   
    
    Y = collect(1:9)
    x = [36, 29, 24, 21, 20, 21, 24, 29, 36]
    n_amostras = length(x)
    q = 0.5
    
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    
    m = Model(solver = ClpSolver())
    @variable(m,  β[1:1] )

    # positive epsilon
    @variable(m,  u[1:n_amostras] >= 0 )
    # negative epsilon
    @variable(m,  v[1:n_amostras] >= 0 )
    @constraint(m,  [i = 1:n_amostras], u[i] - v[i] == Y[i] -  3) #sum(x[i,j]* β[j] for j in 1:n_AR)

    
    @objective(m, Min, sum( q*u[i] + (1-q)*v[i] for i in 1:n_amostras))
    
    status = solve(m)
    
    println("β = ", getvalue(β))
    println("u = ", getvalue(u))
    println("v = ", getvalue(v))
    println("age = ", getvalue(β)[4])
    
    print(m)
    # JuMP.build(m)
end 