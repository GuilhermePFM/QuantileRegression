using JuMP
using Clp
using GLPK
using GLPKMathProgInterface
using MathProgBase
include(joinpath(pwd(), "Simplex", "Simplex.jl" ))
include(joinpath(pwd(), "interiorpoints", "interiorpoints.jl" ))
using DataFrames

function main()
    Ger_op, GSF, P_op_premio, P_op_exer, Q_op, PLD = MRE_read()
    
    # casos
    # 1) 10 agentes 1 etapa 2 cenarios
    # 2) 10 agentes 10 etapa 2 cenarios
    # 2) 10 agentes 10 etapa 10 cenarios
    # [agentes, etapas, cenarios]
    casos = [[10, 1, 1000, "1"], [10, 5, 500, "5"], [10, 10, 500, "10"], [10, 15, 500, "15"], [10, 20, 500, "20"], [10, 25, 500, "25"], [10, 30, 500, "30"], [10, 35, 500, "35"], [10, 40, 500, "40"], [10, 45, 500, "45"], [10, 50, 500, "50"], [10, 55, 500, "55"], [10, 60, 500, "60"]]
    
    resultado = open("resultado_stress.csv", "w")
    write(resultado, "Caso, Clp, Simplex, Pontos Interiores \n")
    for (ag, et, cen, caso) in casos
        
        m = MRE_problem(Ger_op, GSF[1:et, 1:cen], P_op_premio, P_op_exer[1:et, 1:ag], Q_op, PLD[1:et, 1:cen])
        
        results = benchmark(m)

        # z
        @show results.jmp.x
        @show results.smp.x
        @show results.int.z

        write(resultado, "$caso tempo, $(results.jmp.time), $(results.smp.time), $(results.int.time)\n")
        write(resultado, "$caso fobjt, $(results.jmp.z), $(results.smp.z), $(-results.int.z)\n")

    end
    close(resultado)
end 

function plot_results()
    quantiles = [0.25, 0.5, 0.75, 0.9]    
    n_amostras = collect(1:10:40)#collect(1:10:60)
    τ = 0.25
    n_amostra = 10

    # Health
    # df = readtable("quantile_health.csv")
    # names(df)
    # Y = df[:totexp]
    # X = df[:age]


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

function MRE_read()
    P_op_exer = readtable("EXERCICIO.csv")[:,2:end]

    GSF = readtable("GSF.csv")[:,2:end]

    Op = readtable("OPCAO.csv")[:,2:end]
    P_op_premio = Op[1,:]
    Q_op = Op[2,:]

    PLD = readtable("PLD.csv")[:,2:end]
    
    n_cenarios = 10
    Ger_op = zeros(size(PLD)[1], size(PLD)[2], n_cenarios)
    for i in 1:n_cenarios
        Ger_op_df = readtable("GER_$i.csv")[:,2:end]
        Ger_op[:,:,i] = Matrix(Ger_op_df)
    end
    return Ger_op, GSF, P_op_premio, P_op_exer, Q_op, PLD
end

function MRE_problem(Ger_op, GSF, P_op_premio, P_op_exer, Q_op, PLD)
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
    # GSF, P_op_premio, P_op_exer, Q_op, PLD = [GSF[1:et, 1:cen], P_op_premio, P_op_exer[1:et, 1:ag], Q_op, PLD[1:et, 1:cen]]
    # casta matrix
    GSF = Matrix(GSF)
    P_op_premio = Matrix(P_op_premio)
    P_op_exer = Matrix(P_op_exer)
    Q_op = Matrix(Q_op)
    PLD =  Matrix(PLD)

    # Contrato
    GF = 100
    Q_cont0 = 100
    P_cont = 180
   
    #    Q_op = [50 50]
    # P_op_premio = [10 20]
    # P_op_exer = [200 170]
    # Ger_op = [[50; 50; 0; 0; 0] [50; 50; 0; 0; 0]]
    n_op = size(P_op_premio)[2]
   
    # GSF = [0.8; 0.85; 0.9; 0.95; 1.]
    # PLD = [250; 150; 110; 75; 50]
    n_cen = size(GSF)[2]
    t = size(GSF)[1]

    # decisao = Model(solver = ClpSolver())
    decisao = Model(solver = ClpSolver())

    @variable(decisao, a[1:t,1:n_op] >= 0 )
    @variable(decisao, Q_red >= 0 )
    # slacks
    @variable(decisao, s_Q_red >= 0 )
    @variable(decisao, s_ap[1:t,1:n_op] >= 0 )
    @variable(decisao, s_a_sum[1:t] >= 0 )


    # @constraint(decisao, limite_1, Q_red <= 0.05 * Q_cont0 )
    @constraint(decisao, limite_1, Q_red + s_Q_red == 0.05 * Q_cont0 )

    # @constraint(decisao, limite_2[etapa = 1:t, iop = 1:n_op], a[etapa, iop] <= 0.1)
    # @constraint(decisao, limite_3[etapa = 1:t], sum(a[etapa,:]) <= 0.15 )
    @constraint(decisao, limite_2[etapa = 1:t, iop = 1:n_op], a[etapa, iop] + s_ap[etapa, iop] == 0.1)
    @constraint(decisao, limite_3[etapa = 1:t], sum(a[etapa,:]) + s_a_sum[etapa] == 0.15 )

    @objective(decisao, Max, 
    ( sum((P_cont - PLD[etapa, cen]) * Q_cont0 for cen in 1:n_cen, etapa in 1:t) 
    - Q_red * sum((P_cont - PLD[etapa, cen]) for cen in 1:n_cen, etapa in 1:t) 
    + sum(GSF[etapa, cen] * GF * PLD[etapa, cen] for cen in 1:n_cen, etapa in 1:t)
    + sum(sum(a[etapa,:] .* Ger_op[etapa,cen,:])  for cen in 1:n_cen, etapa in 1:t)
    - sum(sum(a[etapa,:] .* Q_op[:] .* P_op_premio[:]) for cen in 1:n_cen, etapa in 1:t)
    - sum(sum(a[etapa,:] .* Ger_op[etapa,cen,:] .* P_op_exer[etapa,:]) for cen in 1:n_cen, etapa in 1:t)) /n_cen )

    # print(decisao)
    # status = solve(decisao)

    # println("O custo da operação é igual a: ", getobjectivevalue(decisao)," reais")
    # println("a1, a2 = ", getvalue(a))
    # println("Reducao = ", getvalue(Q_red))
    # println("Dual de redução = ",getdual(limite_1))
    # println("Dual do max por contrato = ",getdual(limite_2))
    # println("Dual do max total = ",getdual(limite_3))

    JuMP.build(decisao)

    return decisao 
end

function extract_problem(m::JuMP.Model)
    
    # get JuMP internal model
    internal_model = JuMP.internalmodel(m)
    
    # get matrix
    A = Matrix(MathProgBase.getconstrmatrix(internal_model))

    b = MathProgBase.getconstrLB(internal_model)
    if getobjectivesense(m) == :Max
        b = MathProgBase.getconstrUB(internal_model)
    end
    
    c = MathProgBase.getobj(internal_model)

    # writeLP(m, "lp.lp"; genericnames=false)
 
    # print(m)
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
    jmp_result = solve_jump(m, c)

    if getobjectivesense(m) == :Min
        c = -c
    end
    # solve with Simplex
    smp_result  = solve_simplex(A, b, c) # Simplex works only with max

    # solve with Interior Points
    int_result = solve_interior_points(A, b, -c)

    return BenchResult(jmp_result, smp_result, int_result)
end

function solve_jump(m::JuMP.Model,c)
    # solve with JuMP
    tic()
    status = solve(m)
    time = toc()

    z = getobjectivevalue(m)

    # simplex_it = getsimplexiter(m)

    # barrier_it = getbarrieriter(m)

    a_jmp = getvalue(getvariable(m, :a))[:]
    a_jmp = zeros(size(a_jmp))
    
    s_ap = getvalue(getvariable(m, :s_ap))[:]
    s_ap = zeros(size(s_ap))
    
    idx=1
    for j in 1:size(getvariable(m, :a))[2], i in 1:size(getvariable(m, :a))[1]
        a_jmp[idx] =  getvalue(getvariable(m, :a))[i,j]
        s_ap[idx] = getvalue(getvariable(m, :s_ap))[i,j]
        idx += 1
    end

    Q_jmp = getvalue(getvariable(m, :Q_red))
    s_Q_jmp = getvalue(getvariable(m, :s_Q_red))

    s_a_sum = getvalue(getvariable(m, :s_a_sum))
    
    # v_jmp = getvalue(getvariable(m, :v))
    x_jmp2= [a_jmp; Q_jmp; s_Q_jmp; s_ap; s_a_sum]
    
    if status == :Optimal
        status = 1
    elseif status == :Unbounded
        status = -1
    else
        status = -2
    end
    jmp_result = Result(c'*x_jmp2, x_jmp2, status, time, 0, "Clp")

    return jmp_result#z, status, time #, simplex_it, barrier_it
end

function solve_simplex(A, b, c)
    tic()
    # A = [A eye(size(A)[1])]
    # c = -[c; zeros(size(A)[1])]
    x, z, status, it = Simplex(A, b, c, true)
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
