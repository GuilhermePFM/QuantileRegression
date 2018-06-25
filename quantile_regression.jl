using JuMP
using Clp
using MathProgBase
include(joinpath(pwd(), "Simplex", "Simplex.jl" ))
include(joinpath(pwd(), "interiorpoints", "interiorpoints.jl" ))
using DataFrames

function quantile_regression()
    θ = 0.5
    m = build_problem(θ)
    benchmark(m)
end

function build_problem(θ)
    m = Model(solver = ClpSolver())

    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    n_amostras = 10
    n_usinas = 1
    n_AR = 2

    @variable(m,  β[1:n_AR] >= 0 )

    @objective(m, Min, sum(check_function(Y[i], x[i], β) * abs(Y[i] - x[i]) for i in 1:n_amostras))
    
    JuMP.build(m)

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

function quantile_problem(τ, Y, X)
    q = τ
    
    # Y = x * β - ϵ 
    # ϵ = Y - x * β
    # estimador para β:
    # βe = argmin (Σρ*(Y - x * β))
    n_amostras = size(Y)[1]
    n_AR = 1
    
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

function benchmark(m::JuMP.Model)

    A, b, c = extract_problem(m)

    # solve with JuMP
    z, status, time = solve_jump(m)

    # solve with Simplex
    z, x, status, time, it  = solve_simplex(A, b, c) # Simplex works only with max

    # solve with Interior Points
    z, x, status, time, it  = solve_interior_points(A, b, c)

end

function solve_jump(m::JuMP.Model)
    # solve with JuMP
    tic()
    status = solve(m)
    time = toc()
    
    internal_model = internalmodel(m)

    z = getobjectivevalue(m)

    # simplex_it = getsimplexiter(m)

    # barrier_it = getbarrieriter(m)

    return z, status, time #, simplex_it, barrier_it
end

function solve_simplex(A, b, c)
    tic()
    x, z, status, it = Simplex(A, b, c)
    time = toc()

    return z, x, status, time, it 
end

function solve_interior_points(A, b, c)

    tic()
    x, z, status, it = interior_points(A, b, c)
    time = toc()

    return z, x, status, time, it 
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