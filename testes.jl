function teste1()
    #%% ESSE TEM SOLUÇÃO SIMPLES SEM FASE 1 - FVAL = 9.33
    A = float([2 1 1 0; 1 2 0 1])
    b = float([4 ; 4])
    c = -float([4 ; 3; 0; 0])
    return A, b, c,  9.33
end

function teste2()
    #%% ESSE EH INVIAVEL
    
    A = [1 3; 3 2;-1 -3]; 
    b = [8 ;12 ;-13];
    c = -[ 1 1 ];

    return A, b, c, NaN
end
 
function teste3()
    #%% ESSE MAXIMIZA |2x+1| (Ou seja, unbounded)
    
    A = [-1 2 -2; -1 -2 2 ]; 
    b = [-1;-1];
    c = -[1 0 0];

    return A, b, c, Inf
end 

function teste4()

    #%% ESSE TEM SOLUÇÃO PEDINDO FASE 2 - FVAL = -19

    A = [-1 -4 ; -5 -1];
    b = [-24;-25];
    c = -[-1 -3];
    return A, b, c, -19
end

function teste5()

    # %% ESSE CICLA, EXEMPLO DE BEALE FVAL = 1.25
    A = [ 1/4 -8 -1 9;
          1/2 -12 -1/2 3;
          0 0 1 0 ];
    b = [ 0 ; 0 ; 1 ];
    c = [-3/4 20 -1/2 6];

    return A, b, c, 1.25
end


function teste6()
    # %% ESSE EH O PROB DA EXPANSAO FVAL = 11.7143

    A = [ 2 1 -1 0;
        1 2 0 -1;
        0 0 0.7 0.5];
    b = [ 4;4 ;1];
    c = -[4 3 0 0 ];
    return A, b, c, 11.7143
end

function teste7()
    #%% Outro da expansão FVAL = 11.7143

    A = [ 2 1 -1 0;
        1 2 0 -1;
        0 0 0.7 0.5];
    b = [ 4;4 ;1];
    c = -[4 3 0 0 ];

    return A, b, c, 11.7143
end

function teste8()
    # %% ESSE EH O PROB DE MAIS UM PROD FVAL = 9.333
    A = [ 2 1 2;
        1 2 2;];
    b = [ 4;4];
    c = -[4 3 3.5];

    return A, b, c, 9.333
end

function testando()

    A, b, c, result  = teste1()
    x, z, status = Simplex(A, b, c)
    if z != result
        throw("failed!")
    end
end
