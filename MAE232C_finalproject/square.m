E = 3e+4; % unit: psi

v = .4999; % poisson's ratio



% Strain - displacement relationship for plane strain
lambda = v * E / ((1 + v) * (1 - 2 * v));
mu = E / (2 * (1 + v));

Cs = zeros(3, 3, 3, 3);
Cs_v = zeros(3, 3, 3, 3);
Cs_d = zeros(3, 3, 3, 3);
I = eye(3);

for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                Cs(i,j,k,l) = Cs(i,j,k,l) + lambda * I(i,j) * I(k,l) + mu * (I(i,k) * I(j,l) + I(i,l) * I(j,k));
                Cs_v(i,j,k,l) = Cs_v(i,j,k,l) + lambda * I(i,j) * I(k,l);
                Cs_d(i,j,k,l) = Cs_d(i,j,k,l) + mu * (I(i,k) * I(j,l) + I(i,l) * I(j,k));
            end
        end
    end
end


F = [1,1,0;0,-1,0;0,0,1];
[D, T, Eps] = MaterialModel(F, Cs)

function [D, T, Eps] = MaterialModel(F, Cs)
    C = F' * F;
    E = 1/2 * (C - eye(3));
    D = zeros(4,4);
    T = zeros(4,4);
    Eps = zeros(4,1);

    D_tensor = zeros(3,3,3,3);
    S = zeros(3,3);
    
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    S(i,j) = S(i,j) + Cs(i,j,k,l) * E(k,l);
                    for p = 1:3
                        for q = 1:3
                            D_tensor(i,j,k,l) = D_tensor(i,j,k,l) + F(i,p) * F(k,q) * Cs(p,j,q,l);
                        end
                    end
                end
            end
        end
    end
    
    stress = S * F';
    
    index = [1, 1; 2, 2; 1, 2; 2, 1];
    for i = 1:4
        Eps(i,1) = stress(index(i,1), index(i,2));
        for j = 1:4
            D(i,j) = D_tensor(index(i,1), index(i,2), index(j,1), index(j,2));
        end
    end
    
    T(1,1) = S(1,1);
    T(1,3) = S(1,2);
    T(2,2) = S(2,2);
    T(2,4) = S(2,1);
    T(3,1) = S(2,1);
    T(3,3) = S(2,2);
    T(4,2) = S(1,2);
    T(4,4) = S(1,1);
    
end
