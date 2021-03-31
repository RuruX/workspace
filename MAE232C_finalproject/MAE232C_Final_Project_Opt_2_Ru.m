
% Final Project-Option.2
% 2-D beam with Hyperelasticity material (Saint Venant-Kirchhoff Model) 

% June 12, 2019
% Ru Xiang

clc
close all
clear all

% ------------------------------------------------------------------------

% Step 0: Initialization


% A: Read the exact tip diflection

S = load('Ref_Soln_Opt_2.mat');
Load = S.FinalProjectReferenceSolution(:,1);
tipExact = S.FinalProjectReferenceSolution(:,2);


% B: Save the invariants
% setting: 500 load steps, tolerance = 10^-4

length = 50; height = 5; % unit: inch
E = 3e+4; % unit: psi
P = -250; % unit: lb

v = .4999; % poisson's ratio

N_L = 500;
dP = P / N_L;
tol = 1e-4;

% Strain - displacement relationship for plane strain
lambda = v * E / ((1 + v) * (1 - 2 * v));
mu = E / (2 * (1 + v));
Cs = zeros(3, 3, 3, 3);
Cs_v = zeros(3, 3, 3, 3);
Cs_d = zeros(3, 3, 3, 3);
IM = eye(3);


for i = 1:3
    for j = 1:3
        for k = 1:3
            for l = 1:3
                Cs(i,j,k,l) = Cs(i,j,k,l) + lambda * IM(i,j) * IM(k,l) + mu * (IM(i,k) * IM(j,l) + IM(i,l) * IM(j,k));
                Cs_v(i,j,k,l) = Cs_v(i,j,k,l) + lambda * IM(i,j) * IM(k,l);
                Cs_d(i,j,k,l) = Cs_d(i,j,k,l) + mu * (IM(i,k) * IM(j,l) + IM(i,l) * IM(j,k));
            end
        end
    end
end

C = zeros(4);
index = [1, 1; 2, 2; 1, 2; 2, 1];
for i = 1:4
    for j = 1:4
        indice = [index(i,1), index(i,2), index(j,1), index(j,2)];
        C(i,j) = Cs(indice(1),indice(2),indice(3),indice(4));
    end
end


mesh_opts = [2, 20; 4, 40; 8, 80; 16, 160];

option = 1;
num_ele_x = mesh_opts(option, 2);
num_ele_y = mesh_opts(option, 1);

num_ele = num_ele_x * num_ele_y;
num_node = (num_ele_x + 1) * (num_ele_y + 1);
l_ele_x = length / num_ele_x;
l_ele_y = height / num_ele_y;

% element DOF matrix for 4-node elements
ele_DOF = zeros(num_ele, 4);

for i = 1 : num_ele_x
    for j = 1 : num_ele_y
        ele = i + num_ele_x * (j - 1);
        n1 = i + (num_ele_x + 1) * (j - 1); 
        ele_DOF(ele,:) = [n1, n1 + 1, n1 + num_ele_x + 2, n1 + num_ele_x + 1];
    end
end

node_coords = zeros(num_node, 2);

% nodal coordinates for each element
for i = 1 : (num_ele_x + 1)
    for j = 1 : (num_ele_y + 1)
        node = i + (num_ele_x + 1) * (j - 1);
        x_coord = (i - 1) * l_ele_x;
        y_coord = - height / 2 + (j - 1) * l_ele_y;
        node_coords(node, :) = [x_coord, y_coord];
    end
end

% Gauss integration formula of order 2
num_gp = 4;
GP = [-1/sqrt(3), 1/sqrt(3), 1/sqrt(3), -1/sqrt(3);
    -1/sqrt(3), -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)];
W = [1, 1, 1, 1];
% A = 4 * J(0) is the area of the element
A = l_ele_x * l_ele_y;

% RI: reduced integration
pN_RI = [-1/4, 1/4, 1/4, -1/4; -1/4, -1/4, 1/4, 1/4];
pN_RI_xy = [pN_RI(1,:) * 2/l_ele_x; pN_RI(2,:) * 2/l_ele_y];
B_RI_1 = [pN_RI_xy(1,1), 0; 0, pN_RI_xy(2,1); pN_RI_xy(2,1), 0; 0, pN_RI_xy(1,1)];
B_RI_2 = [pN_RI_xy(1,2), 0; 0, pN_RI_xy(2,2); pN_RI_xy(2,2), 0; 0, pN_RI_xy(1,2)];
B_RI_3 = [pN_RI_xy(1,3), 0; 0, pN_RI_xy(2,3); pN_RI_xy(2,3), 0; 0, pN_RI_xy(1,3)];
B_RI_4 = [pN_RI_xy(1,4), 0; 0, pN_RI_xy(2,4); pN_RI_xy(2,4), 0; 0, pN_RI_xy(1,4)];
B_RI = [B_RI_1, B_RI_2, B_RI_3, B_RI_4];

B_total_FI = cell(num_ele, num_gp);
B_total_RI = cell(num_ele, 1);

for ele = 1: num_ele
    B_total_RI{ele, 1} = B_RI;
    for j = 1: num_gp
        xi = GP(1, j);
        eta = GP(2, j);
        pN = [-(1/4)*(1-eta), (1/4)*(1-eta), (1/4)*(1+eta), -(1/4)*(1+eta);
            -(1/4)*(1-xi), -(1/4)*(1+xi), (1/4)*(1+xi), (1/4)*(1-xi)];
        pN_xy = [pN(1,:) * 2/l_ele_x; pN(2,:) * 2/l_ele_y];

        % FI: full integration
        B_1 = [pN_xy(1,1), 0; 0, pN_xy(2,1); pN_xy(2,1), 0; 0, pN_xy(1,1)];
        B_2 = [pN_xy(1,2), 0; 0, pN_xy(2,2); pN_xy(2,2), 0; 0, pN_xy(1,2)];
        B_3 = [pN_xy(1,3), 0; 0, pN_xy(2,3); pN_xy(2,3), 0; 0, pN_xy(1,3)];
        B_4 = [pN_xy(1,4), 0; 0, pN_xy(2,4); pN_xy(2,4), 0; 0, pN_xy(1,4)];
        B_total_FI{ele, j} = [B_1, B_2, B_3, B_4];
        
%         % SRI: Selective reduced integration
%         % K_d: FI
%         B_d_1 = [2/3 * pN_xy(1,1), -1/3 * pN_xy(2,1); -1/3 * pN_xy(1,1), 2/3 * pN_xy(2,1); -1/3 * pN_xy(1,1), -1/3 * pN_xy(2,1); pN_xy(2,1), pN_xy(1,1)];
%         B_d_2 = [2/3 * pN_xy(1,2), -1/3 * pN_xy(2,2); -1/3 * pN_xy(1,2), 2/3 * pN_xy(2,2); -1/3 * pN_xy(1,2), -1/3 * pN_xy(2,2); pN_xy(2,2), pN_xy(1,2)];
%         B_d_3 = [2/3 * pN_xy(1,3), -1/3 * pN_xy(2,3); -1/3 * pN_xy(1,3), 2/3 * pN_xy(2,3); -1/3 * pN_xy(1,3), -1/3 * pN_xy(2,3); pN_xy(2,3), pN_xy(1,3)];
%         B_d_4 = [2/3 * pN_xy(1,4), -1/3 * pN_xy(2,4); -1/3 * pN_xy(1,4), 2/3 * pN_xy(2,4); -1/3 * pN_xy(1,4), -1/3 * pN_xy(2,4); pN_xy(2,4), pN_xy(1,4)];
%         B_d = [B_d_1, B_d_2, B_d_3, B_d_4];
%         B_bar = B_v + B_d;
    end
end


% constuct a matrix with unknown nodes
known_BC = zeros(2*num_ele_x*(num_ele_y+1),1);

s = 0;
for j = 1: num_ele_y + 1
    for i = 2: num_ele_x + 1
        node = i + (num_ele_x + 1) * (j - 1);
        s = s + 1;
        known_BC(2 * s - 1, 1) = 2 * node - 1;
        known_BC(2 * s, 1) = 2 * node;
    end
end


% ------------------------------------------------------------------------


tipLoad = zeros(N_L+1, 1);
tipDeflection_FI = zeros(N_L+1, 1);
tipDeflection_RI = zeros(N_L+1, 1);
tipDeflection_SRI = zeros(N_L+1, 1);


for loadStep = 1: N_L
    
    u_RI = zeros(2 * num_node, 1);
    u_FI = zeros(2 * num_node, 1);
    u_SRI = zeros(2 * num_node, 1);
    Pn = dP * loadStep;
    f_ext = zeros(2 * num_node, 1);
    
    for i = 1: num_ele_y
        ele = i * num_ele_x;
        f_ele_ext = Pn / height * l_ele_y / 2 * [0; 0; 0; 1; 0; 1; 0; 0];
        for m = 1:4
            f_ext(2*ele_DOF(ele,m)-1:2*ele_DOF(ele,m),1) = f_ext(2*ele_DOF(ele,m)-1:2*ele_DOF(ele,m),1) + f_ele_ext(2*m-1:2*m,1);
        end
    end
    
    
% ------------------------------------------------------------------------


% Full Integration


    itr_FI = 0;
    a_FI = 1;

    while a_FI > tol
        
        itr_FI = itr_FI + 1;
        du = zeros(2 * num_node, 1);
        f_int = zeros(2 * num_node, 1);
        K = zeros(2 * num_node, 2 * num_node);
    
        for i = 1: num_ele
            d_ele = zeros(8,1);
            k_ele = zeros(8,8);
            f_ele_int = zeros(8,1);
            

            for j = 1: 4
                d_ele(2*j-1:2*j,1) = u_FI(2*ele_DOF(i,j)-1:2*ele_DOF(i,j),1);
            end
            
            % on each gauss point
            for l = 1: num_gp
                F = eye(3);
                B = B_total_FI{i,l};
                M = B * d_ele;
                F(1,1) = F(1,1) + M(1,1);
                F(1,2) = F(1,2) + M(3,1);
                F(2,1) = F(2,1) + M(4,1);
                F(2,2) = F(2,2) + M(2,1);
                [D, T, Eps] = SVK_MaterialModel(F, Cs);
                k_ele = k_ele + B' * (D + T) * B * A / 4 * W(l);
                f_ele_int = f_ele_int + B' * Eps * A / 4 * W(l);
            end
            
            
            for m = 1:4
                f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) = f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) + f_ele_int(2*m-1:2*m,1);
              
                for n = 1:4
                    K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) ...
                        = K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) + k_ele(2*m-1:2*m,2*n-1:2*n);
                end
            end
        end % end of element
        
        f = f_ext - f_int;

        % Construct a new stiffness matrix with known Essential B.C. applied

       
        K_reduced = zeros(size(known_BC,1));
        f_reduced = zeros(size(known_BC,1),1);

        for i = 1:size(known_BC,1)
            f_reduced(i,1) = f(known_BC(i,1),1) - K(known_BC(i,1),:) * du;
            for j = 1:size(known_BC,1)
                 K_reduced(i,j) = K(known_BC(i,1),known_BC(j,1));
            end
        end

        du_reduced = K_reduced \ f_reduced;
        for i = 1:size(known_BC,1)
            du(known_BC(i,1),1) = du_reduced(i,1);
        end
        u_FI = u_FI + du;
        
        a_FI = norm(du);
    end % end of iteration for FI
%     disp([loadStep, itr_FI]);
%     
%     
% ------------------------------------------------------------------------


% Reduced Integration

    
    itr_RI = 0;
    a_RI = 1; % estimation for convergence
    
    while a_RI > tol
        
        itr_RI = itr_RI + 1;
        du = zeros(2 * num_node, 1);
        f_int = zeros(2 * num_node, 1);
        K = zeros(2 * num_node, 2 * num_node);
    
        for i = 1: num_ele
            d_ele = zeros(8,1);

            for j = 1: 4
                d_ele(2*j-1:2*j,1) = u_RI(2*ele_DOF(i,j)-1:2*ele_DOF(i,j),1);
            end
            
            F = eye(3);

            B = B_total_RI{i, 1};
            M = B * d_ele;
            F(1,1) = F(1,1) + M(1,1);
            F(1,2) = F(1,2) + M(3,1);
            F(2,1) = F(2,1) + M(4,1);
            F(2,2) = F(2,2) + M(2,1);
            [D, T, Eps] = SVK_MaterialModel(F, Cs);
            k_ele = B' * (D + T) * B * A;
            f_ele_int = B' * Eps * A;


            % Assembly
            for m = 1:4
                f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) = f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) + f_ele_int(2*m-1:2*m,1);
              
                for n = 1:4
                    K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) ...
                        = K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) + k_ele(2*m-1:2*m,2*n-1:2*n);
                end
            end
        end % end of element
        
        f = f_ext - f_int;

        % Construct a new stiffness matrix with known Essential B.C. applied

       
        K_reduced = zeros(size(known_BC,1));
        f_reduced = zeros(size(known_BC,1),1);

        for i = 1:size(known_BC,1)
            f_reduced(i,1) = f(known_BC(i,1),1) - K(known_BC(i,1),:) * du;
            for j = 1:size(known_BC,1)
                 K_reduced(i,j) = K(known_BC(i,1),known_BC(j,1));
            end
        end

        du_reduced = K_reduced \ f_reduced;
        for i = 1:size(known_BC,1)
            du(known_BC(i,1),1) = du_reduced(i,1);
        end
        u_RI = u_RI + du;
        
        a_RI = norm(du);
    end % end of iteration for RI
%     disp([loadStep, itr_RI]);
    


% ------------------------------------------------------------------------


% Selective Reduced Integration

    itr_SRI = 0;
    a_SRI = 1;

    while a_SRI > tol
        
        itr_SRI = itr_SRI + 1;
        du = zeros(2 * num_node, 1);
        f_int = zeros(2 * num_node, 1);
        K = zeros(2 * num_node, 2 * num_node);
    
        for i = 1: num_ele
            d_ele = zeros(8,1);
            k_ele = zeros(8,8);
            k_ele_v = zeros(8,8);
            k_ele_d = zeros(8,8);
            f_ele_int_v = zeros(8,1);
            f_ele_int_d = zeros(8,1);
            

            for j = 1: 4
                d_ele(2*j-1:2*j,1) = u_SRI(2*ele_DOF(i,j)-1:2*ele_DOF(i,j),1);
            end
            
%           on each gauss point
            for l = 1: num_gp
                F = eye(3);
                B_d = B_total_FI{i,l};
                M = B_d * d_ele;
                F(1,1) = F(1,1) + M(1,1);
                F(1,2) = F(1,2) + M(3,1);
                F(2,1) = F(2,1) + M(4,1);
                F(2,2) = F(2,2) + M(2,1);
                [D, T, Eps] = SVK_MaterialModel(F, Cs_d);
                k_ele_d = k_ele_d + B_d' * (D + T) * B_d * A / 4 * W(l);
                f_ele_int_d = f_ele_int_d + B_d' * Eps * A / 4 * W(l);
            end
            
            F = eye(3);
            B_v = B_total_RI{i, 1};
            M = B_v * d_ele;
            F(1,1) = F(1,1) + M(1,1);
            F(1,2) = F(1,2) + M(3,1);
            F(2,1) = F(2,1) + M(4,1);
            F(2,2) = F(2,2) + M(2,1);
            [D, T, Eps] = SVK_MaterialModel(F, Cs_v);
            k_ele_v = B_v' * (D + T) * B_v * A;
            f_ele_int_v = B_v' * Eps * A;

            k_ele = k_ele_v + k_ele_d;
            f_ele_int = f_ele_int_d + f_ele_int_v;

%             Assembly
            for m = 1:4
                f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) = f_int(2*ele_DOF(i,m)-1:2*ele_DOF(i,m),1) + f_ele_int(2*m-1:2*m,1);
                for n = 1:4
                    K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) ...
                        = K(2*ele_DOF(i,m)-1:2*ele_DOF(i,m), 2*ele_DOF(i,n)-1:2*ele_DOF(i,n)) + k_ele(2*m-1:2*m,2*n-1:2*n);
                end
            end
        end % end of element
        
        f = f_ext - f_int;

%       Construct a new stiffness matrix with known Essential B.C. applied
        K_reduced = zeros(size(known_BC,1));
        f_reduced = zeros(size(known_BC,1),1);

        for i = 1:size(known_BC,1)
            f_reduced(i,1) = f(known_BC(i,1),1) - K(known_BC(i,1),:) * du;
            for j = 1:size(known_BC,1)
                 K_reduced(i,j) = K(known_BC(i,1),known_BC(j,1));
            end
        end

        du_reduced = K_reduced \ f_reduced;
        for i = 1:size(known_BC,1)
            du(known_BC(i,1),1) = du_reduced(i,1);
        end
        u_SRI = u_SRI + du;
        
        a_SRI = norm(du);
    end % end of iteration for SRI
%     disp([loadStep, itr_SRI])
    
% ------------------------------------------------------------------------

    disp([loadStep, itr_FI, itr_RI, itr_SRI])

    % save tip load
    tipLoad(loadStep+1,1) = Pn;
    tipDeflection_FI(loadStep+1,1) = u_FI((num_ele_y + 2) * (num_ele_x + 1), 1);
    tipDeflection_RI(loadStep+1,1) = u_RI((num_ele_y + 2) * (num_ele_x + 1), 1);
    tipDeflection_SRI(loadStep+1,1) = u_SRI((num_ele_y + 2) * (num_ele_x + 1), 1);
    
end  %end of load step




% ------------------------------------------------------------------------

% Step 5: Plotting

% Transform displacement vector to x-y coordinates
U_FI = zeros(num_node, 2);
U_RI = zeros(num_node, 2);
U_SRI = zeros(num_node, 2);
for i = 1:num_node
    U_FI(i, :) = u_FI(2*i-1:2*i, 1)';
    U_RI(i, :) = u_RI(2*i-1:2*i, 1)';
    U_SRI(i, :) = u_SRI(2*i-1:2*i, 1)';
end

% creat mesh plotting
% (deflections in true scale)
X0 = reshape(node_coords(:, 1), num_ele_x + 1, num_ele_y + 1);
Y0 = reshape(node_coords(:, 2), num_ele_x + 1, num_ele_y + 1);

X_RI = reshape(node_coords(:, 1) + U_RI(:, 1), num_ele_x + 1, num_ele_y + 1);
Y_RI = reshape(node_coords(:, 2) + U_RI(:, 2), num_ele_x + 1, num_ele_y + 1);

X_FI = reshape(node_coords(:, 1) + U_FI(:, 1), num_ele_x + 1, num_ele_y + 1);
Y_FI = reshape(node_coords(:, 2) + U_FI(:, 2), num_ele_x + 1, num_ele_y + 1);

X_SRI = reshape(node_coords(:, 1) + U_SRI(:, 1), num_ele_x + 1, num_ele_y + 1);
Y_SRI = reshape(node_coords(:, 2) + U_SRI(:, 2), num_ele_x + 1, num_ele_y + 1);


figure(1)
surf(X_FI, Y_FI, Y_FI-Y0, 'FaceColor', 'interp')
colormap jet;
view(2)
axis equal
grid on
axis([0, 60, -40, 20])
hold off
xlabel('x (in)', 'FontSize', 14)
ylabel('y (in)', 'FontSize', 14)
legend('deformed shape by FI', 'FontSize', 14)
title('Solution of deformation by Full Integration', 'FontSize', 14)

figure(2)
surf(X_RI, Y_RI, Y_RI-Y0, 'FaceColor', 'interp')
colormap jet;
view(2)
axis equal
grid on
axis([0, 60, -40, 20])
hold off
xlabel('x (in)', 'FontSize', 14)
ylabel('y (in)', 'FontSize', 14)
legend('deformed shape by RI', 'FontSize', 14)
title('Solution of deformation by Reduced Integration', 'FontSize', 14)

figure(3)
surf(X_SRI, Y_SRI, Y_SRI-Y0, 'FaceColor', 'interp')
colormap jet;
view(2)
axis equal
grid on
axis([0, 60, -40, 20])
hold off
xlabel('x (in)', 'FontSize', 14)
ylabel('y (in)', 'FontSize', 14)
legend('deformed shape by SRI', 'FontSize', 14)
title('Solution of deformation by Selective Reduced Integration', 'FontSize', 14)

% tip deflection vs. tip load
figure(4)
plot(Load, tipExact, 'k-')
hold on
plot(-tipLoad, -tipDeflection_FI, 'r-')
plot(-tipLoad, -tipDeflection_RI, 'g-')
plot(-tipLoad, -tipDeflection_SRI, 'b-')
hold off
grid on 
xlabel('Load (lb)', 'FontSize', 14)
ylabel('tip deflection (in)', 'FontSize', 14)
legend({'exact solution', 'Q4-FI', 'Q4-RI', 'Q4-SRI'}, 'FontSize', 14)
title('tip deflection in y axis', 'FontSize', 14)


% Re-organize the solutions in a matrix
Displacement = zeros(2*num_node, 3);

Displacement(:,1) = u_FI;
Displacement(:,2) = u_RI;
Displacement(:,3) = u_SRI;


% Displacement along y = 0
x = 0: l_ele_x: length;

u_y_0 = zeros(num_ele_x + 1, size(Displacement, 2));
for i = 1: size(Displacement, 2)
    for j = 2: num_ele_x + 1
        node = j + num_ele_y * (num_ele_x + 1) / 2;
        u_y_0(j,i) = Displacement(2 * node, i);
    end
end

figure(5)
plot(x, u_y_0(:,1), 'r.-')
hold on
plot(x, u_y_0(:,2), 'g.-')
plot(x, u_y_0(:,3), 'b.-')
hold off
grid on 
xlabel('x (in)', 'FontSize', 14)
ylabel('u_y (in)', 'FontSize', 14)
legend({'Q4-FI', 'Q4-RI', 'Q4-SRI'}, 'FontSize', 14)
title('u_y along y = 0', 'FontSize', 14)


% stresses along x = L/2

%FEM solutions
y = zeros(2*num_ele_y, 1);
for j = 1: num_ele_y
    y(2*j-1,1) = (j - 1) * l_ele_y - height / 2;
    y(2*j,1) = j * l_ele_y - height / 2;
end
stress_xx = zeros(2*num_ele_y,3);
stress_xy = zeros(2*num_ele_y,3);
stress_yy = zeros(2*num_ele_y,3);

for i = 1: size(Displacement, 2)
    for j = 1: num_ele_y
        ele_left = (j - 1) * num_ele_x + num_ele_x / 2;
        d_left_ele = zeros(8,1);
        ele_right = ele_left + 1;
        d_right_ele = zeros(8,1);
        for m = 1: 4
            d_left_ele(2*m-1:2*m,1) = Displacement(2*ele_DOF(ele_left, m)-1:2*ele_DOF(ele_left, m), i);
            d_right_ele(2*m-1:2*m,1) = Displacement(2*ele_DOF(ele_right, m)-1:2*ele_DOF(ele_right, m), i);
        end
        
        % Use the SVK material model to calculate the 1st PK stresses
        F_left = eye(3);
        M_left = B_total_RI{ele_left, 1} * d_left_ele;
        F_left(1,1) = F_left(1,1) + M_left(1,1);
        F_left(1,2) = F_left(1,2) + M_left(3,1);
        F_left(2,1) = F_left(2,1) + M_left(4,1);
        F_left(2,2) = F_left(2,2) + M_left(2,1);
        
        F_right = eye(3);
        M_right = B_total_RI{ele_right, 1} * d_right_ele;
        F_right(1,1) = F_right(1,1) + M_right(1,1);
        F_right(1,2) = F_right(1,2) + M_right(3,1);
        F_right(2,1) = F_right(2,1) + M_right(4,1);
        F_right(2,2) = F_right(2,2) + M_right(2,1);
        
        [~, ~, Eps_left] = SVK_MaterialModel(F_left,Cs);
        [~, ~, Eps_right] = SVK_MaterialModel(F_right,Cs);
        stress = (Eps_left + Eps_right) / 2;
        stress_xx(2*j-1,i) = stress(1,1); % stress_11
        stress_xx(2*j,i) = stress(1,1);
        stress_yy(2*j-1,i) = stress(2,1); % stress_11
        stress_yy(2*j,i) = stress(2,1);
        stress_xy(2*j-1,i) = stress(4,1); % stress_12
        stress_xy(2*j,i) = stress(4,1);
    end
end

figure(6) % stress_xx
plot(y, stress_xx(:,1), 'r.-')
hold on
plot(y, stress_xx(:,2), 'g.-')
plot(y, stress_xx(:,3), 'b.-')
hold off
grid on 
xlabel('y (in)', 'FontSize', 14)
ylabel('{\sigma}_{xx} (psi)', 'FontSize', 14)
legend({'Q4-FI', 'Q4-RI', 'Q4-SRI'}, 'FontSize', 14)
title('{\sigma}_{xx} along x = L/2', 'FontSize', 14)

figure(7) %stress_xy
plot(y, stress_xy(:,1), 'r.-')
hold on
plot(y, stress_xy(:,2), 'g.-')
plot(y, stress_xy(:,3), 'b.-')
hold off
grid on 
xlabel('y (in)', 'FontSize', 14)
ylabel('{\sigma}_{xy} (psi)', 'FontSize', 14)
legend({'Q4-FI', 'Q4-RI', 'Q4-SRI'}, 'FontSize', 14)
title('{\sigma}_{xy} along x = L/2', 'FontSize', 14)

% figure(8) %stress_yy
% plot(y, stress_yy(:,1), 'r.-')
% hold on
% plot(y, stress_yy(:,2), 'g.-')
% plot(y, stress_yy(:,3), 'b.-')
% hold off
% grid on 
% xlabel('y (in)', 'FontSize', 14)
% ylabel('{\sigma}_{yy} (psi)', 'FontSize', 14)
% legend({'Q4-FI', 'Q4-RI', 'Q4-SRI'}, 'FontSize', 14)
% title('{\sigma}_{yy} along x = L/2', 'FontSize', 14)
% 

 
 
function [D, T, Eps] = SVK_MaterialModel(F, Cs)
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
        for j = 1:4
            D(i,j) = D_tensor(index(i,1), index(i,2), index(j,1), index(j,2));
        end
    end

    Eps(1,1) = stress(1,1);
    Eps(2,1) = stress(2,2);
    Eps(3,1) = stress(2,1);
    Eps(4,1) = stress(1,2);
    
    T(1,1) = S(1,1);
    T(1,3) = S(1,2);
    T(2,2) = S(2,2);
    T(2,4) = S(2,1);
    T(3,1) = S(2,1);
    T(3,3) = S(2,2);
    T(4,2) = S(1,2);
    T(4,4) = S(1,1);
end









