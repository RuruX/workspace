%define the parameters
E = 1000;
R_i = 6;
R_o = 9;
v = .3;
rho = 0.01;
C = E / ((1 + v) * (1 - 2 * v)) * [1 - v, v; v, 1 - v];
T = 0.05;        % period T for internal pressure P.
T_end = 0.05;    % time to the end.

num_ele_array = [81];

for p = 1: length(num_ele_array)
    %for 1D 2-node linear approximation
    num_ele = num_ele_array(p);
    num_node = num_ele + 1; 
    ele_length = (R_o - R_i) / num_ele;

    %initialize the mass matrix, stiffness matrix, displacement vector and force vector.
    M = zeros(num_node);
    K = zeros(num_node);
    U = zeros(num_node, 1);
    F = zeros(num_node, 1);

    %element DOF matrix for 2-node element
    ele_DOF = zeros(num_ele, 2);

    for i = 1:num_ele
        ele_DOF(i,:) = [i, i + 1];
    end

    %Gauss integration of order 2
    n_XI = 2;
    XI_GF = [-1 / sqrt(3), 1 / sqrt(3)];
    w_XI = [1, 1];

    B_mid = zeros(2, 2*num_ele);
    %Calculate the stiffness matrix and the mass matrix
    for i = 1 : num_ele
        r_1 = R_i + (i - 1) * ele_length;
        r_2 = R_i + i * ele_length;
        r_ele = [r_1; r_2];
        %initialize the local stiffness and force matrices
        k_ele = zeros(2);
        m_ele = zeros(2);

        r_mid = (r_1 + r_2) / 2;
        B_mid(:,(2*i-1):2*i) = 1/ele_length*[-1, 1; -1+r_2/r_mid, 1-r_1/r_mid];

        for j = 1 : n_XI
            xi = XI_GF(j);
            w_xi = w_XI(j);

            %formulate shape function N and gradient matrix B
            N = [1/2 * (1 - xi), 1/2 * (xi + 1)]; 
            r_xi = N * r_ele;
            B = [-1/ele_length, 1/ele_length; N/r_xi];

            k_ele = k_ele + ele_length/2 * (B') * C * B * r_xi * w_xi;
            m_ele = m_ele + ele_length/2 * rho * (N') * N * r_xi * w_xi;
        end

        %global assembly
        for m = 1:2
            for n = 1:2
                M(ele_DOF(i,m), ele_DOF(i,n)) = M(ele_DOF(i,m), ...
                    ele_DOF(i,n)) + m_ele(m,n);
                K(ele_DOF(i,m), ele_DOF(i,n)) = K(ele_DOF(i,m), ...
                    ele_DOF(i,n)) + k_ele(m,n);
            end
        end
    end

    % Here Row Sum method is used to obtain lumped mass.
    M_lumped = zeros(num_node);
    for k = 1:num_node
        M_lumped(k, k) = sum(M(k, :));
    end

    lambda = max(eig(K, M_lumped));


    beta = 0;
    gamma = .5;


    t_cr = sqrt(lambda) \ 2;

    delta_t = 1e-5;

    nt = ceil(delta_t \ T_end);

    % initialize d, v and a
    d = zeros(num_node, 1);
    v = zeros(num_node, 1);
    a = zeros(num_node, 1);
    U_d = zeros(num_node, nt + 1);
    U_v = zeros(num_node, nt + 1);
    U_a = zeros(num_node, nt + 1);

    A = M_lumped + beta * delta_t ^ 2 * K;

    stress_fem = zeros(num_ele, 2);
    radial_stress = zeros(num_ele, nt + 1);
    circum_stress = zeros(num_ele, nt + 1);

    for n = 1: nt
        time = n * delta_t;

        % predictor phase
        d_p = d + delta_t * v + delta_t^2 / 2 * (1 - 2 * beta) * a;
        v_p = v + (1 - gamma) * delta_t * a;

        % solution phase
        if time - T * floor(time/T) - T / 2 <= 1e-12
            P = sin(2 * pi * time / T);
        else
            P = 0;
        end
        f = R_i * P;
        F(1, 1) = f;
        F_star = F - K * d_p;
        a = A \ F_star;

        % correction phase
        d = d_p + beta * delta_t ^ 2 * a;
        v = v_p + gamma * delta_t * a;

        %store value
        U_d(:, n + 1) = d;
        U_v(:, n + 1) = v;
        U_a(:, n + 1) = a;

        for ele = 1:num_ele
            stress_fem(ele, :) = (C * B_mid(:, (2*ele-1):2*ele) * d(ele:(ele+1),1))';
        end

        radial_stress(:, n + 1) = stress_fem(:, 1);
        circum_stress(:, n + 1) = stress_fem(:, 2);


    end
    time_array = linspace(0, T_end, nt + 1);
    radius_array = R_i : ele_length : R_o;


    figure(1)
    subplot(3,1,1)
    plot(time_array, radial_stress(1, :), 'r')
    hold on
    plot(time_array, radial_stress((num_ele+1)/2, :), 'b')
    plot(time_array, radial_stress(num_ele, :), 'g')
    hold off
    legend('inner surface r = R_i', 'mid-surface(r = (R_i+R_o)/2', 'outer surface r = R_o', 'Location', 'southeast')
    xlabel('time (s)')
    ylabel('radial stress \sigma_{rr} (psi)')
    title('Radial Stress by Central Diffenrence Method')
    
    subplot(3,1,2)
    plot(time_array, circum_stress(1, :), 'r')
    hold on
    plot(time_array, circum_stress((num_ele+1)/2, :), 'b')
    plot(time_array, circum_stress(num_ele, :), 'g')
    hold off
    legend('inner surface r = R_i', 'mid-surface(r = (R_i+R_o)/2', 'outer surface r = R_o', 'Location', 'southeast')
    xlabel('time (s)')
    ylabel('circumferential stress \sigma_{\theta\theta} (psi)')
    title('Circumferential Stress by Central Diffenrence Method')
    
    subplot(3,1,3)
    plot(time_array, U_d(1, :), 'r')
    hold on
    plot(time_array, (U_d((num_ele+1)/2, :) + U_d((num_ele+1)/2+1, :))/2, 'b')
    plot(time_array, U_d(num_ele+1, :), 'g')
    hold off
    legend('inner surface r = R_i', 'mid-surface(r = (R_i+R_o)/2', 'outer surface r = R_o', 'Location', 'southeast')
    xlabel('time (s)')
    ylabel('Displacement d (in)')
    title('Displacements by Central Diffenrence Method')

    
%     % For study in the effect of element refinement on the accuracy
% 
%     figure(p)
%     subplot(3,1,1)
%     plot(radius_array(1:num_ele), radial_stress(:, 1), 'r+')
%     hold on
%     plot(radius_array(1:num_ele), radial_stress(:, round(nt/4)), 'b+')
%     plot(radius_array(1:num_ele), radial_stress(:, round(nt/2)), 'g+')
%     plot(radius_array(1:num_ele), radial_stress(:, round(3*nt/4)), 'm+')
%     plot(radius_array(1:num_ele), radial_stress(:, nt), 'k+')
%     
%     plot(radius_401(1:401), radial_stress_401(:, 1), 'r--')
%     plot(radius_401(1:401), radial_stress_401(:, round(nt/4)), 'b--')
%     plot(radius_401(1:401), radial_stress_401(:, round(nt/2)), 'g--')
%     plot(radius_401(1:401), radial_stress_401(:, round(3*nt/4)), 'm--')
%     plot(radius_401(1:401), radial_stress_401(:, nt), 'k--')
%     
%     hold off
%     legend('t = 0', 't = T/4', 't = T/2', 't = 3T/4', 't = T')
%     xlabel('radius (in)')
%     ylabel('radial stress \sigma_{rr} (psi)')
%     title('Radial stress distribution by Central Diffenrence Method')
% 
%     subplot(3,1,2)
%     plot(radius_array(1:num_ele), circum_stress(:, 1), 'r+')
%     hold on
%     plot(radius_array(1:num_ele), circum_stress(:, round(nt/4)), 'b+')
%     plot(radius_array(1:num_ele), circum_stress(:, round(nt/2)), 'g+')
%     plot(radius_array(1:num_ele), circum_stress(:, round(3*nt/4)), 'm+')
%     plot(radius_array(1:num_ele), circum_stress(:, nt), 'k+')
%     
%     plot(radius_401(1:401), circum_stress_401(:, 1), 'r--')
%     plot(radius_401(1:401), circum_stress_401(:, round(nt/4)), 'b--')
%     plot(radius_401(1:401), circum_stress_401(:, round(nt/2)), 'g--')
%     plot(radius_401(1:401), circum_stress_401(:, round(3*nt/4)), 'm--')
%     plot(radius_401(1:401), circum_stress_401(:, nt), 'k--')
%     hold off
%     legend('t = 0', 't = T/4', 't = T/2', 't = 3T/4', 't = T')
%     xlabel('radius (in)')
%     ylabel('circumferential stress \sigma_{\theta\theta} (psi)')
%     title('Circumferential Stress distribution by Central Diffenrence Method')
% 
% 
%     subplot(3,1,3)
%     plot(radius_array, U_d(:, 1), 'r+')
%     hold on
%     plot(radius_array, U_d(:, round(nt/4)), 'b+')
%     plot(radius_array, U_d(:, round(nt/2)), 'g+')
%     plot(radius_array, U_d(:, round(3*nt/4)), 'm+')
%     plot(radius_array, U_d(:, nt), 'k+')
%     
%     plot(radius_401(1:402), U_d_401(:, 1), 'r--')
%     plot(radius_401(1:402), U_d_401(:, round(nt/4)), 'b--')
%     plot(radius_401(1:402), U_d_401(:, round(nt/2)), 'g--')
%     plot(radius_401(1:402), U_d_401(:, round(3*nt/4)), 'm--')
%     plot(radius_401(1:402), U_d_401(:, nt), 'k--')
%     hold off
%     legend('t = 0', 't = T/4', 't = T/2', 't = 3T/4', 't = T')
%     xlabel('radius (in)')
%     ylabel('Displacement d (in)')
%     title('Displacement distribution by Central Diffenrence Method')

end






