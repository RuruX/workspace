clear all;
close all;
clc;

Lx = 25;
Ly = 25;

tot_angle = pi/2;

Nelx=5;
Nely=10;
Nel = Nelx*Nely;

dx = Lx/Nelx;
% dy = Ly/Nely;
alpha = tot_angle/Nely;

Nnodes = (Nelx+1)*(Nely+1);
Nnodesx = Nelx+1;
Nnodesy = Nely+1;

x_vec = zeros(Nnodes,1);
y_vec = zeros(Nnodes,1);
z_vec = zeros(Nnodes,1);
xyz_mat = zeros(Nnodes,3);

vertex_ID = zeros(Nnodes,1);

M_quad = zeros(Nel,4);
M_tri = zeros(2*Nel,3);

dy_old=0;

node_number = 1;
for i = 1:Nely+1
    for j = 1:Nelx+1
    
    curr_angle = (i-1)*alpha;
    bi = Ly*cos(curr_angle);
        
    x_vec(node_number,1) = (j-1)*dx;                     %x coordinate
    y_vec(node_number,1) = Ly-bi;                        %y coordinate
    z_vec(node_number,1) = sqrt(2*(Ly-bi)*Ly-(Ly-bi)^2); %z coordinate
    
    xyz_mat(node_number,1) = (j-1)*dx;
    xyz_mat(node_number,2) = Ly-bi;
    xyz_mat(node_number,3) = sqrt(2*(Ly-bi)*Ly-(Ly-bi)^2);
    
    vertex_ID(node_number,1) = node_number;
    
    node_number = node_number+1;
    
    end

end

el_number = 1;
for ii = 1:Nely
    for jj = 1:Nelx
        
        M_quad(el_number,1)=(ii-1)*Nnodesx+jj;
        M_quad(el_number,2)=(ii-1)*Nnodesx+jj+1;
        M_quad(el_number,3)=(ii)*Nnodesx+jj+1;
        M_quad(el_number,4)=(ii)*Nnodesx+jj;
        
        
        M_tri(2*el_number-1,1)=(ii-1)*Nnodesx+jj;
        M_tri(2*el_number-1,2)=(ii-1)*Nnodesx+jj+1;
        M_tri(2*el_number-1,3)=(ii)*Nnodesx+jj;
        M_tri(2*el_number,1)=(ii-1)*Nnodesx+jj+1;
        M_tri(2*el_number,2)=(ii)*Nnodesx+jj+1;
        M_tri(2*el_number,3)=(ii)*Nnodesx+jj;
        
        
        el_number=el_number+1;
        
    end
end

M_quad_touse = M_quad-1;
M_tri_touse = M_tri-1;

figure
scatter3(x_vec,y_vec,z_vec)