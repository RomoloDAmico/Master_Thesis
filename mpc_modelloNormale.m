%% ======================= MPC MODELLO NORMALE ============================
clc; clear all; close all;

%% ======================== PARAMETRI FISICI =============================
global g m1 m2 l1 l2 v1 v2
g  = 9.81;
l1 = 1.0; l2 = 0.7;
m1 = 2.0; m2 = 1.5;
v1 = 6.0; v2 = 3.0;
nx = 4; nu = 2;

%% ====================== STATO INIZIALE E RIFERIMENTO ===================
x0   = [0; 0; 0; 0];
xref = [pi/4; pi/6; 0; 0];

%% === CALCOLO COPPIA DI EQUILIBRIO (GRAVITÀ) ===
% Calcoliamo u_eq tale che il robot stia fermo a xref
% Estraggo la gravità G(xref) dalla funzione dynamics
% Trucco: chiamo dynamics con accelerazioni e velocità nulle
% M*ddq + C*dq + G = u  --> se ddq=0, dq=0, allora u = G
fake_x = xref; fake_x(3:4) = 0; % Velocità zero
% Poiché dynamics calcola ddq, facciamo il calcolo manuale di G qui per precisione:
G1_ref = (m1+m2)*g*l1*sin(xref(1)) + m2*g*l2*sin(xref(1)+xref(2));
G2_ref = m2*g*l2*sin(xref(1)+xref(2));
u_eq = [G1_ref; G2_ref];

fprintf('Coppia necessaria per stare fermi al target: u1=%.2f, u2=%.2f\n', u_eq(1), u_eq(2));

%% ======== LINEARIZZAZIONE AUTOMATICA DEL MODELLO NON LINEARE ===========
f = @(x,u) dynamicslink(0,x,u);
x_star = xref;
u_star = u_eq; % <--- LINEARIZZIAMO ATTORNO ALL'EQUILIBRIO REALE, NON ZERO
eps_num = 1e-5;
A_c = zeros(nx);
B_c = zeros(nx, nu);

% Jacobiano rispetto allo stato (A)
for i = 1:nx
    dx = zeros(nx,1); dx(i)=eps_num;
    fp = f(x_star+dx,u_star);
    fm = f(x_star-dx,u_star);
    A_c(:,i) = (fp - fm)/(2*eps_num);
end

% Jacobiano rispetto all'input (B)
for j = 1:nu
    du = zeros(nu,1); du(j)=eps_num;
    fp = f(x_star,u_star+du);
    fm = f(x_star,u_star-du);
    B_c(:,j) = (fp - fm)/(2*eps_num);
end

%% ======================== DISCRETIZZAZIONE =============================
dt = 0.01;
A_d = eye(nx) + dt*A_c;
B_d = dt * B_c;

% Calcoliamo il termine affine (costante) per compensare la linearizzazione
% Il sistema è: x_k+1 = A*x_k + B*u_k + affine_term
% All'equilibrio: xref = A*xref + B*u_eq + affine_term
% Quindi: affine_term = xref - A*xref - B*u_eq
affine_term = xref - A_d*xref - B_d*u_eq;

%% ========================= COSTRUZIONE MPC =============================
Np = 20;   % Orizzonte aumentato per fluidità
Q = diag([500, 500, 10, 10]); % Più peso alla posizione
R = 0.01 * eye(nu); % Costo controllo basso per permettere sforzo
Hu = [eye(nu); -eye(nu)];
bu = 30 * ones(2*nu,1); % Limiti saturazione

u = sdpvar(nu, Np);
x = sdpvar(nx, Np+1);
x_init = sdpvar(nx,1);
x_target = sdpvar(nx,1); % Riferimento variabile nel solver

constraints = [];
objective = 0;

constraints = [constraints, x(:,1) == x_init];

for k = 1:Np
    % Costo: penalizziamo deviazione dal target e deviazione dalla coppia di equilibrio
    objective = objective + (x(:,k)-x_target)'*Q*(x(:,k)-x_target) + ...
                (u(:,k)-u_eq)'*R*(u(:,k)-u_eq);
    
    % Dinamica Linearizzata con termine AFFINE
    constraints = [constraints, x(:,k+1) == A_d*x(:,k) + B_d*u(:,k) + affine_term];
    
    % Vincoli input
    constraints = [constraints, Hu*u(:,k) <= bu];
end

ops = sdpsettings('solver','gurobi','verbose',0);
mpc_ctrl = optimizer(constraints, objective, ops, {x_init, x_target}, u(:,1));

%% ===================== CLOSED-LOOP SIMULATION ==========================
Tfinal = 3.0;
time = 0:dt:Tfinal;
xh = x0;
uh = [];

fprintf('Avvio simulazione MPC...\n');
for k = 1:length(time)
    % Controllo MPC
    [u_curr, errorcode] = mpc_ctrl(xh(:,end), xref);
    
    if errorcode ~= 0
        warning('Gurobi non ha trovato soluzione!');
        u_curr = zeros(2,1);
    end
    
    uh = [uh, u_curr];
    
    % Dinamica non lineare reale
    [~, xx] = ode45(@(tt,xx) dynamicslink(tt,xx,u_curr), [0 dt], xh(:,end));
    xh = [xh, xx(end,:)'];
end

%% ============================ PLOT RISULTATI ============================
figure('Color','w'); sgtitle("MPC Basato su Modello Fisico (Con Feedforward)");
subplot(4,1,1)
plot(time, rad2deg(xh(1,1:end-1)),'LineWidth',2); hold on;
yline(rad2deg(xref(1)), '--r', 'Target');
ylabel('\theta_1 [deg]'); grid on;
ylim([-10 60]);

subplot(4,1,2)
plot(time, rad2deg(xh(2,1:end-1)),'LineWidth',2); hold on;
yline(rad2deg(xref(2)), '--r', 'Target');
ylabel('\theta_2 [deg]'); grid on;
ylim([-10 50]);

subplot(4,1,3)
plot(time, xh(3,1:end-1),'LineWidth',2); hold on;
plot(time, xh(4,1:end-1),'LineWidth',2);
ylabel('vel [rad/s]'); grid on; legend('\omega_1','\omega_2');

subplot(4,1,4)
stairs(time, uh(1,:), 'LineWidth',2); hold on;
stairs(time, uh(2,:), 'LineWidth',2);
yline(u_eq(1), ':b', 'u_{eq,1}'); 
yline(u_eq(2), ':r', 'u_{eq,2}');
xlabel('time [s]'); ylabel('u [Nm]'); grid on; legend('u_1','u_2');
ylim([-30 30]);

%% ===================== DINAMICA DOUBLE LINK ============================
function dx = dynamicslink(~, x, u)
    global g l1 l2 m1 m2 v1 v2
    q1 = x(1); q2 = x(2);
    dq1 = x(3); dq2 = x(4);
    
    % Matrice inerzia
    M11 = (m1+m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*cos(q2);
    M12 = m2*l2^2 + m2*l1*l2*cos(q2);
    M = [M11 M12; M12 m2*l2^2];
    
    % Coriolis
    h = -m2*l1*l2*sin(q2);
    C = [h*dq2*(2*dq1 + dq2); -h*dq1^2]; % Forma corretta
    
    % Gravità
    G1 = (m1+m2)*g*l1*sin(q1) + m2*g*l2*sin(q1+q2);
    G2 = m2*g*l2*sin(q1+q2);
    G = [G1; G2];
    
    % Attrito
    Ddq = [v1*dq1; v2*dq2];
    
    ddq = M \ (u - C - G - Ddq);
    dx = [dq1; dq2; ddq];
end