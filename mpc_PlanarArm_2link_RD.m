%% ======================= MODEL PREDICTIVE CONTROL =======================
% ============================= DOUBLE LINK ============================= 

clc; clear all; close all; % pulizia ambiente e grafici

%set(0,'DefaultFigureColor','w');
%set(0,'DefaultAxesColor','w');
%set(0,'DefaultFigureInvertHardcopy','off');

%% DMD matrices (devono essere del double-link)
if ~isfile('dynamics_reduced_dim.mat')
    error('File dynamics_reduced_dim.mat non trovato nella cartella corrente');
end

load('dynamics_reduced_dim.mat'); % carica le matrici ridotte di prima
% Uhat - base ridotta (1000 x 30)
% approxA, approxB - modello lineare ridotto (A, B)
% Load the DMD matrices for the model predictive control
% Uhat = load('Uhat_matrix.mat'); Assuming Uhat is stored in a .mat file
% approxA = load('approxA_matrix.mat'); Load the approximate A matrix
% approxB = load('approxB_matrix.mat'); Load the approximate B matrix

%% Physical parameters (double pendulum)
global g l1 l2 m1 m2 v1 v2 height width n_states n_controls xref

height = 100;
width = 100;

g = 9.81;           % [m/s^2]
l1 = 1.0; l2 = 0.7  % [m]
m1 = 2.0; m2 = 1.5  % [kg]
v1 = 6.0; v2 = 3.0  % [kgms]

%% x_ini and x_ref
x_initial = [0; 0; 0; 0];     % angolo e velocità iniziale 
xref = [pi/4; pi/6; 0; 0];    % angolo e velocità desiderati

% --- reference angolari (in rad e in deg per comodità)
theta1_ref = xref(1);
theta2_ref = xref(2);
theta1_ref_deg = rad2deg(theta1_ref);
theta2_ref_deg = rad2deg(theta2_ref);

%% Image-based DMD initialization (if images missing, create placeholders)
x_ini_path = './x_ini_double_link.png';
x_ref_path = './x_ref_double_link.png';

%if ~isfile(x_ini_path)
    % crea immagine di inizio dal vero stato (salva current _img.png)
 %   img_generation(x_initial);
  %  movefile('./current_img.png', x_ini_path);
%end

%if ~isfile(x_ref_path)
 %   img_generation(xref);
  %  movefile('./current_img.png', x_ref_path);
%end

x_ini = img2array(x_ini_path); % 100x100
x_ref = img2array(x_ref_path); % 100x100

% proietta nello spazio ridotto (controlla dimensioni compatibili)
if size(Uhat, 1) ~= numel(x_ini)
    warning('La dimensione di Uhat (%d) non coincide con il numel dell''immagine (%d). Controlla le matrici DMD.', size(Uhat, 1), numel(x_ini));
end

% stato ridotto
x_ini = Uhat'*x_ini(:); % 30x1
x_ref = Uhat'*x_ref(:); % 30x1
% ciascuna immagine (100x100) viene convertita in scala di grigi e
% normalizzata. Poi viene proiettata nello spazio ridotto DMD

%% Hyperparameters MPC
A = approxA;
B = approxB;

nx = size(A,2); % dimensione stato ridotto (30)
nu = size(B,2); % dimensione input ridotto (attesa 2 per double-link controllato su 2 giunti)

Q = 100 * eye(nx); 
R = 10 * eye(nu);  

Np = 10; % era 5

% input constraints
Hu = [eye(nu); -eye(nu)];
bu = 20 * ones(2*nu,1); % -5 <= u <= 5, PRIMA ERA 5

%% Optimizer: YALMIP con GUROBI come solver
% variable initialization
u_var = sdpvar(nu, Np);
x_var = sdpvar(nx, Np+1);
x0_var = sdpvar(nx, 1);
xr_var = sdpvar(nx, 1);

constraints = [x_var(:,1) == x0_var];  % constraints = []
objective = 0;

% initial state constraint
constraints = [constraints, x_var(:,1)==x0_var];

for k = 1:Np
    % stage cost
    objective = objective + (x_var(:,k)-xr_var)'*Q*(x_var(:,k)-xr_var) + u_var(:,k)'*R*u_var(:,k);
    
    % state propagation constraints
    constraints = [constraints, x_var(:,k+1) == A*x_var(:,k) + B*u_var(:,k)];
    
    % state and input constraints
    % constraints = [constraints, Hx*x(:,k+1)<=bx];
    constraints = [constraints, Hu*u_var(:,k) <= bu];
end

% terminal cost
% objective = objective + (x(:,Np+1)-xr)'*Q*(x(:,Np+1)-xr);

% optimizer
% try gurobi, else fall back to quadprog
ops = sdpsettings('verbose', 2, 'solver', 'gurobi', 'gurobi.Presolve', 0); %,'gurobi.DualReduction',0);
try
    predictor = optimizer(constraints, objective, ops, {x0_var, xr_var}, {x_var, u_var}); % creazione oggetto ottimizzatore
    solver_used = 'gurobi';

catch
    warning('Gurobi non disponibile o errore: uso quadprog come fallback');
    ops = sdpsettings('verbose', 0, 'solver', 'quadprog');
    predictor = optimizer(constraints, objective, ops, {x0_var, xr_var}, {x_var, u_var});
    solver_used = 'quadprog';
end

fprintf('MPC optimizer creato (solver: %s). nx = %d nu = %d\n', solver_used, nx, nu);

%% Closed-Loop
dt = 0.1; % prima usavo tspan
t_final = 20; % era 10 prima
time_vec = 0:dt:(t_final - dt);

xh = x_initial; % 2x1
x_dmd = x_ini;  % 30x1
uh = []; % storico inputs (nu x N)

filename = 'double_link_animation.gif';

%% Closed-Loop simulation
% ...
% disp(['Dimensione dell''input nu: ', num2str(nu)]); 
% if nu ~= 1
    % error('L''input di controllo u per la dinamica è scalare, ma nu è diverso da 1.');
% end
% ...

for k = 1:length(time_vec)-1
    % disp(k)
    % solve MPC on reduced space
    sol = predictor(x_dmd(:,end), x_ref); % 30x1
    x_pred = sol{1}; % predicted states (not used directly here)
    u_seq = sol{2};  % sequence of inputs (nu x Np)
    u_now = u_seq(:,1);

    
    % Ensure u_now has length 2 for dynamics: if not, pad with zeros
    %if length(u_now) < 2
     %   u_applied = [u_now; zeros(2 - length(u_now), 1)];
    %else
    %    u_applied = u_now(1:2);
    %end
    
    uh = [uh, u_now];
    % disp(uh(:,end));

    % disp(size(xh(:,end)));
    % disp(uh(:,end));

    % --- integrate true nonlinear double-pendulum for dt seconds
    % initial conditions is last column of xh
    x0_true = xh(:, end);

    % Dynamics
    [t_int, x_traj] = ode45(@(t,x) double_dynamics(t,x,u_now), [0 dt], x0_true);
    % [t,x] = ode45(@(t,x) dynamics(t,x,uh(:,end)), [0 dt], xh(:,end));
    x_next = x_traj(end,:)';
    xh = [xh, x_next]; % 4x1, aggiorna lo stato finale

    % Generate and save image (for DMD projection and GIF)
    cla;
    img_path = img_generation(x_next);
    
    %frame = getframe(gcf); % Added
    %im = frame2im(frame); % Added
    %[imind, cm] = rgb2ind(im, 256); % Added
    %if k == 1 % Added
        %imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', dt);
        % drawnow;
    %else
        %imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
        % drawnow;
    %end % Added
   %drawnow;

    % --- update DMD reduced-state by projecting the saved image
    x_img = img2array(img_path); %100x100
    x_dmd = [x_dmd, Uhat' * x_img(:)]; %30x1
   
end

fprintf('Simulazione chiusa. Ultimo stato xh:\n');
disp(xh(:,end))

% === Plot risultati closed-loop ===
%figure;
%subplot(2,1,1);
%plot(time_vec, xh(1,:), 'r', time_vec, xh(2,:), 'b');
%xlabel('Tempo [s]');
%ylabel('\theta_1, \theta_2 [rad]');
%title('Evoluzione degli angoli');

%subplot(2,1,2);
% safety check: esiste uh e ha due righe?
%if exist('uh','var') && size(uh,1) >= 2
%    plot(time_vec(1:end-1), uh(1,:), 'r', time_vec(1:end-1), uh(2,:), 'b');
%else
%    warning('Variabile "uh" non disponibile o dimensioni non corrette: salta il plot dei controlli');
%end
%xlabel('Tempo [s]');
%ylabel('Coppie [Nm]');
%title('Controlli MPC');

% Salvataggio automatico
%saveas(gcf, fullfile('dataset', 'Simulazioni', 'risultati_MPC.png'));

% === Plot risultati closed-loop ===
%figure('Color','w');
%tN = 0:dt:dt*(size(xh,2)-1);  % asse temporale per xh

% Angoli (in rad) con reference
%subplot(2,1,1);
%plot(tN, xh(1,:), 'r', 'LineWidth', 1.5); hold on;       % theta1
%plot(tN, xh(2,:), 'b', 'LineWidth', 1.5);                % theta2
%yline(theta1_ref, '--r', 'LineWidth', 1.2);
%yline(theta2_ref, '--b', 'LineWidth', 1.2);
%xlabel('Tempo [s]');
%ylabel('\theta [rad]');
%title('Evoluzione degli angoli (con reference)');
%legend('\theta_1','\theta_2','\theta_{1,ref}','\theta_{2,ref}','Location','best');
%grid on;

% Controlli
%subplot(2,1,2);
%if exist('uh','var') && ~isempty(uh) && size(uh,1) >= 2
    %tU = 0:dt:dt*(size(uh,2)-1);
    %stairs(tU, uh(1,:), 'r', 'LineWidth', 1.5); hold on;
    %stairs(tU, uh(2,:), 'b', 'LineWidth', 1.5);
    %xlabel('Tempo [s]');
    %ylabel('Coppie [Nm]');
    %title('Controlli MPC');
    %legend('u_1','u_2','Location','best');
    %grid on;
%else
    %warning('Variabile "uh" non disponibile o dimensioni non corrette: salto il plot dei controlli');
%end

% Salvataggio automatico
%saveas(gcf, fullfile('dataset', 'Simulazioni', 'risultati_MPC.png'));

%save('results_MPC_true.mat', 'xh', 'uh', 'dt', 'xref', 'theta1_ref', 'theta2_ref');


%save('cl_data.mat','xh','uh');


% %% Closed Loop
% t = 0;
% x_dmd = x_ini; %X, Y,v,psi
% 
% ref_dmd = x_ref;
% Ts_s=0.1;
% 
% x_current = x0;
% u = 0;
% 
% %historian variables
% th=[t];
% xh=[x_current];
% uh=[u];
% 
% 
% simtime = 10;
%     tic;
% for i=1:(simtime/Ts_s)
%     disp(i)
%     opti.set_value(P, ref_dmd);
%     opti.set_value(X0, x_dmd);
%     sol = opti.solve();
%     uout=sol.value(U);
%     u=uout(:,1);
%     % integra con RK4 per 0.1 s
%     x_current = rk4_simulation(x_current, @dynamics, u);
%     % genera immagine
%     path = img_generation(x_current);
%     % calcola x_update
%     x_dmd = img2array(path);
% 
%     th=[th,t];
%     xh=[xh,x_current];
%     uh=[uh,u]; 
% end
%     toc;

%% Plot time histories

%figure(1); % Risultati numerici (in gradi)
%set(gcf, 'Color', 'w');

% Assi tempo coerenti
%tX = 0:dt:dt*(size(xh,2)-1);
%tU = 0:dt:dt*(size(uh,2)-1);

% --- θ1 (deg) ---
%subplot(3,1,1);
%plot(tX, rad2deg(xh(1,:)),'Color','#0072BD','LineWidth', 2); hold on;
%yline(theta1_ref_deg, '--r','LineWidth', 1.5);
%ylabel('\theta_1 [deg]');
%legend('\theta_1', '\theta_{1,ref}', 'Location', 'best');
%grid on;

% --- θ2 (deg) ---
%subplot(3,1,2);
%plot(tX, rad2deg(xh(2,:)),'Color','#D95319','LineWidth', 2); hold on;
%yline(theta2_ref_deg, '--r','LineWidth',1.5);
%ylabel('\theta_2 [deg]');
%legend('\theta_2', '\theta_{2,ref}', 'Location', 'best');
%grid on;

% --- Controlli ---
%subplot(3,1,3);
%if exist('uh','var') && ~isempty(uh) && size(uh,1) >= 2
    %stairs(tU, uh(1,:), 'LineWidth', 2); hold on;
    %stairs(tU, uh(2,:), 'LineWidth', 2);
    %yline(max(bu), '--', 'LineWidth', 1.2);
    %yline(-min(bu), '--', 'LineWidth', 1.2);
    %xlabel('Tempo [s]');
    %ylabel('u');
    %legend('u_1','u_2','limite sup','limite inf','Location','best');
    %grid on;
%else
    %text(0.1,0.5,'Nessun controllo disponibile','FontSize',12);
%end

figure(3); % Risultati numerici
clf; % pulisce figura precedente
set(gcf, 'Color', 'w');             % sfondo bianco per la figura
set(gca, 'Color', 'w');             % sfondo bianco per gli assi
set(0, 'DefaultFigureColor', 'w');  % sfondo di default bianco
set(0, 'DefaultAxesColor', 'w');    % sfondo assi bianco
set(0, 'DefaultFigureInvertHardcopy', 'off'); % evita inversione automatica (dark mode)

subplot(3,1,1)
set(gca, 'Color', 'w');  % forza sfondo bianco anche dentro i subplot
plot(rad2deg(xh(1,:)),'Color','#0072BD','LineWidth', 2); % x(:,1) è in radianti
yline(rad2deg(xref(1,:)), '--r','LineWidth', 1.5);
ylabel('Angle')
grid on;
legend('x_1(t)');
xlim([-0.5 102.5]);
ylim([min(rad2deg(xh(1,:))-5) max(rad2deg(xh(1,:))+5)]);

subplot(3,1,2)
set(gcf, 'Color', 'w')
set(gca, 'Color', 'w');  % forza sfondo bianco anche dentro i subplot
plot(xh(2,:),'Color','#D95319','LineWidth', 2);
yline(xref(2,:),'--r','LineWidth',1.5);
ylabel('Speed');
grid on;
legend('x_2(t)');
xlim([-0.5 102.5]);
ylim([min(xh(2,:))-0.3 max(xh(2,:))+0.3]);

subplot(3,1,3)
set(gca, 'Color', 'w');  % forza sfondo bianco anche dentro i subplot
stairs(uh,'Color','#EDB120','LineWidth',2);
yline(max(bu),'--','Color','#EDB120','LineWidth',1.5)
yline(-min(bu),'--','Color','#EDB120','LineWidth',1.5)
xlabel('Tempo');
ylabel('Control')
grid on;
legend('u(t)');
xlim([-0.5 102.5]);
ylim([-min(bu)-1 max(bu)+1]);

%figure;
%subplot(3,1,1);
%plot(time_vec, rad2deg(xh(1,:)), 'LineWidth', 1.5); hold on;
%plot(time_vec, rad2deg(xh(3,:)), 'LineWidth', 1.5);
%legend('theta1 [deg]','theta2 [deg]');
%ylabel('Angle [deg]'); grid on;

%subplot(3,1,2);
%plot(time_vec, xh(2,:), 'LineWidth', 1.5); hold on;
%plot(time_vec, xh(4,:), 'LineWidth', 1.5);
%legend('dtheta1','dtheta2'); ylabel('Angular velocity'); grid on;

%subplot(3,1,3);
%if ~isempty(uh)
    %stairs(0:dt:dt*(size(uh,2)-1), uh(1,:), 'LineWidth', 1.5); hold on;
    %stairs(0:dt:dt*(size(uh,2)-1), uh(2,:), 'LineWidth', 1.5);
    %legend('u1','u2');
    %ylabel('Control'); xlabel('Time [s]'); grid on;
%else
    %text(0.1,0.5,'No control history','FontSize',12);
%end

% ===== helpers
%function array = img2array(path)
    %global height width
    %img = imread(path);
    %resized_img = imresize(img, [height width]);
    %gray_img = rgb2gray(resized_img);
    %array = double(gray_img)/255.0;   
%end

function array = img2array(path)
    global height width
    img = imread(path);
    resized_img = imresize(img, [height width]);
    if size(resized_img,3) == 3
        gray_img = rgb2gray(resized_img);
    else
        gray_img = resized_img;
    end
    array = double(gray_img)/255.0;
end

function img_path = img_generation(state_t)
    % Disegna e salva immagine del double-link
    % --- STESSA DI QUELLA NEL DATASET ---
    global l1 l2 height width

    th1 = state_t(1);
    th2 = state_t(2);

    % coordinate giunti
    x1 =  l1 * sin(th1);
    y1 = -l1 * cos(th1);
    x2 = x1 + l2 * sin(th2);
    y2 = y1 - l2 * cos(th2);

    % CREA FIGURA INVISIBILE (prevenire zoom automatico)
    h = figure('Visible', 'off');
    set(h, 'Position', [100 100 width height]); % SAFEST
    axes('Position',[0 0 1 1]);                 % niente margini

    % ---- DISEGNO STANDARDIZZATO ----
    plot([0 x1], [0 y1], 'k', 'LineWidth', 6); hold on;
    plot([x1 x2], [y1 y2], 'k', 'LineWidth', 6);
    plot([0 x1 x2], [0 y1 y2], 'ko', 'MarkerFaceColor','k', 'MarkerSize', 8);
    axis equal;

    % LIMITE UNIVERSALE (fisso!)
    M = (l1 + l2) + 0.2;
    xlim([-M M]);
    ylim([-M M]);

    axis off;

    % ---- SALVATAGGIO IMMAGINE ----
    frame = getframe(gca);
    img = frame2im(frame);
    img = rgb2gray(img);                 % identico a dataset
    img = imresize(img, [height width]); % identico a dataset
    img_path = './current_img.png';
    imwrite(img, img_path);

    close(h);
end

%function img_path = img_generation(state_t)
    % draws double pendulum and saves ./current_img.png, returns path
    %global l1 l2
    %th1 = state_t(1);
    %th2 = state_t(2);
    %dth1 = state_t(3);
    %dth2 = state_t(4);
    
    %x1 = l1 * sin(th1);
    %y1 = -l1 * cos(th1);
    %x2 = x1 + l2 * sin(th2);
    %y2 = y1 - l2 * cos(th2);
    
    % draw
    %plot([0 x1], [0 y1], 'k-', 'LineWidth', 6); hold on;
    %plot([x1 x2], [y1 y2], 'k-', 'LineWidth', 6);
    %plot([0 x1 x2], [0 y1 y2], 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    %axis equal;
    %margin = 0.3;
    %xlim([-(l1+l2)-margin, (l1+l2)+margin]);
    %ylim([-(l1+l2)-margin, (l1+l2)+margin]);
    %set(gca,'XColor','none','YColor','none');
    %set(gcf,'Color','w');
    
    %img_path = './current_img.png';
    %saveas(gcf, img_path);
    %hold off;
%end

function final_state = rk4_simulation(x0, dynamics, u)
    global n_states
    dt = 0.01;
    tf = 0.1;

    time = 0:dt:tf;

    state = zeros(n_states, length(time));
    state(:,1) = x0;
    control = u;

    for q = 1:length(t)-1
        k1 = dynamics(time(q), state(:,q), control);
        k2 = dynamics(time(q) + dt/2, state(:,q) + dt/2 * k1, control);
        k3 = dynamics(time(q) + dt/2, state(:,q) + dt/2 * k2, control);
        k4 = dynamics(time(q) + dt, state(:,q) + dt * k3, control);

        state(:,q+1) = state(:,q) + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);        
    end

    final_state = state(:,end);

end

%function dx = double_dynamics(~, x, u)
    % Nonlinear double pendulum dynamics (4 states). u is 2x1 torques.
    %global m1 m2 l1 l2 g v1 v2
    %th1 = x(1); dth1 = x(3);
    %th2 = x(2); dth2 = x(4);
    %if isempty(u); u = [0;0]; end
    %tau1 = 0; tau2 = 0;
    %if numel(u) >= 1; tau1 = u(1); end
    %if numel(u) >= 2; tau2 = u(2); end

    % Standard equations (as used commonly; numerically stable for typical params)
    % Derived from Lagrange; here we use common form used in many references.

    % Intermediate terms
    %M11 = (m1 + m2) * l1^2;
    %M12 = m2 * l1 * l2 * cos(th1 - th2);
    %M21 = M12;
    %M22 = m2 * l2^2;

    %C1 = -m2 * l1 * l2 * dth2^2 * sin(th1 - th2) - (m1 + m2) * g * l1 * sin(th1);
    %C2 =  m2 * l1 * l2 * dth1^2 * sin(th1 - th2) - m2 * g * l2 * sin(th2);

    % Add viscous damping as simple torques -v*dtheta
    %tau_visc1 = -v1 * dth1;
    %tau_visc2 = -v2 * dth2;

    % Mass matrix and rhs
    %M = [M11, M12; M21, M22];
    %rhs = [tau1 + tau_visc1 - C1; tau2 + tau_visc2 - C2];

    % Solve for accelerations
    %dd = M \ rhs;

    %ddth1 = dd(1);
    %ddth2 = dd(2);

    %dx = [dth1; dth2; ddth1; ddth2];
%end

function dx = double_dynamics(~, x, u)
    % Nonlinear 2-Link Manipulator Dynamics (4 states)
    global m1 m2 l1 l2 g v1 v2
    
    % Stato: [th1; th2; dth1; dth2] <--- Coerente con lo standard e la cinematica
    th1 = x(1); dth1 = x(3);
    th2 = x(2); dth2 = x(4);
    
    if isempty(u); u = [0;0]; end
    tau1 = 0; tau2 = 0;
    if numel(u) >= 1; tau1 = u(1); end
    if numel(u) >= 2; tau2 = u(2); end

    % Calcolo della Mass Matrix (M), Termini Coriolis/Centrifughi (C), e Gravità (G)
    % C'è una doppia convenzione qui. Useremo la forma del Manipolatore standard.
    
    % Termini Intermedi
    delta = th1 - th2; % Non usato in questa convenzione (di solito si usano angoli giunto)
    
    % Angoli giunto assoluti (assumiamo siano questi)
    c1 = cos(th1); c2 = cos(th2); c12 = cos(th1+th2);
    s1 = sin(th1); s2 = sin(th2); s12 = sin(th1+th2);
    
    % Termini della matrice di massa M
    M11 = m1*l1^2/4 + m2*(l1^2 + l2^2/4 + l1*l2*c2);
    M12 = m2*(l2^2/4 + l1*l2/2*c2); % Era l2^2/4 + l1*l2/2*c2
    M21 = M12;
    M22 = m2*l2^2/4;
    
    % Termini C (Coriolis e Centrifughi)
    h = -m2*l1*l2*s2*dth2; % Parte di Coriolis/Centrifuga
    C1 = h * dth2;
    C2 = -h * dth1;
    
    % Termini G (Gravità)
    G1 = (m1*l1/2 + m2*l1)*g*c1 + m2*l2/2*g*c12;
    G2 = m2*l2/2*g*c12;
    
    % Matrici
    M = [M11, M12; M21, M22];
    C = [C1; C2];
    G = [G1; G2];
    
    % Vettori di coppia (Input) e Attrito Viscoso
    Tau = [tau1; tau2];
    Tau_viscous = [-v1 * dth1; -v2 * dth2];
    
    % M * ddtheta = Tau - C - G - Tau_viscous
    rhs = Tau + Tau_viscous - C - G; 
    
    % Risoluzione per le accelerazioni
    dd = M \ rhs;
    ddth1 = dd(1);
    ddth2 = dd(2);
    
    % Output: [dth1; dth2; ddth1; ddth2] <--- Ordine Standard (Coerente)
    dx = [dth1; dth2; ddth1; ddth2];
end
