%% ======================= MPC FULL MODEL (ORIGINALE) =======================
% ============================= DOUBLE LINK ============================= 
clc; clear all; close all;

%% ================= DMD ORIGINAL MATRICES (NO SVD) =================   
% Carichiamo le matrici GIGANTI (Full State)
if ~isfile('dynamics_original_dim.mat')
    error('File dynamics_original_dim.mat non trovato. Esegui il dmd (con risoluzione bassa!)');
end
load('dynamics_original_dim.mat'); % Deve contenere: Atilde, Btilde (NON serve Uhat)

% Assegno alle variabili standard del modello
A = Atilde;
B = Btilde;

% 1. CORREZIONE DEL GUADAGNO (Stesso scaling del ridotto per coerenza)
B = B * 500; 
% 2. CORREZIONE DEL SEGNO 
B(:, 2) = -B(:, 2);

clear Atilde Btilde; % Rimuovo le copie per salvare RAM

%% ========================= PHYSICAL PARATEMERS =========================
global g m1 m2 l1 l2 v1 v2 height width n_states n_controls xref

% IMPORTANTE: Queste dimensioni devono coincidere con quelle usate nel dmd
% Se A è 400x400, allora sqrt(400) = 20.
nx_full = size(A, 1);       
side_len = sqrt(nx_full);   

height = side_len;
width  = side_len;
fprintf('Running MPC on Full Model with resolution: %dx%d (%d states)\n', height, width, nx_full);

% Parametri fisici (per la simulazione ode45)
g  = 9.81;            
l1 = 1.0; l2 = 0.7;        
m1 = 2.0; m2 = 1.5;   
v1 = 6.0; v2 = 3.0;   
n_states    = 4;   % Stati fisici reali
n_controls  = 2;   % Ingressi

%% ========== STATI INIZIALI E RIFERIMENTO ==========
% Stato fisico (Realtà)
x_initial = [0; 0; 0; 0];        
xref      = [pi/4; pi/6; 0; 0];  

% Immagini iniziale e riferimento
x_ini_path = './x_ini_double_link.png';
x_ref_path = './x_ref_double_link.png';

% Converto in array (rispettando la nuova width/height)
x_ini_img = img2array(x_ini_path);   
x_ref_img = img2array(x_ref_path);   

% --- DIFFERENZA CHIAVE RISPETTO ALL'MPC RIDOTTO ---
% NESSUNA PROIEZIONE Uhat'. Lo stato è l'immagine intera.
x_ini = x_ini_img(:);        % nx_full x 1
x_ref = x_ref_img(:);        % nx_full x 1

%% ========== HYPERPARAMETRI MPC (DOMINIO ORIGINALE) ==========
nx = size(A,2);        % Dimensione stato (es. 400)
nu = size(B,2);        % Dimensione input (2)

% Usa 'speye' (sparse eye) per risparmiare memoria
Q = 500 * speye(nx);     
R = 0.1 * eye(nu);      

% Orizzonte RIDOTTO per sopravvivere al calcolo pesante
Np = 3;                % Nell'mpc ridotto era 10, qui dobbiamo abbassare

% Vincoli
Hu = [eye(nu); -eye(nu)];
bu = 20 * ones(2*nu,1); 

%% ========== COSTRUZIONE OPTIMIZER (YALMIP + GUROBI) ==========
u  = sdpvar(nu, Np);
x  = sdpvar(nx, Np+1);  % Questa matrice pesa molto in RAM
x0 = sdpvar(nx, 1);     
xr = sdpvar(nx, 1);     

constraints = [];
objective   = 0;

constraints = [constraints, x(:,1) == x0];

for k = 1:Np
    % Costo di stadio
    objective = objective + (x(:,k) - xr)'*Q*(x(:,k) - xr) + u(:,k)'*R*u(:,k);
    
    % Dinamica FULL (Collo di bottiglia computazionale)
    constraints = [constraints, x(:,k+1) == A*x(:,k) + B*u(:,k)];
    
    % Vincoli
    constraints = [constraints, Hu*u(:,k) <= bu];
end

% Verbose = 1 per vedere se il solver lavora o è bloccato
ops = sdpsettings('verbose', 1, 'solver','gurobi', 'gurobi.Presolve', 0);
predictor = optimizer(constraints, objective, ops, {x0, xr}, {x, u});

%% ========== SIMULAZIONE CLOSED-LOOP ==========
dt_step = 0.01;
tspan = [0 dt_step];     
t_final = 3.0;   

xh    = x_initial;       % Storico stati fisici
x_mpc = x_ini;           % Storico stati MPC (Full Pixel vectors)
uh    = [];    

filename = 'animation_full_model.gif';

figure(1); clf;
set(gcf, 'Color', 'w');

disp('Avvio simulazione MPC Full Model...');
i_step = 0;

for t = 0: dt_step: t_final
    i_step = i_step + 1;
    
    % === MPC nel dominio ORIGINALE ===
    % Passo il vettore pixel intero (es. 400x1)
    [sol, errorcode] = predictor(x_mpc(:,end), x_ref);
    
    if errorcode ~= 0 
        warning('Solver fallito o timeout!');
    end

    x_pred = sol{1};
    u_pred = sol{2};
    u_curr = u_pred(:,1);         
    uh     = [uh, u_curr];
    
    % === Dinamica nel dominio fisico (ode45) ===
    % Questa parte è IDENTICA all'MPC ridotto (La fisica non cambia)
    [~, x_ode] = ode45(@(tt,xx) dynamics2link(tt, xx, u_curr), tspan, xh(:,end));  
    xh = [xh, x_ode(end,:)'];     
    
    % === Generazione immagine e GIF ===
    cla;
    % Nota: img_generation salverà l'immagine. 
    % Assicurarsi che non venga ridimensionata internamente a 100x100 hardcoded,
    % ma vengano usate le proporzioni corrette.
    img_generation(xh(1:2, end), '.');   
    
    if exist('current_img.png', 'file'), delete('current_img.png'); end
    movefile('frame_0001.png', 'current_img.png');
    path = './current_img.png';
    
    drawnow;
    
    % Gif creation (opzionale, rallenta un po')
    frame = getframe(gcf);
    im    = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if i_step == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', 1, 'DelayTime', 0.05);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
    
    % === Aggiorno stato MPC leggendo i PIXEL REALI ===
    x_img_new = img2array(path);         % Legge e ridimensiona a (height x width)
    x_mpc     = [x_mpc, x_img_new(:)];   % Accoda vettore colonna intero
    
    fprintf('Time: %.2f | u: [%.2f, %.2f]\n', t, u_curr(1), u_curr(2));
end

%% ========== PLOT RISULTATI NUMERICI ==========
% IDENTICO ALL'MPC RIDOTTO (Plotta le variabili fisiche xh e uh)
N = size(xh,2);
t_vec = linspace(0, t_final, N);

figure(2); clf;

% q1
subplot(4,1,1); 
plot(t_vec, rad2deg(xh(1,:)), 'Color', '#0072BD', 'LineWidth', 2); hold on;
yline(rad2deg(xref(1)), '--r', 'LineWidth', 1.5); 
ylabel('\theta_1 [deg]'); 
grid on; 
xlim([0 t_final]);

% q2
subplot(4,1,2); 
plot(t_vec, rad2deg(xh(2,:)), 'Color', '#D95319', 'LineWidth', 2); hold on;
yline(rad2deg(xref(2)), '--r', 'LineWidth', 1.5); 
ylabel('\theta_2 [deg]'); 
grid on;
xlim([0 t_final]);

% velocità
subplot(4,1,3); 
plot(t_vec, xh(3,:), 'LineWidth', 2); hold on; 
plot(t_vec, xh(4,:), 'LineWidth', 2);
yline(0, '--k', 'LineWidth', 1);
ylabel('velocities [rad/s]');
legend('\omega_1', '\omega_2', 'Location', 'best');
grid on;
xlim([0 t_final]);

% controlli
subplot(4,1,4); 
if ~isempty(uh)
    t_u = t_vec(1:size(uh,2));  % stessa lunghezza di uh
    stairs(t_u, uh(1,:), 'LineWidth', 2); hold on; 
    stairs(t_u, uh(2,:), 'LineWidth', 2);
    yline(max(bu),  '--', 'LineWidth', 1.5);
    yline(-max(bu), '--', 'LineWidth', 1.5);
end

xlabel('Tempoo [s]');
ylabel('u_1, u_2');
legend('u_1', 'u_2', 'u_{max}', 'u_{min}', 'Location', 'best');
grid on;
xlim([0 t_final]);

sgtitle('MPC modello originale');
saveas(gcf, 'MPC modello originale.png');

%% ===================== FUNZIONI DI SUPPORTO ============================
function array = img2array(path)
    global height width
    img = imread(path);
    % Ridimensionamento fondamentale per matchare la matrice A
    img = imresize(img, [height width]); 
    if size(img,3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    array = double(gray_img) / 255.0;
end

%function dx = dynamics2link(~, x, u)
    %global g l1 l2 m1 m2 v1 v2
    %q1=x(1); q2=x(2); dq1=x(3); dq2=x(4);
    %M11 = (m1+m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*cos(q2);
    %M12 = m2*l2^2 + m2*l1*l2*cos(q2);
    %M22 = m2*l2^2;
    %M = [M11 M12; M12 M22];
    %h = -m2*l1*l2*sin(q2);
    %C = [h*dq2*(2*dq1+dq2); -h*dq1^2];
    %G = [(m1+m2)*g*l1*sin(q1)+m2*g*l2*sin(q1+q2); m2*g*l2*sin(q1+q2)];
    %D = [v1*dq1; v2*dq2];
    %ddq = M\(u - C - G - D);
    %dx = [dq1; dq2; ddq];
%end