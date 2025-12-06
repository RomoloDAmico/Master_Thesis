%% ================ DYNAMIC MODE DECOMPOSITION ================

clc; clear all; close all;

clear mod; % evita conflitti con la funzione mod()

tic; % avvio timer

%% ========================= MODALITY =========================
mod = 2; % 1: Img, 2: Coordinates
height = 100; width = 100; % immagini ridimensionate a 100x100 pixel

%% ======================== DATA LOAD ========================
% lettura di tutte le sottocartelle della cartella dataset
% ogni sottocartella rappresenta una simulazione contentente una serie di immagini e un file sim_data.mat con i comandi di controllo u
% consideriamo solo le sottocartelle reali (escludendo file come .DS_Store)
path = './Simulazioni/';
subdirs = dir(fullfile(path)); 
subdirs = subdirs([subdirs.isdir]) % prende solo directory 
subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'}));

fprintf('🗂️ Trovate %d cartelle di simulazione.\n', length(subdirs));

%% ====================== ORDINAMENTO NATURALE ======================
% Riordina correttamente sim1, sim2, ..., sim10, e mette SimFinale in fondo
names = {subdirs.name};

numVals = zeros(size(names));
for k = 1:numel(names)
    num = sscanf(names{k}, 'sim%d');
    if isempty(num)
        num = inf; % mette le cartelle non numeriche in fondo
    end
    numVals(k) = num;
end

[~, idx] = sort(numVals);
subdirs = subdirs(idx);

%% ====================== DIVISIONE TRAIN/TEST ======================
% Se esiste una cartella chiamata "SimFinale" (case-insensitive) in fondo
final_idx = find(strcmpi({subdirs.name}, 'SimFinale'));
if ~isempty(final_idx)
    % subdirs = [subdirs(setdiff(1:numel(subdirs), final_idx)), subdirs(final_idx)];
    % Divisione: simFinale nel test set, le altre nel training set
    train_dirs = subdirs(setdiff(1:numel(subdirs), final_idx));
    test_dirs = subdirs(final_idx);
else
    train_dirs = subdirs;
    test_dirs = [];
end

fprintf('\n📜 Divisione simulazioni:\n');
fprintf(' --> %d nel training set\n', numel(train_dirs));
fprintf(' --> %d nel test set (simFinale)\n\n', numel(test_dirs));
disp({subdirs.name});

%% ========= INITIALIZE MATRICES FOR STORING RESULTS =========
X = [];
Xp = [];
Ups = [];
Xtest = [];
Utest = [];

% contatori per debug
valid_train = 0;
valid_test = 0;
skipped = 0;

%% ==================== CICLO TRAIN SET ====================
% Per ogni simulazione:
% - legge ogni immagine;
% - la converte in una scala di grigi;
% - la normalizza tra 0 e 1;
% - la "appiattisce" in un vettore colonna;
% - costruisce la matrice data, dove ogni colonna è uno stato x_k
fprintf('🏋️‍♂️ Elaborazione TRAIN SET...\n');
for j = 1:length(train_dirs)
    subdir_name = train_dirs(j).name;
    subdir_path = fullfile(path, subdir_name);
    disp(['📂 Processing folder: ', subdir_path])
    
    % ricerca di immagini .png nella sottocartella
    imgfile  = dir(fullfile(subdir_path, '*.png'));

    if isempty(imgfile)
        disp('⚠️️ -> no png found, skippo questa cartella')
        skipped = skipped + 1;
        continue
    end
    
    data = [];
    figure('Visible','off');
    for i = 1:length(imgfile)
        disp([' reading: ', fullfile(subdir_path, imgfile(i).name)])
        img = imread(fullfile(subdir_path, imgfile(i).name));

        % conversione e normalizzazione
        resized_img = imresize(img, [height width]);
        gray_img = rgb2gray(resized_img);
        normalized_img = double(gray_img)/255.0;            

        imshow(normalized_img, [])

        % appiattisce immagine in vettore colonna e accumula 
        data = [data, normalized_img(:)];
    end
    close(gcf)
    
    %% ================ COSTRUZIONE MATRICI DMD ================
    simdata_path = fullfile(subdir_path, 'sim_data.mat');
    if ~isfile(simdata_path)
        warning(['⚠️ sim_data.mat non trovato in ', subdir_path, ' -> salto'])
        skipped = skipped + 1;
        continue    
    end

    % Caricamento dei comandi di controllo u_k
    U_data = load(simdata_path);
    if ~isfield(U_data, 'u')
        warning(['⚠️ sim_data.mat non contiene variabile u in ', subdir_path, ' -> salto'])
        skipped = skipped + 1;
        continue    
    end
    
    ups = U_data.u; % 2 x N
    disp(['Original u size: ', mat2str(size(ups))]); 

    % correzione della forma di 'u' (deve essere riga)
    % ups = ups(:)'; % Ensure 'ups' is a row vector, 1xN  % PRIMA non era commentato
    
    fnames = {imgfile.name};
    frame_ids = zeros(1, numel(fnames));
    for kk = 1:numel(fnames)
        tok = regexp(fnames{kk}, '(\d+)', 'match');
        if ~isempty(tok)
            frame_ids(kk) = str2double(tok{end});
        else
            error('Nome file frame non nel formato atteso: %s', fnames{kk});
        end
    end

    [frame_ids_sorted, sort_idx] = sort(frame_ids);
    fnames = fnames(sort_idx);

    % verifica coerenza lunghezze: data ha M colonne
    M = size(data, 2);
    % per DMDc usiamo X=data(:,1:end-1), Xp=data(:,2:end) -> quindi #comandi = M-1
    % expected_u_len = max(0, M-1);

    if M < 2
        disp('️️️⚠️ -> meno di 2 colonne, salto')
        skipped = skipped + 1;
        continue
    end

    Tfull = size(ups, 2);
    if any(frame_ids_sorted > Tfull)
        frame_ids_sorted(frame_ids_sorted > Tfull) = Tfull;
    end

    ups_ds = zeros(size(ups, 1), M-1);
    for kf = 1:(M-1)
        idx_u = frame_ids_sorted(kf);
        ups_ds(:, kf) = ups(:, idx_u);
    end

    disp(['Downsampled ups size: ', mat2str(size(ups_ds))]);

    % ✅ Accumulo nel training set
    X = [X, data(:, 1:end-1)];
    Xp = [Xp, data(:, 2:end)];
    Ups = [Ups, ups_ds];
    valid_train = valid_train + 1;
end

%% ======================= CICLO TEST SET =======================
fprintf('\n🧪 Elaborazione TEST SET (simFinale)... \n');
for j = 1:length(test_dirs)
    subdir_name = test_dirs(j).name;
    subdir_path = fullfile(path, subdir_name);
    disp(['📂 Processing folder: ', subdir_path])
    
    imgfile  = dir(fullfile(subdir_path, '*.png'));
    if isempty(imgfile)
        disp('⚠️️ -> no png found, skippo questa cartella')
        skipped = skipped + 1;
        continue
    end
    
    data = [];
    figure('Visible','off');
    for i = 1:length(imgfile)
        disp([' reading: ', fullfile(subdir_path, imgfile(i).name)])
        img = imread(fullfile(subdir_path, imgfile(i).name));
        resized_img = imresize(img, [height width]);
        gray_img = rgb2gray(resized_img);
        normalized_img = double(gray_img)/255.0;
        data = [data, normalized_img(:)];
    end
    close(gcf)

    simdata_path = fullfile(subdir_path, 'sim_data.mat');
    if ~isfile(simdata_path)
        warning(['⚠️ sim_data.mat non trovato in ', subdir_path, ' -> salto'])
        skipped = skipped + 1;
        continue    
    end

    U_data = load(simdata_path);
    if ~isfield(U_data, 'u')
        warning(['⚠️ sim_data.mat non contiene variabile u in ', subdir_path, ' -> salto'])
        skipped = skipped + 1;
        continue    
    end
    
    ups = U_data.u; 
    fnames = {imgfile.name};
    frame_ids = zeros(1, numel(fnames));
    for kk = 1:numel(fnames)
        tok = regexp(fnames{kk}, '(\d+)', 'match');
        if ~isempty(tok)
            frame_ids(kk) = str2double(tok{end});
        else
            error('Nome file frame non nel formato atteso: %s', fnames{kk});
        end
    end

    [frame_ids_sorted, sort_idx] = sort(frame_ids);
    fnames = fnames(sort_idx);

    M = size(data, 2);
    if M < 2
        disp('⚠️ -> meno di 2 colonne, salto')
        skipped = skipped + 1;
        continue
    end

    Tfull = size(ups, 2);
    if any(frame_ids_sorted > Tfull)
        frame_ids_sorted(frame_ids_sorted > Tfull) = Tfull;
    end

    ups_ds = zeros(size(ups, 1), M-1);
    for kf = 1:(M-1)
        idx_u = frame_ids_sorted(kf);
        ups_ds(:, kf) = ups(:, idx_u);
    end

    % ✅ Accumulo nel test set
    Xtest = [Xtest, data(:,1:end-1)];
    Utest = [Utest, ups_ds];
    valid_test = valid_test + 1;
end

disp(['----------------------------------------'])
disp('📊 CHECK COMPLETO DATI')
disp('Sizes after data load')
disp(['valid_train size: '  num2str(valid_train)])
disp(['valid_test size: '  num2str(valid_test)])
disp(['skipped size: '  num2str(skipped)])
disp(['X size: ' mat2str(size(X))]) 
disp(['Xp size: ' mat2str(size(Xp))])
disp(['Ups size: ' mat2str(size(Ups))])
disp(['Xtest size: ' mat2str(size(Xtest))])
disp(['Utest size: ' mat2str(size(Utest))])

if isempty(X) || isempty(Xp) || isempty(Ups)
    error('❌ ERRORE: una o più matrici (X, Xp, Ups) sono vuote. Controllare caricamenti')
else
    disp('✅ Tutte le matrici contengono dati validi')
end

%% ========================= DMDc =========================
% DMDc cerca di identificare matrici A e B tali che X'≈ AX + BU

% controlli preliminari
if isempty(X) || isempty(Xp)
    error('X o Xp vuote: assicurarsi che il dataset di training sia caricato correttamente')
end

Ups = Ups(:, 1:size(X, 2));

% Creazione matrice
Omega = [X;Ups]; % dimensione: (n+q) x m, con m = numero colonne

[U, Sig, V] = svd(Omega, 'econ');

% Troncamento degli r-autovalori principali per ridurre la dimensionalità
thresh = 1e-10;
rtil_eff = length(find(diag(Sig) > thresh));
% rtil = length(find(diag(Sig) > thresh)); % TRY: SPECTRUM

% evita di superare le dimensioni effettive di U
rtil = min([30, rtil_eff, size(U, 2)])
% rtil = 30;
r_plot(Sig,rtil);

size(U)
rtil

Util    = U(:,1:rtil); 
Sigtil  = Sig(1:rtil,1:rtil);
Vtil    = V(:,1:rtil); 

% Stessa cosa per X'
[U_xp, Sig_xp, V_xp] = svd(Xp,'econ');

% thresh = 1e-10;
r_eff = length(find(diag(Sig_xp) > thresh));
r = min([30, r_eff, size(U_xp, 2)])
% r = 30;
r_plot(Sig_xp, r);

Uhat    = U_xp(:,1:r); 
Sighat  = Sig_xp(1:r,1:r);
Vbar    = V_xp(:,1:r); 

%% ================== SEPARAZIONE U_1 e U_2 ==================
% U_1 e U_2: parti di Util associate rispettivamente a X e U
n = size(X, 1); 
q = size(Ups, 1);
U_1 = Util(1:n, :);
U_2 = Util(n+1:end, :);

%% ========== MATRICI DINAMICHE DEL MODELLO RIDOTTO ==========
disp("------ DIMENSIONI ------")
disp(["size(Xp): ", mat2str(size(Xp))])
disp(["size(Vtil): ", mat2str(size(Vtil))])
disp(["size(Sigtil): ", mat2str(size(Sigtil))])
disp(["size(U_1): ", mat2str(size(U_1))])
disp(["size(Uhat): ", mat2str(size(Uhat))])

% controlli per evitare prodotti non compatibili
if size(Xp, 2) ~= size(Vtil, 1)
    warning('Missmatch: #cols(Xp) != #rows(Vtil). Verifica i dati (Xp e Omega devono avere stesso numero di colonne).')
end

if size(Sigtil, 1) ~= size(Vtil, 2)
    warning('Attenzione: dimensioni Sigtil/Vtil non coerenti.')
end

approxA = Uhat' * (Xp) * Vtil * inv(Sigtil) * U_1' * Uhat;
approxB = Uhat' * (Xp) * Vtil * inv(Sigtil) * U_2';

[W,D] = eig(approxA);
dmd_spectrum(D)

%%
Phi = Xp * Vtil * inv(Sigtil) * U_1'*Uhat * W;

Atilde = Xp * Vtil * inv(Sigtil) * U_1';
Btilde = Xp * Vtil * inv(Sigtil) * U_2';

%% ======================== DATA SAVE ========================
% save('./dynamics_original_dim.mat','Atilde','Btilde')
% save('./dynamics_reduced_dim.mat','approxA','approxB','Uhat')
save(fullfile(pwd, 'dynamics_original_dim.mat'), 'Atilde', 'Btilde');
save(fullfile(pwd, 'dynamics_reduced_dim.mat'), 'approxA', 'approxB', 'Uhat');
disp('✅ Matrici salvate con successo!')

%% ====================== DMD EVALUATION ======================
% Vogliamo visualizzare la DINAMICA ricostruita dalla DMDc.
% 1) Proiettiamo Xtest nello spazio ridotto → z_test
% 2) Evolviamo z_test con approxA, approxB → z_dmd
% 3) Ricostruiamo x_dmd = Uhat * z_dmd
% 4) Creiamo la GIF della dinamica (immagini)

disp('🎬 Ricostruzione dinamica DMDc sul test set...')

% Coordinate ridotte del test set (proiezione)
Ztest = Uhat' * Xtest;       % dimensione: r × T

% Evoluzione nel sottospazio ridotto
T = size(Ztest, 2);
Z_dmd = zeros(size(Ztest));
Z_dmd(:,1) = Ztest(:,1);     % stato iniziale

for k = 1:T-1
    Z_dmd(:,k+1) = approxA * Z_dmd(:,k) + approxB * Utest(:,k);
end

% Ricostruzione nello spazio originale
X_dmd = Uhat * Z_dmd;        % n × T

% -------------------------------------------------------------
% CREAZIONE GIF (animazione delle immagini ricostruite)
% -------------------------------------------------------------

gif_filename = 'dmd_animation.gif';
figure;
axis tight manual;

disp('🎥 Generazione GIF della dinamica...')

for frame = 1:T
    clf;

    % Stato ricostruito
    img_vec = X_dmd(:, frame);
    img = reshape(img_vec, height, width);     % Ricostruzione immagine

    imshow(img, []);
    title(sprintf('DMDc reconstructed frame %d', frame));

    drawnow;

    % Salvataggio frame
    frame_img = getframe(gcf);
    im = frame2im(frame_img);
    [imind, cm] = rgb2ind(im, 256);

    if frame == 1
        imwrite(imind, cm, gif_filename, 'gif', ...
                'LoopCount', inf, 'DelayTime', 0.1);
    else
        imwrite(imind, cm, gif_filename, 'gif', ...
                'WriteMode', 'append', 'DelayTime', 0.1);
    end
end

disp('✅ GIF della dinamica generata con successo!')


%% =================== RECONSTRUCTION ERROR ===================
% calcolo dell'errore medio tra gli stati reali e quelli previsti dal
% modello DMDc
disp('📈 Calcolo errori di ricostruzione...');

% Errore nello spazio ridotto (modello dinamico)
E_reduced = mean(abs(Ztest - Z_dmd), 1);

% Errore nello spazio originale (immagini reali);
E_original = mean(abs(Xtest - X_dmd), 1);

% Plot1 - Errore nello spazio ridotto
figure();
plot(E_reduced, 'LineWidth', 1.5); 
title('Reduced-space reconstruction error');
xlabel('Frame');
ylabel('Mean Absolute Error');
saveas(gcf, 'Error_ReducedSpace.png');

% Plot2 - Errore nello spazio originale
figure();
plot(E_original, 'LineWidth', 1.5); 
title('Original-space reconstruction error');
xlabel('Frame');
ylabel('Mean Absolute Error');
saveas(gcf, 'Error_OriginalSpace.png');

% Plot3 - Confronto diretto
figure();
plot(E_reduced, 'LineWidth', 1.5); hold on;
plot(E_original, 'LineWidth', 1.5, 'LineStyle', '--');
legend('Errore ridotto', 'Errore originale');
title('Confronto errori DMDc');
xlabel('Frame');
ylabel('Mean Absolute Error');
saveas(gcf, 'Error_Comparison.png');

%% ===================== FUNCTIONS =====================
% mostra la quantità di energia spiegata dai modi (SVD energy plot)
function r_plot(Sigma,num_modes)
    singular_values = diag(Sigma);
    total_energy = sum(singular_values.^2);
    cumulative_energy = cumsum(singular_values.^2);
    energy_percentage = (cumulative_energy / total_energy) * 100;

    figure();
    plot(energy_percentage, 'o-', 'LineWidth', 1.5);
    hold on
    xline(num_modes, '--r', 'LineWidth', 1.5);
    xlabel('Number of Modes', 'FontSize', 12);
    ylabel('Cumulative Energy (%)', 'FontSize', 12);
    title('Cumulative Energy Captured by Singular Values', 'FontSize', 14);
    text(num_modes + 1, 50, sprintf('Modes: %d', num_modes), 'Color', 'red', 'FontSize', 12);
    grid on;
end

% mostra lo spettro DMD (autovalori sul piano complesso)
function dmd_spectrum(L)
    figure()
    theta = (0:1:100)*2*pi/100;
    plot(cos(theta),sin(theta),'k--') % plot unit circle
    hold on, grid on
    scatter(real(diag(L)),imag(diag(L)),'ok')
    axis([-1.1 1.1 -1.1 1.1]);
    title('DMD Spectrum', 'FontSize', 14);
    saveas(gcf, 'DMD Spectrum');
end

elapsed_time = toc;  % ⏱️ Ferma timer e restituisce il tempo in secondi
fprintf('⏳ Durata totale esecuzione: %.2f secondi\n', elapsed_time);
