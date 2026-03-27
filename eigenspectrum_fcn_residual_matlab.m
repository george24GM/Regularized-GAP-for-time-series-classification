function eigenspectrum_fcn_residual_matlab()
    % ============================================================
    % General FCN / residual FCN on AR(1) input
    %
    % Implements:
    %   H_t^(l) = I^(l) H_t^(l-1) + sigma( b^(l) + sum_{j=0}^{k_l-1} W_j^(l) H_{t+j}^{(l-1)} )
    %
    % with options:
    %   - arbitrary number of layers L
    %   - arbitrary filter widths k_l
    %   - arbitrary layer widths m_l
    %   - residual connection yes / no
    %   - He initialization
    %   - AR(1) input
    %   - Bartlett HAC long-run covariance estimation
    % ============================================================

    clear; clc; close all;

    % ============================================================
    % Global settings
    % ============================================================
    thetas = [0.0, 0.99];
    n = 100000;
    sigma_ar = 5.0;
    act = 'relu';

    use_residual = false; % true / false

    % ============================================================
    % Default architecture
    % ============================================================
    default_arch.L = 1; %3;
    default_arch.k_list = 1;%[8, 5, 3];
    default_arch.m_list = [10];%[10, 20, 10];
    default_arch.d = 1;    % scalar AR(1) input

    % =======================================================
    % Experiment A: vary theta, fix one architecture
    % =======================================================
    results_theta = [];

    params_fixed = draw_fcn_parameters(default_arch, use_residual, 35);

    for i = 1:length(thetas)
        theta = thetas(i);

        [Sigma_hat, eigvals, evr1, evr3, H_gap, X] = one_covariance_run_fixed_params( ...
            theta, params_fixed, n, sigma_ar, act, 5000 + i, []); %#ok<NASGU>

        row.theta = theta;
        row.arch = architecture_string(params_fixed);
        row.lambda1 = eigvals(1);
        row.evr1 = evr1;
        row.evr3 = evr3;
        results_theta = [results_theta; row]; %#ok<AGROW>

        filename = sprintf('Corr_theta_%s_%s_resid_%d.eps', ...
            strrep(sprintf('%.3f', theta), '.', 'p'), ...
            short_architecture_string(params_fixed), ...
            use_residual);

        title_str = make_latex_title(theta, params_fixed);

        plot_correlation_matrix(Sigma_hat, title_str, filename);
    end

    print_results_table(results_theta, ...
        'Experiment A: influence of theta (fixed architecture)');

    % =======================================================
    % Experiment B: vary architecture, fixed theta
    % =======================================================
    fixed_theta = 0.65;
    results_arch = [];

    arch_list = {};
    arch1.L = 1; arch1.k_list = [1]; arch1.m_list = [10]; arch1.d = 1;
    arch2.L = 2; arch2.k_list = [5, 3]; arch2.m_list = [10, 10]; arch2.d = 1;
    arch3.L = 3; arch3.k_list = [8, 5, 3]; arch3.m_list = [10, 20, 10]; arch3.d = 1;
    arch4.L = 4; arch4.k_list = [8, 5, 5, 3]; arch4.m_list = [10, 20, 20, 10]; arch4.d = 1;

    arch_list{1} = arch1;
    arch_list{2} = arch2;
    arch_list{3} = arch3;
    arch_list{4} = arch4;

    for j = 1:length(arch_list)
        params_arch = draw_fcn_parameters(arch_list{j}, use_residual, 100 + j);

        [Sigma_hat, eigvals, evr1, evr3, H_gap, X] = one_covariance_run_fixed_params( ...
            fixed_theta, params_arch, n, sigma_ar, act, 7000 + j, []); %#ok<NASGU>

        row.theta = fixed_theta;
        row.arch = architecture_string(params_arch);
        row.lambda1 = eigvals(1);
        row.evr1 = evr1;
        row.evr3 = evr3;
        results_arch = [results_arch; row]; %#ok<AGROW>

        filename = sprintf('Corr_theta_%s_%s_resid_%d.eps', ...
            strrep(sprintf('%.3f', fixed_theta), '.', 'p'), ...
            short_architecture_string(params_arch), ...
            use_residual);

        title_str = make_latex_title(fixed_theta, params_arch);

        plot_correlation_matrix(Sigma_hat, title_str, filename);
    end

    print_results_table(results_arch, ...
        'Experiment B: influence of architecture (fixed theta)');
end


% ============================================================
% 1. AR(1) generator
% ============================================================
function X = generate_ar1(n, theta, sigma, burnin, seed)
    if nargin < 4 || isempty(burnin)
        burnin = max(1000, ceil(20 / max(1e-3, 1 - abs(theta))));
    end
    if nargin >= 5 && ~isempty(seed)
        rng(seed);
    end

    if abs(theta) >= 1
        error('Need |theta| < 1 for stationary AR(1).');
    end

    eps = sigma * randn(n + burnin, 1);
    X = zeros(n + burnin, 1);
    X(1) = randn * sigma / sqrt(1.0 - theta^2);

    for t = 2:(n + burnin)
        X(t) = theta * X(t - 1) + eps(t);
    end

    X = X((burnin + 1):end);
end


% ============================================================
% 2. Activation
% ============================================================
function y = activation(x, kind)
    switch lower(kind)
        case 'relu'
            y = max(x, 0.0);
        case 'sigmoid'
            y = 1.0 ./ (1.0 + exp(-x));
        otherwise
            error('kind must be ''relu'' or ''sigmoid''.');
    end
end


% ============================================================
% 3. He initialization
% ============================================================
function W = he_weight(shape, fan_in)
    W = sqrt(2.0 / fan_in) * randn(shape(1), shape(2));
end


% ============================================================
% 4. Draw FCN parameters
% ============================================================
function params = draw_fcn_parameters(arch, use_residual, seed)
    rng(seed);

    L = arch.L;
    k_list = arch.k_list(:)';
    m_list = arch.m_list(:)';
    d = arch.d;

    if length(k_list) ~= L || length(m_list) ~= L
        error('Lengths of k_list and m_list must both be L.');
    end

    params.L = L;
    params.k_list = k_list;
    params.m_list = m_list;
    params.d = d;
    params.use_residual = use_residual;

    params.W = cell(L, 1);
    params.b = cell(L, 1);
    params.I = cell(L, 1);

    m_prev = d;

    for ell = 1:L
        k_ell = k_list(ell);
        m_ell = m_list(ell);

        params.W{ell} = cell(k_ell, 1);
        for j = 1:k_ell
            fan_in = m_prev * k_ell;
            params.W{ell}{j} = he_weight([m_ell, m_prev], fan_in);
        end

        params.b{ell} = 0 * randn(m_ell, 1);

        if use_residual
            if m_ell == m_prev
                params.I{ell} = eye(m_ell);
            else
                % projection residual if dimensions change
                params.I{ell} = he_weight([m_ell, m_prev], m_prev);
            end
        else
            params.I{ell} = zeros(m_ell, m_prev);
        end

        m_prev = m_ell;
    end
end


% ============================================================
% 5. Build zero-padded input H^(0)
% ============================================================
function H0 = build_input_process(X, d)
    X = X(:);
    n = length(X);

    if d ~= 1
        error('Current AR(1) generator gives scalar input, so d must be 1.');
    end

    H0 = X';  % 1 x n
end


% ============================================================
% 6. Forward pass through all layers
% ============================================================
function [H_all, H_gap] = fcn_feature_process(X, params, act)
    % Representation convention:
    % H_all{ell} is m_ell x n
    % columns correspond to time t=1,...,n

    n = length(X);
    L = params.L;

    H_all = cell(L + 1, 1);
    H_all{1} = build_input_process(X, params.d);  % m0 x n

    for ell = 1:L
        k_ell = params.k_list(ell);
        m_ell = params.m_list(ell);
        H_prev = H_all{ell};              % m_prev x n
        m_prev = size(H_prev, 1);

        H_prev_pad = [H_prev, zeros(m_prev, k_ell - 1)];  % zero padding to the right
        H_curr = zeros(m_ell, n);

        for t = 1:n
            conv_sum = params.b{ell};

            for j = 0:(k_ell - 1)
                Wj = params.W{ell}{j + 1};      % m_ell x m_prev
                h_in = H_prev_pad(:, t + j);    % m_prev x 1
                conv_sum = conv_sum + Wj * h_in;
            end

            residual_term = params.I{ell} * H_prev(:, t);
            H_curr(:, t) = residual_term + activation(conv_sum, act);
        end

        H_all{ell + 1} = H_curr;
    end

    H_L = H_all{L + 1};          % m_L x n
    H_gap = mean(H_L, 2);        % m_L x 1
end


% ============================================================
% 7. Bandwidth selector for HAC
% ============================================================
function bandwidth = choose_hac_bandwidth(n_eff, theta)
    bw_base = floor(n_eff^(1/3));

    if theta >= 0.95
        bw_persist = ceil(5 / max(1e-6, 1 - theta));
    else
        bw_persist = ceil(2 / max(1e-6, 1 - theta));
    end

    bandwidth = max(bw_base, bw_persist);
    bandwidth = min(bandwidth, floor(n_eff / 6));
    bandwidth = max(bandwidth, 1);
end


% ============================================================
% 8. Long-run covariance estimator (Bartlett HAC)
% ============================================================
function Sigma_hat = estimate_long_run_covariance(G, bandwidth)
    % Expects G of shape (time, dimension)

    if isvector(G)
        G = G(:);
    end

    [n_eff, ~] = size(G);

    if nargin < 2 || isempty(bandwidth)
        error('bandwidth must be provided explicitly.');
    end

    Gc = G - mean(G, 1);

    Sigma_hat = (Gc' * Gc) / n_eff;

    for h = 1:bandwidth
        weight = 1.0 - h / (bandwidth + 1.0);
        Gh = (Gc(1:end-h, :)' * Gc(1+h:end, :)) / n_eff;
        Sigma_hat = Sigma_hat + weight * (Gh + Gh');
    end

    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat');
end


% ============================================================
% 9. One run with fixed parameters
% ============================================================
function [Sigma_hat, eigvals, evr1, evr3, H_gap, X] = one_covariance_run_fixed_params( ...
    theta, params, n, sigma_ar, act, seed_data, bandwidth)

    if nargin < 7 || isempty(bandwidth)
        bandwidth = [];
    end

    burnin = max(1000, ceil(20 / max(1e-3, 1 - abs(theta))));
    X = generate_ar1(n, theta, sigma_ar, burnin, seed_data);

    [H_all, H_gap] = fcn_feature_process(X, params, act); %#ok<NASGU>
    H_L = H_all{end};              % m_L x n
    G = H_L';                      % n x m_L

    if isempty(bandwidth)
        bandwidth = choose_hac_bandwidth(size(G, 1), theta);
    end

    fprintf('theta=%.3f | HAC bandwidth=%d | burnin=%d\n', theta, bandwidth, burnin);

    Sigma_hat = estimate_long_run_covariance(G, bandwidth);

    eigvals = sort(eig(Sigma_hat), 'descend');
    tr = trace(Sigma_hat);

    if tr <= 0
        evr1 = NaN;
        evr3 = NaN;
    else
        evr1 = eigvals(1) / tr;
        evr3 = sum(eigvals(1:min(3, length(eigvals)))) / tr;
    end
end


% ============================================================
% 10. Plot empirical autocorrelation matrix
% ============================================================
function plot_correlation_matrix(Sigma_hat, title_str, filename_eps)
    d = sqrt(max(diag(Sigma_hat), 1e-12));
    Corr_hat = Sigma_hat ./ (d * d');
    Corr_hat = max(min(Corr_hat, 1.0), -1.0);
    Corr_hat(1:size(Corr_hat,1)+1:end) = 1.0;

    [V, D] = eig(Corr_hat);
    [~, idx] = max(diag(D));
    [~, order] = sort(V(:, idx), 'descend');
    Corr_hat = Corr_hat(order, order);

    C = size(Corr_hat, 1);

    figure('Color', 'w', 'Position', [100, 100, 650, 550]);
    imagesc(Corr_hat, [-1, 1]);
    axis equal tight;

    colormap(coolwarm_map(256));

    cb = colorbar;
    cb.FontSize = 22;

    tick_pos = unique([ ...
        1, ...
        round(0.25 * (C - 1)) + 1, ...
        round(0.75 * (C - 1)) + 1, ...
        C ...
    ]);

    set(gca, 'XTick', tick_pos, 'YTick', tick_pos, 'FontSize', 24);

    xlabel('Output neuron', 'FontSize', 24);
    ylabel('Output neuron', 'FontSize', 24);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 36);

    set(gca, 'YDir', 'reverse');

    drawnow;

    if nargin >= 3 && ~isempty(filename_eps)
        print(gcf, filename_eps, '-depsc', '-r300');
    end
end


% ============================================================
% 11. Colormap
% ============================================================
function cmap = coolwarm_map(m)
    if nargin < 1
        m = 256;
    end

    anchors = [
        0.2298, 0.2987, 0.7537
        0.8650, 0.8650, 0.8650
        0.7057, 0.0156, 0.1502
    ];

    x = [0; 0.5; 1];
    xi = linspace(0, 1, m)';

    cmap = zeros(m, 3);
    for k = 1:3
        cmap(:,k) = interp1(x, anchors(:,k), xi, 'pchip');
    end

    cmap = max(min(cmap, 1), 0);
end


% ============================================================
% 12. Pretty-print tables
% ============================================================
function print_results_table(results, label)
    fprintf('\n%s\n', label);
    fprintf('%s\n', repmat('-', 1, length(label)));
    fprintf('%8s %28s %16s %12s %12s\n', ...
        'theta', 'architecture', 'lambda1', 'EVR1', 'EVR3');

    for i = 1:length(results)
        fprintf('%8.3f %28s %16.6f %12.6f %12.6f\n', ...
            results(i).theta, ...
            results(i).arch, ...
            results(i).lambda1, ...
            results(i).evr1, ...
            results(i).evr3);
    end
end


% ============================================================
% 13. Architecture string helpers
% ============================================================
function s = architecture_string(params)
    s = sprintf('L=%d, k=[%s], m=[%s], res=%d', ...
        params.L, vec_to_comma_string(params.k_list), ...
        vec_to_comma_string(params.m_list), ...
        params.use_residual);
end

function s = short_architecture_string(params)
    s = sprintf('L%d_k%s_m%s', ...
        params.L, ...
        strrep(vec_to_comma_string(params.k_list), ',', '-'), ...
        strrep(vec_to_comma_string(params.m_list), ',', '-'));
end


% ============================================================
% 14. Helper for comma-separated vector strings
% ============================================================
function s = vec_to_comma_string(v)
    v = v(:)';
    c = arrayfun(@num2str, v, 'UniformOutput', false);
    s = strjoin(c, ',');
end


% ============================================================
% 15. Helper for LaTeX titles
% ============================================================
function title_str = make_latex_title(theta, params)
    m_str = vec_to_comma_string(params.m_list);
    k_str = vec_to_comma_string(params.k_list);

    line1 = sprintf('$\\theta=%.3f$, $L=%d$', theta, params.L);
    line2 = sprintf('$m_{\\ell}=\\{%s\\}$, $k_{\\ell}=\\{%s\\}$', m_str, k_str);

    title_str = {line1, line2};
end