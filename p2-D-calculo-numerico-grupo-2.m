### ANÁLISE 1: SELEÇÃO DAS VARIÁVEIS ###

## ITENS (1.1) & (1.2) ##

# URL do arquivo CSV
url = 'https://raw.githubusercontent.com/VictorDaisukeAraqui/p2-D-calculo-numerico-grupo-2/refs/heads/main/healthy_lifestyle_city_2021_cleaned.csv';

# Caminho para salvar o arquivo localmente
local_file = 'healthy_lifestyle_city_2021_cleaned.csv';

# Baixar o arquivo
urlwrite(url, local_file);

# Carregar dados do arquivo
data = csvread(local_file);

# Resto do código permanece o mesmo
columns_names = {'Sunshine Hours', 'Cost Bottle of Water', ...
                 'Obesity Levels', 'Life Expectancy (years)', ...
                 'Pollution (Index Score)', 'Annual Avg. Hours Worked', ...
                 'Happiness Levels (Country)', 'Outdoor Activities', ...
                 'No. of Take Out Places', 'Monthly Gym Membership'};

# Remover linha com strings dos títulos e coluna com nomes das cidades
data(1, :) = [];
data(:, 1) = [];

# Definir vetor 'happiness_level' com os níveis de felicidade (coluna 7)
happiness_level = data(:, 7);

# Número de colunas a serem plotadas
num_columns = size(data, 2);

# Loop para gerar os gráficos de cada variável em relação ao nível de felicidade
for ii = 1:num_columns

    # Pular coluna de nível de felicidade (índice 7)
    if ii == 7

        continue;

    endif

    figure;

    # Normalizar os dados da coluna atual
    norm_column_data = (data(:, ii) - min(data(:, ii))) / (max(data(:, ii)) - min(data(:, ii)));

    # Filtrar e plotar cidades felizes e menos felizes
    scatter(happiness_level(happiness_level == 1), norm_column_data(happiness_level == 1), 'b', 'x');
    hold on;
    scatter(happiness_level(happiness_level == 0), norm_column_data(happiness_level == 0), 'r', 'o');

    # Configurar rótulos e legendas
    xlabel('Nível de Felicidade');
    ylabel(columns_names{ii});
    title(['Nível de Felicidade versus ', columns_names{ii}, ' (Normalizado)']);
    legend('Cidades Felizes (1)', 'Cidades Menos Felizes (0)');
    grid on;
    hold off;

endfor

## Variáveis para cidades mais felizes: 'Cost Bottle of Water' e 'Life Expectancy'
## Variáveis para cidades menos felizes: 'Sunshine Hours' e 'Pollution Index'

## ITEM (1.3) ##

# Calcular matriz de correlação para todas as variáveis
correlation_matrix = corr(data);

# Exibir matriz de correlação em um mapa de calor para melhor visualização
figure;
imagesc(correlation_matrix);
colorbar;
colormap(jet)
title('Matriz de Correlação entre Variáveis');
xticks(1:num_columns);
yticks(1:num_columns);
xticklabels(columns_names);
yticklabels(columns_names);

# Inicializar vetor para armazenar os coeficientes de correlação
correlations = zeros(1, num_columns);

# Calcular correlação entre cada variável e o nível de felicidade
for ii = 1:num_columns

    correlations(ii) = corr(data(:, ii), happiness_level);

endfor

# Exibir coeficientes de correlação
correlations

## Análise numérica (Coeficiente de Correlação de Pearson):

## Cost Bottle of Water: 0.6202
## Life Expectancy: 0.4523

## Sunshine Hours: -0.4383
## Pollution Index: -0.6550

### ANÁLISE 2: ANÁLISE DE REGRESSÃO EXPONENCIAL ###

## ITEM (2.1) ##

# Substituir valores binários da coluna 'Happiness Levels' pela nova coluna
happiness_replaced = [7.44; 7.22; 7.29; 7.35;
                      7.64; 7.80; 5.87; 7.07;
                      6.40; 7.23; 7.22; 5.12;
                      5.99; 5.97; 7.23; 6.40;
                      5.28; 5.87; 7.07; 7.56;
                      7.12; 5.13; 4.15; 5.12;
                      6.94; 3.57; 6.94; 7.09;
                      5.87; 6.94; 5.51; 5.12;
                      6.86; 6.94; 6.66; 6.37;
                      7.56; 7.16; 4.81; 6.38;
                      6.94; 6.94; 5.54; 6.46];

data(:, 7) = happiness_replaced;

# Filtrar dados para cidades menos felizes
data_less_happy_cities = data(happiness_level < 7, :);

# Percorre cada par de variáveis para gerar gráficos de dispersão
for i = 1:num_columns

        # Pular coluna de nível de felicidade (índice 7)
        if i == 7

            continue;

        endif

        # Criar gráficos de dispersão entre as variáveis filtradas
        figure;
        scatter(data_less_happy_cities(:, i), data_less_happy_cities(:, 7), 'r', 'o');
        xlabel(columns_names{i});
        ylabel('Happiness Level');
        title(['Dispersão de ', columns_names{i}, ' versus Happiness Level (Cidades Menos Felizes)']);
        grid on;

endfor

## Variável 'Life Expectancy (Years)' versuss Happiness Level (Cidades Menos Felizes)

## ITEM (2.2) & (2.3) ##

## MÉTODO 1

# Variável independente (Life Expectancy)
x = data_less_happy_cities(:, 4);

# Variável dependente transformada (logaritmo dos níveis de felicidade)
log_y = log(data_less_happy_cities(:, 7));

# Montar matriz de design (X) e o vetor de resposta (log_y)
X = [ones(length(x), 1), x];

# Decomposição LU
[L, U] = lu(X' * X);
log_y_transformed = X' * log_y;

# Resolução de Ly = b usando substituição direta
y_temp = linsolve(L, log_y_transformed);

# Resolução de Ux = y usando substituição retroativa
b = linsolve(U, y_temp);

# Transformar de volta para a forma exponencial
a = exp(b(1));
b_exp = b(2);

# Criar um vetor denso para x, para suavizar a linha de ajuste
x_dense = linspace(min(x), max(x), 100);

# Prever novos valores para x_dense
y_pred_dense = a * exp(b_exp * x_dense);

# Calcular valores previstos para x
y_pred = a * exp(b_exp * x);

# Calcular os resíduos
residuals = data_less_happy_cities(:, 7) - y_pred;

# Soma dos resíduos
sum_residuals = sum(residuals);

# Cálculo do erro padrão
std_error = std(residuals);

# Cálculo do coeficiente de determinação R^2
SStot = sum((data_less_happy_cities(:, 7) - mean(data_less_happy_cities(:, 7))).^2); % Soma total dos quadrados
SSres = sum(residuals.^2); % Soma dos quadrados dos resíduos
r_squared = 1 - (SSres / SStot); % Cálculo do R^2

# Exibir os resultados
fprintf('Soma dos Resíduos: %.4f\n', sum_residuals);
fprintf('Erro Padrão: %.4f\n', std_error);
fprintf('Coeficiente de Determinação R^2: %.4f\n', r_squared);

# Plotar a curva exponencial junto aos pontos de dados
figure;
scatter(x, data_less_happy_cities(:, 7), 'r', 'o');
hold on;
plot(x_dense, y_pred_dense, 'b-', 'LineWidth', 2);
xlabel('Life Expectancy (years)');
ylabel('Happiness Levels');
title('Curva de Regressão Exponencial versus Happiness Levels (Cidades Menos Felizes)');
legend('Dados Originais', 'Ajuste Exponencial');
grid on;
hold off;

## MÉTODO 2

% Variável independente (Life Expectancy)
x = data_less_happy_cities(:, 4);  % Expectativa de vida

% Variável dependente (Níveis de felicidade)
y = data_less_happy_cities(:, 7);

% Transformar os dados de y
y_log = log(y);  % Aplicar logaritmo natural

% Ajustar uma regressão linear
n = length(x);
sum_x = sum(x);
sum_y_log = sum(y_log);
sum_xy_log = sum(x .* y_log);
sum_x2 = sum(x .^ 2);

% Calcular os coeficientes da regressão linear
b = (n * sum_xy_log - sum_x * sum_y_log) / (n * sum_x2 - sum_x^2);
a_log = mean(y_log) - b * mean(x);

% Transformar a para a forma original
a = exp(a_log);  % Exponenciar o coeficiente

% Prever os valores ajustados
y_fit = a * exp(b * x);  % Função original

% Calcular as métricas
residuals = y - y_fit;  % Cálculo dos resíduos
SSR = sum(residuals .^ 2);  % Soma dos quadrados dos resíduos

% Cálculo do R^2
SS_tot = sum((y - mean(y)) .^ 2);  % Soma total dos quadrados
R2 = 1 - (SSR / SS_tot);  % Cálculo do R^2

% Erro Padrão da Estimativa
Sy_x = sqrt(SSR / (n - 2));  % Erro padrão

% Criar um vetor denso para x, para suavizar a linha de ajuste
x_dense = linspace(min(x), max(x), 100); % 100 pontos igualmente espaçados entre o mínimo e o máximo de x
y_fit_dense = a * exp(b * x_dense);  % Prever novos valores

% Exibir resultados
disp(['Soma de Resíduos: ', num2str(SSR)]);
disp(['Erro Padrão: ', num2str(Sy_x)]);
disp(['Coeficiente de Determinação R^2: ', num2str(R2)]);

% Plotar a curva exponencial junto aos pontos de dados
figure;
scatter(x, y, 'r', 'o');  % Pontos originais
hold on;
plot(x_dense, y_fit_dense, 'b-', 'LineWidth', 2);  % Curva de regressão exponencial suavizada
xlabel('Life Expectancy (years)');
ylabel('Happiness Levels');
title('Curva de Regressão Exponencial vs Happiness Levels (Cidades Menos Felizes)');
legend('Dados Originais', 'Ajuste Exponencial');
grid on;
hold off;

### ANÁLISE 3: PREDIÇÃO ###

## ITEM (3.1) & (3.3) ##

# GRÁFICO ENTRE EXPECTATIVA DE VIDA E FELICIDADE #
# definir as colunas necessárias
x = data(:, 4); % Expectativa
y = data(:, 7); % Felicidade

# Utilizar formulas para a regressão
n = length(x);
sum_x = sum(x);
sum_y = sum(y);
sum_xy = sum(x .* y);
sum_x2 = sum(x .^ 2);

b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2);
a = mean(y) - b * mean(x);

# Regressão linear calculada
y_fit = a + b * x;

# Calcular métricas de qualidade do ajuste
residuals = y - y_fit;
SS_res = sum(residuals .^ 2);
SS_tot = sum((y - mean(y)) .^ 2);
R2 = 1 - (SS_res / SS_tot);
RMSE = sqrt(mean(residuals .^ 2)); % erro médio
MAE = mean(abs(residuals)); % erro absoluto
g = length(y);
h = size(x,2);
t = sqrt(SS_res / (g - h));

fprintf('Ajuste (expectativa): y1 = %.2f + %.2f * x1\n', a, b);
fprintf('Sr: %.2f\n', SS_res);
fprintf('R^2: %.2f\n', R2);
fprintf('Sy/x: %.2f\n', t);
disp('                     ')

# Plotar os dados e a linha ajustada
figure;
scatter(x, y, 'r', 'x');
hold on;
plot(x, y_fit, '-m');
xlabel('Expectativa de Vida');
ylabel('Nível de Felicidade');
title('Regressão Linear: Expectativa X Níveis de Felicidade');
legend('Dados', 'Ajuste Linear');
grid on;

# Adicionar equação no centro do gráfico
x_position = mean(x); % Posição x do texto (média dos valores de x)
y_position = mean(y); % Posição y do texto (média dos valores de y)
text(x_position, y_position, sprintf('y = %.2f + %.2f * x1', a, b), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'center')
hold off;

## GRÁFICO ENTRE ÁGUA E FELICIDADE
# Definir as colunas relevantes
w = data(:, 2); % Água
z = data(:, 7); % Felicidade

# Calcular os coeficientes da regressão linear manualmente
n = length(w);
sum_w = sum(w);
sum_z = sum(z);
sum_zw = sum(z .* w);
sum_w2 = sum(w .^ 2);

d = (n * sum_zw - sum_w * sum_z) / (n * sum_w2 - sum_w^2);
c = mean(z) - d * mean(w);

# Calcular os valores ajustados
y_fit1 = c + d * w;

# Calcular métricas de qualidade do ajuste
residuals1 = z - y_fit1;
SS_res1 = sum(residuals1 .^ 2);
SS_tot1 = sum((z - mean(z)) .^ 2);
R21 = 1 - (SS_res1 / SS_tot1);
RMSE1 = sqrt(mean(residuals1 .^ 2));
MAE1 = mean(abs(residuals1));

o = length(z);
d = size(x,2);
e = sqrt(SS_res1 / (o - d));

fprintf('Ajuste (garrafa de água): y2 = %.2f + %.2f * x2\n', c, d);
fprintf('Sr: %.2f\n', SS_res1)
fprintf('R^2: %.2f\n', R21);
fprintf('Sy/x: %.2f\n', e);
disp('                     ')

# Plotar os dados e a linha ajustada
figure;
scatter(w, z, 'm', 'o');
hold on;
plot(w, y_fit1, '-g');
xlabel('Custo da Garrafa de Água');
ylabel('Nível de Felicidade');
title('Regressão Linear: Água X Níveis de Felicidade');
legend('Dados', 'Ajuste Linear');
grid on;

# Adicionar equação no centro do gráfico
w_position = mean(w); % Posição x do texto (média dos valores de x)
z_position = mean(z); % Posição y do texto (média dos valores de y)
text(w_position, z_position, sprintf('y = %.2f + %.2f * x2', c, d), 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'center');

## ITEM (3.2) & (3.3) ##

# Definir as variáveis
k = data(:, 4); % Expectativa
u = data(:, 7); % Felicidade
s = data(:, 2); % Água

# Criar a matriz de design
N = length(u);
X = [ones(N, 1), k, s];

# Resolução do sistema usando a pseudo-inversa para obter os coeficientes
beta = pinv(X) * u;

# Calcular as previsões
u_fit = X * beta;

# Calcular os resíduos
residuals2 = u - u_fit;

# Calcular as métricas
S_r = sum(residuals2); % Soma dos Resíduos
SS_total = sum((u - mean(u)).^2);
SS_residual = sum(residuals2.^2);
R_squared = 1 - (SS_residual / SS_total);
n = length(u);
p = length(beta);
s_y_x = sqrt(SS_residual / (n - p));

% Extrair coeficientes
coefficients = beta; % Atribuir os coeficientes calculados
a0 = coefficients(1); % Intercepto
a1 = coefficients(2); % Coeficiente para k
a2 = coefficients(3); % Coeficiente para s

# Exibir os resultados
fprintf('Ajuste: (Regressão Múltipla) y = %.2f + %.2f*x1 + %.2f*x2\n', a0, a1, a2);
fprintf('Sr: %.2f\n', S_r);
fprintf('R^2: %.2f\n', R_squared);
fprintf('Sy/x: %.2f\n', s_y_x);
disp('                       ')

# Plotar dados e plano de regressão
figure;
scatter3(k, s, u, 'b', 'filled'); % Gráfico de dispersão dos dados reais
hold on;
[X1, X2] = meshgrid(linspace(min(k), max(k), 10), linspace(min(s), max(s), 10));
Y_pred = a0 + a1*X1 + a2*X2; % Superfície prevista

# Gráfico da superfície de regressão
mesh(X1, X2, Y_pred);
xlabel('Expectativa de Vida');
ylabel('Custo da Garrafa de Água');
zlabel('Nível de Felicidade');
title(sprintf('Plano de Regressão: y = %.2f + %.2f*x1 + %.2f*x2', a0, a1, a2));
legend('Dados reais', 'Plano de regressão');
grid on;
hold off;

disp('Melhor Modelo é o Múltiplo !!');
