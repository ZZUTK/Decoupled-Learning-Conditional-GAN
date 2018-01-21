%% compute Normalized Relative Discriminator Score (NRSD)
clc; clear; close all

%% load data
data_path = './save/mat';
files = dir(fullfile(data_path, '*.mat'));
curve_real = [];
curve_adv1 = [];
curve_adv1e2 = [];
curve_adv1e3 = [];
curve_adv1e4 = [];
for i = 1:length(files)
    file = files(i).name;
    load(fullfile(data_path, file))
    curve_real = [ curve_real ; [mean(real), std(real)] ]; 
    curve_adv1 = [ curve_adv1 ; [mean(adv1), std(adv1)] ]; 
    curve_adv1e2 = [ curve_adv1e2 ; [mean(adv1e2), std(adv1e2)] ]; 
    curve_adv1e3 = [ curve_adv1e3 ; [mean(adv1e3), std(adv1e3)] ]; 
    curve_adv1e4 = [ curve_adv1e4 ; [mean(adv1e4), std(adv1e4)] ]; 
end

%% plot
figure; hold on
plot(1:epochs, curve_real(:,1), '-', 'linewidth', 2)
plot(1:epochs, curve_adv1(:,1), '--', 'linewidth', 2)
plot(1:epochs, curve_adv1e2(:,1), ':', 'linewidth', 2)
plot(1:epochs, curve_adv1e3(:,1), '-.', 'linewidth', 2)
plot(1:epochs, curve_adv1e4(:,1), '-', 'linewidth', 2)
legend('real', 'adv1', 'adv1e-2', 'adv1e-3', 'adv1e-4', 'location', 'best')
xlabel('Epoch')
ylabel('Avg. output of discriminator')
grid on
set(gca, 'fontsize', 16)

%% compute area under the curves
a_real = trapz(curve_real(:,1));
a_adv1 = trapz(curve_adv1(:,1));
a_adv1e2 = trapz(curve_adv1e2(:,1));
a_adv1e3 = trapz(curve_adv1e3(:,1));
a_adv1e4 = trapz(curve_adv1e4(:,1));

din = a_adv1 + a_adv1e2 + a_adv1e3 + a_adv1e4;

score = nan(4, 1);
score(1) = a_adv1 / din;
score(2) = a_adv1e2 / din;
score(3) = a_adv1e3 / din;
score(4) = a_adv1e4 / din;

score









