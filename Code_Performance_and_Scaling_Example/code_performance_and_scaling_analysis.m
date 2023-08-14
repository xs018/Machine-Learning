close all; clear; clc

%% XSEDE Research Renewal 2021

n_ideal = (1:1:1000);
speedup_ideal = n_ideal;

% CNN: ResNet-50
figure
X = categorical({'1 GPU','2 GPUs','3 GPUs','4 GPUs','8 GPUs'});

ResNet_time = [798.68 834.186 841.329; 468.716 463.153 421.232; 485.995 474.437 316.075; 506.131 275.526 272.461; 249.56 222.638 187.621] / 10;

bar(X,ResNet_time);
ylabel('Computational time (s/epoch)')
legend('batch size = 256','batch size = 512','batch size = 1024')
set(gca,'FontSize',16);

n_gpu = [1 2 3 4 8];
resnet_a = [798.68 468.716 485.995 506.131 249.56];
resnet_b = [834.186 463.153 474.437 275.526 222.638];
resnet_c = [841.329 421.232 316.075 272.461 187.621];

resnet_a_speedup = resnet_a(1) ./ resnet_a;
resnet_b_speedup = resnet_b(1) ./ resnet_b;
resnet_c_speedup = resnet_c(1) ./ resnet_c;

resnet_a_efficiency = resnet_a_speedup ./ n_gpu;
resnet_b_efficiency = resnet_b_speedup ./ n_gpu;
resnet_c_efficiency = resnet_c_speedup ./ n_gpu;

figure
plot(n_gpu,resnet_a_speedup,'s--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_gpu,resnet_b_speedup,'^--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_gpu,resnet_c_speedup,'d--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of GPUs')
ylabel('Speedup')
legend('batch size = 256','batch size = 512','batch size = 1024')
set(gca,'FontSize',16);
xlim([1 8])

figure
plot(n_gpu,resnet_a_efficiency * 100,'s--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_gpu,resnet_b_efficiency * 100,'^--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_gpu,resnet_c_efficiency * 100,'d--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of GPUs')
ylabel('Parallel efficiency')
legend('batch size = 256','batch size = 512','batch size = 1024')
set(gca,'FontSize',16);
xlim([1 8])
ylim([0 100])

% CNN and ConvLSTM: GPU vs CPU
figure
X = categorical({'Xeon E2286M','Bridges2 GPU-AI'});
ConvLSTM_time = [1894.499 55.50];
CNN_time = [2600.8496 54.9368];
Time = [2600.8496 1894.499; 54.9368 55.50];
Y = 3600 ./ Time;

% Y = [1.3842    1.9002; 65.5298   64.8649];
b = bar(X,Y);
ylabel('Training Speed (epochs/hr)')
legend('CNN','ConvLSTM')
set(gca,'FontSize',16);

% MLP & CNN: Speedup vs. Number of GPUs
n_gpu = [1 2 3 4 8];
mlp_time = [846.387 377.168 292.789 276.854 216.178];
mlp_speedup = mlp_time(1) ./ mlp_time;

cnn_time = [802.558 468.716 492.066 478.524 478.524];
cnn_speedup = cnn_time(1) ./ cnn_time;

figure
% plot(n_gpu,cnn_speedup,'sb--','LineWidth',1.5,'MarkerSize',8);
% hold on
plot(n_gpu,mlp_speedup,'^r--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of GPUs')
ylabel('Speedup')
legend('MLP','Ideal speedup')
set(gca,'FontSize',16);
xlim([1 8])

mlp_efficiency = mlp_speedup ./ n_gpu;
figure
plot(n_gpu,mlp_efficiency * 100,'r^--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of GPUs')
ylabel('Parallel efficiency')
set(gca,'FontSize',16);
xlim([1 8])
ylim([0 115])

% DFT
n_cpu = [1 2 4 8 16 32];
dft_time = [4197 2774 1463 1365 622 447];
dft_speedup = dft_time(1) ./ dft_time;
figure
plot(n_cpu,dft_speedup,'sb--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of CPU cores')
ylabel('Speedup')
legend('DFT','Ideal speedup')
set(gca,'FontSize',16);
xlim([1 32])

dft_efficiency = dft_speedup ./ n_cpu * 100;
figure
plot(n_cpu,dft_efficiency,'sb--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Fixed-Size Efficiency')
set(gca,'FontSize',16);
xlim([1 32])

test = importdata('test.txt');
a(:,1) = test.data(:,1) ./ 20.052889;
a(:,2) = test.data(:,2) ./ 20.052889;
a(:,3) = test.data(:,3) ./ 28.650528;

%MD

md_hea_0_stampede2_core = [24 48 96 192 384 768];
md_hea_0_scaled_stampede2_time = [170 164 214 208 225 280];
md_hea_0_fixed_stampede2_time = [1699 835 436 209 115 79];
md_hea_0_fixed_stampede2_speedup = md_hea_0_fixed_stampede2_time(1) ./ md_hea_0_fixed_stampede2_time * 24;

figure
% subplot(1,2,1)
plot(md_hea_0_stampede2_core,md_hea_0_fixed_stampede2_speedup,'or--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of CPU cores')
ylabel('Speedup')
legend('MD: HAP 0K','Ideal speedup')
set(gca,'FontSize',16);
xlim([0 800])

md_hea_0_fixed_stampede2_efficiency = md_hea_0_fixed_stampede2_speedup ./ md_hea_0_stampede2_core;
% subplot(1,2,2)
figure
plot(md_hea_0_stampede2_core,md_hea_0_fixed_stampede2_efficiency * 100,'or--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Fixed-Size Efficiency')
set(gca,'FontSize',16);
ylim([0 105])
md_hea_0_scaled_stampede2_efficiency = md_hea_0_scaled_stampede2_time(1) ./ md_hea_0_scaled_stampede2_time;

% figure 
% X = categorical({'24 CPUs','48 CPUs','96 CPUs','192 CPUs','384 GPUs','768 GPUs'});
% bar(X,md_hea_0_scaled_stampede2_time);
% ylabel('Computational time (s)')
% set(gca,'FontSize',16);

figure
plot(md_hea_0_stampede2_core,md_hea_0_scaled_stampede2_efficiency * 100,'or--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Scaled-Size Efficiency')
set(gca,'FontSize',16);
ylim([0 105])


md_hea_0_bridges2_core = [16 32 64 96 128];
md_hea_0_scaled_bridges2_time = [24 30 30 50 64];
md_hea_0_fixed_bridges2_time = [524 244 145 142 135];
md_hea_0_fixed_bridges2_speedup = md_hea_0_fixed_bridges2_time(1) ./ md_hea_0_fixed_bridges2_time * 16;
md_hea_0_fixed_bridges2_efficiency = md_hea_0_fixed_bridges2_speedup ./ md_hea_0_bridges2_core;
md_hea_0_scaled_bridges2_efficiency = md_hea_0_scaled_bridges2_time(1) ./ md_hea_0_scaled_bridges2_time;

md_hea_300_bridges2_core = [16 32 64 96 128];
md_hea_300_scaled_bridges2_time = [17 18 19 26 37];
md_hea_300_fixed_bridges2_time = [460 232 108 83 85];
md_hea_300_fixed_bridges2_speedup = md_hea_300_fixed_bridges2_time(1) ./ md_hea_300_fixed_bridges2_time * 16;
md_hea_300_fixed_bridges2_efficiency = md_hea_300_fixed_bridges2_speedup ./ md_hea_300_bridges2_core;
md_hea_300_scaled_bridges2_efficiency = md_hea_300_scaled_bridges2_time(1) ./ md_hea_300_scaled_bridges2_time;


figure
% subplot(1,2,1)
plot(md_hea_0_bridges2_core,md_hea_0_fixed_bridges2_speedup,'bs--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(md_hea_300_bridges2_core,md_hea_300_fixed_bridges2_speedup,'r^--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of CPU cores')
ylabel('Speedup')
legend('MD: HAP 0K','MD: HAP 300K','Ideal speedup')
set(gca,'FontSize',16);
xlim([0 150])


% subplot(1,2,2)
figure
plot(md_hea_0_bridges2_core,md_hea_0_fixed_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(md_hea_300_bridges2_core,md_hea_300_fixed_bridges2_efficiency * 100,'r^--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Fixed-Size Efficiency')
legend('MD: HAP 0K','MD: HAP 300K')
set(gca,'FontSize',16);
ylim([0 115])


figure
plot(md_hea_0_bridges2_core,md_hea_0_scaled_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(md_hea_300_bridges2_core,md_hea_300_scaled_bridges2_efficiency * 100,'r^--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Scaled-Size Efficiency')
legend('MD: HAP 0K','MD: HAP 300K')
set(gca,'FontSize',16);
ylim([0 105])



figure
% % subplot(1,2,1)
% plot(md_hea_300_bridges2_core,md_hea_300_fixed_bridges2_speedup,'bs--','LineWidth',1.5,'MarkerSize',8);
% hold on
% plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
% xlabel('Number of CPU cores')
% ylabel('Speedup')
% legend('MD: HAP 300K','Ideal speedup')
% set(gca,'FontSize',16);
% xlim([0 150])
% 
% % subplot(1,2,2)
% figure
% plot(md_hea_300_bridges2_core,md_hea_300_fixed_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
% ytickformat('percentage')
% xlabel('Number of CPU cores')
% ylabel('Fixed-Size Efficiency')
% set(gca,'FontSize',16);
% ylim([0 115])
% 
% figure
% plot(md_hea_300_bridges2_core,md_hea_300_scaled_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
% ytickformat('percentage')
% xlabel('Number of CPU cores')
% ylabel('Scaled-Size Efficiency')
% set(gca,'FontSize',16);
% ylim([0 105])

md_FeNi_bridges2_core = [16 32 64 128 256 512];
md_FeNi_scaled_bridges2_time = [41 41 42 49 51 81];
md_FeNi_fixed_bridges2_time = [1261 637 330 193 109 67];
md_FeNi_fixed_bridges2_speedup = md_FeNi_fixed_bridges2_time(1) ./ md_FeNi_fixed_bridges2_time * 16;

figure
% subplot(1,2,1)
plot(md_FeNi_bridges2_core,md_FeNi_fixed_bridges2_speedup,'bs--','LineWidth',1.5,'MarkerSize',8);
hold on
plot(n_ideal,speedup_ideal,'k--','LineWidth',1.5);
xlabel('Number of CPU cores')
ylabel('Speedup')
legend('MD: FeNi','Ideal speedup')
set(gca,'FontSize',16);
xlim([0 550])

md_FeNi_fixed_bridges2_efficiency = md_FeNi_fixed_bridges2_speedup ./ md_FeNi_bridges2_core;
% subplot(1,2,2)
figure
plot(md_FeNi_bridges2_core,md_FeNi_fixed_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Fixed-Size Efficiency')
set(gca,'FontSize',16);
xlim([0 550])
ylim([0 105])
md_FeNi_scaled_bridges2_efficiency = md_FeNi_scaled_bridges2_time(1) ./ md_FeNi_scaled_bridges2_time;

figure
plot(md_FeNi_bridges2_core,md_FeNi_scaled_bridges2_efficiency * 100,'sb--','LineWidth',1.5,'MarkerSize',8);
ytickformat('percentage')
xlabel('Number of CPU cores')
ylabel('Scaled-Size Efficiency')
set(gca,'FontSize',16);
xlim([0 550])
ylim([0 105])
%% XSEDE Research New 2020
% figure
% 
% X = categorical({'DS-1','DS-2','DS-3'});
% % X = reordercats(X,{'DS-1','DS-2','DS-3'});
% Y = [1 1; 0.57 0.43; 0.28 0.11];
% b = bar(X,Y);
% 
% % text([1:length(Y)], Y', num2str(Y','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
% % text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center'); 
% ylabel('Accuracy')
% legend('ConvLSTM','CNN')
% set(gca,'FontSize',16);
% 
% figure
% X = categorical({'Xeon E2286M','Bridges GPU','Bridges GPU-AI'});
% X = reordercats(X,{'Xeon E2286M','Bridges GPU','Bridges GPU-AI'});
% 
% Time = [1894.499 224.35 55.50];
% Y = 3600 ./ Time;
% % Y = [1 1; 0.57 0.43; 0.28 0.11];
% b = bar(X,Y);
% 
% % text([1:length(Y)], Y', num2str(Y','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
% % text(1:length(Y),Y,num2str(Y'),'vert','bottom','horiz','center'); 
% ylabel('Training Speed (epochs/hr)')
% 
% % legend('LSTM','CNN')
% set(gca,'FontSize',16);

% xtips1 = b(1).XEndPoints;
% ytips1 = b(1).YEndPoints;
% labels1 = string(b(1).YData);
% text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')
% xtips2 = b(2).XEndPoints;
% ytips2 = b(2).YEndPoints;
% labels2 = string(b(2).YData);
% text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
%     'VerticalAlignment','bottom')

% GT = [1 1.5 2 5 10];
% t_CNN = [848	835	835	841	852	851	850	849	852	852	855	860	863	868	872;
%     883	877	887	879	887	879	893	886	887	892	890	900	899	899	896;
%     899	906	912	917	917	916	927	920	924	923	926	930	929	923	929;
%     913	912	916	918	926	922	918	921	931	928	928	929	937	937	930;
%     963	960	957	965	967	970	967	975	982	981	983	988	984	980	983];
% t_TL = [103	102	103	103	104	104	105	105	106	107	107	108	108	109	110;
%     148	153	147	160	150	151	157	154	157	168	165	156	163	163	160;
%     246	246	247	249	250	250	250	251	252	254	254	256	255	257	257;
%     663	663	661	665	667	665	669	672	671	670	673	673	675	676	676;
%     1300	1308	1310	1309	1310	1310	1342	1357	1324	1329	1333	1331	1334	1343	1341];
% 
% epoch_CNN = [20 35 27 39 34 37 46 56 31 39 45 45 30 36 49;
%     44 43 32 48 39 29 58 63 44 42 44 46 50 56 53;
%     21 57 45 50 25 32 33 50 28 39 47 42 51 36 28;
%     43 32 26 61 49 31 36 55 35 48 36 35 30 39 43;
%     36 49 57 57 42 45 74 52 31 51 80 56 65 26 45];
% 
% epoch_TL = [834 757 886 699 992 964 991 869 925 703 864 974 964 902 885
%     968 928 816 879 493 992 1000 789 666 961 894 663 992 995 962;
%     888 982 983 643 864 995 983 872 605 955 983 939 684 914 885;
%     606 327 600 444 378 225 321 558 663 417 417 636 501 456 435;
%     392 417 326 477 331 300 414 379 330 337 456 392 387 403 398];
% 
% t_CNN_mean = mean(t_CNN,2);
% t_TL_mean = mean(t_TL,2);
% epoch_CNN_mean = mean(epoch_CNN,2);
% epoch_TL_mean = mean(epoch_TL,2);
% 
% t_CNN_std = std(t_CNN,0,2);
% t_TL_std = std(t_TL,0,2);
% epoch_CNN_std = std(epoch_CNN,0,2);
% epoch_TL_std = std(epoch_TL,0,2);
% 
% GT_ref = (0:1:12);
% t_TL_fitting = 136.31 * GT_ref - 29.87;
% 
% figure
% ax = axes;
% errorbar(GT,t_CNN_mean,t_CNN_std,'bs');
% hold on
% errorbar(GT,t_TL_mean,t_TL_std,'r^');
% xtickformat(ax, 'percentage');
% xlabel('Percentage of DS2 data for training');
% ylabel('Training time for best model, \it\rm (s)');
% 
% hold on
% plot(GT_ref,t_TL_fitting,'r--','LineWidth',1);
% legend('CNN','TL');
% set(gca,'FontSize',15);
% xlim([0 12]);
% ylim([0 1400])
% 
% figure
% ax = axes;
% errorbar(GT,epoch_CNN_mean,epoch_CNN_std,'bs');
% hold on
% errorbar(GT,epoch_TL_mean,epoch_TL_std,'r^');
% xtickformat(ax, 'percentage');
% xlabel('Percentage of DS2 data for training');
% ylabel('Epoch for best model');
% legend('CNN','TL');
% set(gca,'FontSize',15);
% xlim([0 12]);
% 
% % t_TL_1000 = [5*60+41 6*60+6 5*60+51 5*60+37 5*60+44; 15*60+9 15*60+16 15*60+5 15*60+5 15*60+8;32*60+12 32*60+27 32*60+40 32*60+6 31*60+41];
% % t_TL_1000_mean = mean(t_TL_1000,2);
% % epoch_TL_data = [816 974 974 930 781; 198 72 168 376 128;282 306 222 297 324];
% % epoch_TL = mean(epoch_TL_data,2);
% % t_TL = t_TL_1000_mean .* epoch_TL / 1000;
% % t_CNN_data = [2695 2337 5103 4420 3375; 3159 4131 3969 2511 2640;2765 2686 3713 4740 2923];
% % t_CNN = mean(t_CNN_data,2);
% % epoch_CNN_data = [35 30 63 54 43; 39 52 49 31 33; 35 34 47 60 37];
% % epoch_CNN = mean(epoch_CNN_data,2);
% % 
% % t_TL_matrix = t_TL_1000 / 1000 .* epoch_TL_data;
% % t_TL_mean = mean(t_TL_matrix,2);
% % t_TL_std = std(t_TL_matrix,0,2);
% % epoch_TL_mean = mean(epoch_TL_data,2);
% % epoch_TL_std = std(epoch_TL_data,0,2);
% % 
% % t_CNN_mean = mean(t_CNN_data,2);
% % t_CNN_std = std(t_CNN_data,0,2);
% % epoch_CNN_mean = mean(epoch_CNN_data,2);
% % epoch_CNN_std = std(epoch_CNN_data,0,2);
% % 
% % figure
% % ax = axes;
% % errorbar(GT,t_CNN_mean,t_CNN_std,'bs');
% % hold on
% % errorbar(GT,t_TL_mean,t_TL_std,'r^');
% % % hold on
% % % plot(0,0.3445,'ko','LineWidth',1);
% % xtickformat(ax, 'percentage');
% % xlabel('Percentage of GT data for training');
% % ylabel('Computational time, \it\rm (s)');
% % legend('CNN','TL');
% % set(gca,'FontSize',15);
% % xlim([0 12]);
% % 
% % figure
% % ax = axes;
% % errorbar(GT,epoch_CNN_mean,epoch_CNN_std,'bs');
% % % errorbar(GT,epoch_TL_mean,epoch_TL_std,'LineStyle','None');
% % % hold on 
% % % plot(GT,epoch_TL_mean,'bs','LineWidth',1.5);
% % hold on
% % errorbar(GT,epoch_TL_mean,epoch_TL_std,'r^');
% % 
% % % hold on
% % % plot(0,0.3445,'ko','LineWidth',1);
% % xtickformat(ax, 'percentage');
% % xlabel('Percentage of GT data for training');
% % ylabel('Epoch for best model');
% % legend('CNN','TL');
% % set(gca,'FontSize',15);
% % xlim([0 12]);
% % 
% % % xlim([0 15]);
% % 
% % % figure
% % % ax = axes;
% % % plot(GT_2,epoch_CNN,'bs','LineWidth',1);
% % % 
% % % hold on
% % % plot(GT_2,epoch_TL,'r^','LineWidth',1);
% % % 
% % % xtickformat(ax, 'percentage');
% % % xlabel('Percentage of GT data for training');
% % % ylabel('Epoch for the best model');
% % % legend('CNN','TL');
% % % 
% % % set(gca,'FontSize',15);
% % % 
% % % figure
% % % ax = axes;
% % % plot(GT_2,t_CNN,'bs','LineWidth',1);
% % % 
% % % hold on
% % % plot(GT_2,t_TL,'r^','LineWidth',1);
% % % 
% % % xtickformat(ax, 'percentage');
% % % xlabel('Percentage of GT data for training');
% % % ylabel('Computational time (s)');
% % % legend('CNN','TL');
% % % 
% % % set(gca,'FontSize',15);