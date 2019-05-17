% Script for obtaining the required results for IROS'18
clear all
close all
filter = 1;
original = 1;
if filter == 1 && original == 1
    base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtrados/';
elseif filter == 1 && original == 0
    base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtradosNuevo/';
else
    base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/noFiltrados/';
end
gt_neigh = 40;
compensate = false;

% Configuring subpaths
addpath('AcademicFigures/');
agentsNumber = 25;
step = 5;

% Obtaining CityCenter results
% curr_dir = strcat(base_dir, 'CityCentre/')
% [PR_CCPurg] = process(curr_dir, agentsNumber, 'CityCentre', 5, agentsNumber, gt_neigh, compensate, step);
% % imgvstime_CC.time = smooth(imgvstime_CC.time);
% 
% curr_dir = strcat(base_dir, 'Lip6In/')
% [PR_L6IPurg] = process(curr_dir, agentsNumber, 'Lip6In', 5, agentsNumber, gt_neigh, compensate, step);
% % imgvstime_L6I.time = smooth(imgvstime_L6I.time);
% 
% curr_dir = strcat(base_dir, 'Lip6Out/')
% [PR_L6OPurg] = process(curr_dir, agentsNumber, 'Lip6Out', 5, agentsNumber, gt_neigh, compensate, step);
% imgvstime_L6O.time = smooth(imgvstime_L6O.time);
% 
curr_dir = strcat(base_dir, 'KITTI00/')
[PR_K0Purg] = process(curr_dir, agentsNumber, 'KITTI00', 5, agentsNumber, gt_neigh, compensate, step);
% imgvstime_K0.time = smooth(imgvstime_K0.time);
% 
% curr_dir = strcat(base_dir, 'KITTI05/')
% [PR_K5Purg] = process(curr_dir, agentsNumber, 'KITTI05', 70, agentsNumber, gt_neigh, compensate, step);
% imgvstime_K5.time = smooth(imgvstime_K5.time);

% curr_dir = strcat(base_dir, 'KITTI06/');
% [PR_K6, imgvssize_K6, imgvstime_K6] = process(curr_dir, gt_neigh, compensate);
% imgvstime_K6.time = smooth(imgvstime_K6.time);

% P/R curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_L6IPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_L6IPurg(i).R, PR_L6IPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i));
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'Lip6IPurg'));
% if filter == 1 && original == 1
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGItrad.png');
% elseif filter == 1 && original == 0
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGInew.png');
% else
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGInonFiltered.png');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_L6OPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_L6OPurg(i).R, PR_L6OPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'Lip6OPurg'));
% if filter == 1 && original == 1
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOtrad.png');
% elseif filter == 1 && original == 0
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOnew.png');
% else
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOnonFiltered.png');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_CCPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_CCPurg(i).R, PR_CCPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'CCPurg'));
% if filter == 1 && original == 1
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCtrad.png');
% elseif filter == 1 && original == 0
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCnew.png');
% else
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCnonFiltered.png');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
PRSize = size(PR_K0Purg);

for i = 1:agentsNumber
    plot(PR_K0Purg(i).R, PR_K0Purg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
end
legend show
xlabel('Recall');
ylabel('Precision');
xlim([0, 1]);
ylim([0.4, 1.02]);
hold off;
print('-depsc', strcat(base_dir, 'K0Purg'));
% if filter == 1 && original == 1
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0trad.png');
% elseif filter == 1 && original == 0
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0new.png');
% else
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0nonFiltered.png');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_K5Purg);
% 
% for i = 1:agentsNumber
%     plot(PR_K5Purg(i).R, PR_K5Purg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'K5Purg'));
% if filter == 1 && original == 1
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5trad.png');
% elseif filter == 1 && original == 0
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5new.png');
% else
%     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5nonFiltered.png');
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% print('-depsc', strcat(base_dir, 'PR_curves'));

% if filter == 1 && original == 1
%     base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtrados/';
% elseif filter == 1 && original == 0
%     base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtradosNuevo/';
% else
%     base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/noFiltrados/';
% end
% 
% % Obtaining CityCenter results
% curr_dir = strcat(base_dir, 'CityCentre/')
% [PR_CCnPurg] = process(curr_dir, agentsNumber, 'CityCentre', 5, agentsNumber, gt_neigh, compensate, step);
% % imgvstime_CC.time = smooth(imgvstime_CC.time);
% 
% % curr_dir = strcat(base_dir, 'NewCollege/');
% % [PR_NC, imgvssize_NC, imgvstime_NC] = process(curr_dir, gt_neigh, compensate);
% % imgvstime_NC.time = smooth(imgvstime_NC.time);
% 
% curr_dir = strcat(base_dir, 'Lip6In/')
% [PR_L6InPurg] = process(curr_dir, agentsNumber, 'Lip6In', 5, agentsNumber, gt_neigh, compensate, step);
% % imgvstime_L6I.time = smooth(imgvstime_L6I.time);
% 
% curr_dir = strcat(base_dir, 'Lip6Out/')
% [PR_L6OnPurg] = process(curr_dir, agentsNumber, 'Lip6Out', 5, agentsNumber, gt_neigh, compensate, step);
% % imgvstime_L6O.time = smooth(imgvstime_L6O.time);
% % 
% % curr_dir = strcat(base_dir, 'KITTI00/')
% % [PR_K0] = process(curr_dir, agentsNumber, 'KITTI00', 5, agentsNumber, gt_neigh, compensate);
% % imgvstime_K0.time = smooth(imgvstime_K0.time);
% % 
% % curr_dir = strcat(base_dir, 'KITTI05/')
% % [PR_K5] = process(curr_dir, agentsNumber, 'KITTI05', 5, agentsNumber, gt_neigh, compensate);
% % imgvstime_K5.time = smooth(imgvstime_K5.time);
% 
% % curr_dir = strcat(base_dir, 'KITTI06/');
% % [PR_K6, imgvssize_K6, imgvstime_K6] = process(curr_dir, gt_neigh, compensate);
% % imgvstime_K6.time = smooth(imgvstime_K6.time);
% 
% % P/R curves
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_L6InPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_L6InPurg(i).R, PR_L6InPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i));
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'Lip6InPurg'));
% % if filter == 1 && original == 1
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGItrad.png');
% % elseif filter == 1 && original == 0
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGInew.png');
% % else
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGInonFiltered.png');
% % end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_L6OnPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_L6OnPurg(i).R, PR_L6OnPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'Lip6OnPurg'));
% % if filter == 1 && original == 1
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOtrad.png');
% % elseif filter == 1 && original == 0
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOnew.png');
% % else
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/LGOnonFiltered.png');
% % end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% hold on;
% PRSize = size(PR_CCnPurg);
% 
% for i = 1:agentsNumber
%     plot(PR_CCnPurg(i).R, PR_CCnPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% end
% legend show
% xlabel('Recall');
% ylabel('Precision');
% xlim([0, 1]);
% ylim([0.4, 1.02]);
% hold off;
% print('-depsc', strcat(base_dir, 'CCnPurg'));
% % if filter == 1 && original == 1
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCtrad.png');
% % elseif filter == 1 && original == 0
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCnew.png');
% % else
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/CCnonFiltered.png');
% % end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure;
% % hold on;
% % PRSize = size(PR_K0);
% % 
% % for i = 1:agentsNumber
% %     plot(PR_K0(i).R, PR_K0(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% % end
% % legend show
% % xlabel('Recall');
% % ylabel('Precision');
% % xlim([0, 1]);
% % ylim([0.4, 1.02]);
% % hold off;
% % if filter == 1 && original == 1
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0trad.png');
% % elseif filter == 1 && original == 0
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0new.png');
% % else
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K0nonFiltered.png');
% % end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % figure;
% % hold on;
% % PRSize = size(PR_K5);
% % 
% % for i = 1:agentsNumber
% %     plot(PR_K5(i).R, PR_K5(i).P, 'color', rand(1,3), 'DisplayName', num2str(i))
% % end
% % legend show
% % xlabel('Recall');
% % ylabel('Precision');
% % xlim([0, 1]);
% % ylim([0.4, 1.02]);
% % hold off;
% % if filter == 1 && original == 1
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5trad.png');
% % elseif filter == 1 && original == 0
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5new.png');
% % else
% %     saveas(gcf, '/home/mamlpm/Documentos/TrabajoFinMaster/Results/Figuras/K5nonFiltered.png');
% % end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
