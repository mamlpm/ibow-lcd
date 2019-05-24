% Script for obtaining the required results for IROS'18
clear all
close all
filter = 1;
original = 1;
distribuido = 1;

if distribuido
    if filter == 1 && original == 1
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtrados/distributed/';
    elseif filter == 1 && original == 0
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtradosNuevo/distributed/';
    else
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/noFiltrados/distributed/';
    end
else
    if filter == 1 && original == 1
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtrados/noDistributed/';
    elseif filter == 1 && original == 0
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/filtradosNuevo/noDistributed/';
    else
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/purgados/noFiltrados/noDistributed/';
    end
end
gt_neigh = 40;
compensate = false;

% Configuring subpaths
addpath('AcademicFigures/');
agentsNumber = 5;
step = 1;

curr_dir = strcat(base_dir, 'Lip6In/')
[PR_L6IPurg] = process(curr_dir, agentsNumber, 'Lip6In', 5, agentsNumber, gt_neigh, compensate, step);


% P/R curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
PRSize = size(PR_L6IPurg);

for i = 1:agentsNumber
    plot(PR_L6IPurg(i).R, PR_L6IPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i));
end
legend show
xlabel('Recall');
ylabel('Precision');
xlim([0, 1]);
ylim([0.4, 1.02]);
hold off;


if distribuido
    if filter == 1 && original == 1
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtrados/distributed/';
    elseif filter == 1 && original == 0
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtradosNuevo/distributed/';
    else
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/noFiltrados/distributed/';
    end
else
    if filter == 1 && original == 1
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtrados/noDistributed/';
    elseif filter == 1 && original == 0
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/filtradosNuevo/noDistributed/';
    else
        base_dir = '/home/mamlpm/Documentos/TrabajoFinMaster/Results/noPurgados/noFiltrados/noDistributed/';
    end
end

curr_dir = strcat(base_dir, 'Lip6In/')
[PR_L6InPurg] = process(curr_dir, agentsNumber, 'Lip6In', 5, agentsNumber, gt_neigh, compensate, step);

% P/R curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
hold on;
PRSize = size(PR_L6InPurg);

for i = 1:agentsNumber
    plot(PR_L6InPurg(i).R, PR_L6InPurg(i).P, 'color', rand(1,3), 'DisplayName', num2str(i));
end
legend show
xlabel('Recall');
ylabel('Precision');
xlim([0, 1]);
ylim([0.4, 1.02]);
hold off;
print('-depsc', strcat(base_dir, 'Lip6InPurg'));
