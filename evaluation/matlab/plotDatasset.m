clear all
close all
datasetName = 'K5';
coords_file = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets';
tmap_file = '/home/mamlpm/Documentos/TrabajoFinMaster/Results';
original = 0;
NumberofAgents = 3;

if strcmp(datasetName, 'K0')
    coords_file = strcat(coords_file, '/KITTI/00/imageCoords00.mat');
    if original
        tmap_file = strcat(tmap_file, '/purgados/filtrados/distributed/KITTI00/', int2str(NumberofAgents), '.txt');
    else
        tmap_file = strcat(tmap_file, '/purgados/filtradosNuevo/distributed/KITTI00/', int2str(3), '.txt');
    end
elseif strcmp(datasetName, 'K5')
    coords_file = strcat(coords_file, '/KITTI/05/imageCoords05.mat');
    if original
        tmap_file = strcat(tmap_file, '/purgados/filtrados/distributed/KITTI05/', int2str(NumberofAgents), '.txt');
    else
        tmap_file = strcat(tmap_file, '/purgados/filtradosNuevo/distributed/KITTI05/', int2str(3), '.txt');
    end
else
    coords_file = strcat(coords_file, '/CityCentre/ImageCollectionCoordinates.mat');
    if original
        tmap_file = strcat(tmap_file, '/purgados/filtrados/distributed/CityCentre/', int2str(NumberofAgents), '.txt');
    else
        tmap_file = strcat(tmap_file, '/purgados/filtradosNuevo/distributed/CityCentre/', int2str(3), '.txt');
    end
end

plot_topmap(tmap_file, coords_file, datasetName)