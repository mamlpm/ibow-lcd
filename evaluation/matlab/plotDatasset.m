clear all
close all
datasetName = 'K0';
coords_file = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets';
tmap_file = '/home/mamlpm/Documentos/TrabajoFinMaster/Results';
NumberofAgents = 25;

if strcmp(datasetName, 'K0')
    coords_file = strcat(coords_file, '/KITTI/00/imageCoords00.mat');
    tmap_file = strcat(tmap_file, '/purgados/filtrados/KITTI00/', int2str(NumberofAgents), '.txt');
elseif strcmp(datasetName, 'K5')
    coords_file = strcat(coords_file, '/KITTI/05/imageCoords05.mat');
    tmap_file = strcat(tmap_file, '/purgados/filtrados/KITTI05/', int2str(NumberofAgents), '.txt');
    tmap_file = strcat(tmap_file);
else strcmp(datasetName, 'CC')
    coords_file = strcat(coords_file, '/CityCentre/ImageCollectionCoordinates.mat');
    tmap_file = strcat(tmap_file, '/purgados/filtrados/CityCentre/', int2str(NumberofAgents), '.txt');
    tmap_file = strcat(tmap_file);
end

plot_topmap(tmap_file, coords_file, datasetName)