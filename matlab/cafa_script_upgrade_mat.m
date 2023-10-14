
d = pred_dir;

subdirs = dir(fullfile(d,'*'))
for i = 1:length(subdirs)
    subd = subdirs(i)
    
    d = [subd.folder, '/', subd.name]
    newd = [pred_dir, '/../upgraded_prediction/', subd.name]
    myFiles = dir(fullfile(d,'*.mat'));
    for k = 1:length(myFiles)
        baseFileName = myFiles(k).name;
        disp(baseFileName)
        
    end
end