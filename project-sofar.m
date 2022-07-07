
%loading the dataset
data_inner = load(fullfile(matlabroot, 'toolbox', 'predmaint', ...
    'predmaintdemos', 'bearingFaultDiagnosis', ...
    'train_data', 'InnerRaceFault_vload_1.mat'));

% first plot of the data (inner race fault for ex.)

plotBearingSignalAndScalogram(data_inner)




%converts the signal into images we will use to train

fileLocation = fullfile('.', 'RollingElementBearingFaultDiagnosis-Data-master', 'train_data');
fileExtension = '.mat';
ensembleTrain = fileEnsembleDatastore(fileLocation, fileExtension);
ensembleTrain.ReadFcn = @readMFPTBearing;
ensembleTrain.DataVariables = ["gs", "sr", "rate", "load", "BPFO", "BPFI", "FTF", "BSF"];
ensembleTrain.ConditionVariables = ["Label", "FileName"];
ensembleTrain.SelectedVariables = ["gs", "sr", "rate", "load", "BPFO", "BPFI", "FTF", "BSF", "Label", "FileName"];

reset(ensembleTrain)
while hasdata(ensembleTrain)
  folderName = 'train_image';
  convertSignalToScalogram(ensembleTrain,folderName);
end
path = fullfile('.', folderName);
imds = imageDatastore(path, ...
  'IncludeSubfolders',true,'LabelSource','foldernames');





%converts the signal into images we will use to test our learning algorithm

fileLocation = fullfile('.', 'RollingElementBearingFaultDiagnosis-Data-master', 'test_data');
fileExtension = '.mat';
ensembleTest = fileEnsembleDatastore(fileLocation, fileExtension);
ensembleTest.ReadFcn = @readMFPTBearing;
ensembleTest.DataVariables = ["gs", "sr", "rate", "load", "BPFO", "BPFI", "FTF", "BSF"];
ensembleTest.ConditionVariables = ["Label", "FileName"];
ensembleTest.SelectedVariables = ["gs", "sr", "rate", "load", "BPFO", "BPFI", "FTF", "BSF", "Label", "FileName"];

reset(ensembleTest)
while hasdata(ensembleTest)
  folderName = 'test_image';
  convertSignalToScalogram(ensembleTest,folderName);
end
path = fullfile('.','test_image');
imdsTest = imageDatastore(path, ...
  'IncludeSubfolders',true,'LabelSource','foldernames');



% splitting these images into train and validate sets to check accuracy

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.67,'randomize');


% defining and training the convolutional neural network

net = squeezenet;
lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

newConvLayer = convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
newClassificationLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificationLayer);

analyzeNetwork(net)

options = trainingOptions('sgdm', ...
  'InitialLearnRate',0.0001, ...
  'MaxEpochs',4, ...
  'Shuffle','every-epoch', ...
  'ValidationData',imdsValidation, ...
  'ValidationFrequency',30, ...
  'Verbose',false, ...
  'MiniBatchSize',20, ...
  'Plots','training-progress');


%training network, calculating accuracy and classification matrix

net = trainNetwork(imdsTrain,lgraph,options);

label_predicts = classify(net,imdsTest);
label_actual = imdsTest.Labels;

accuracy = sum(label_predicts==label_actual)/numel(imdsTest);


figure
confusionchart(label_actual,label_predicts);






















%plotting function

function plotBearingSignalAndScalogram(data)
fs = data.bearing.sr;
t_total = 0.1; % seconds
n = round(t_total*fs);
bearing = data.bearing.gs(1:n);
[cfs,frq] = cwt(bearing,'amor', fs);

figure
subplot(2,1,1)
plot(0:1/fs:(n-1)/fs,bearing)
xlim([0,0.1])
title('Vibration Signal')
xlabel('Time (s)')
ylabel('Amplitude')
subplot(2,1,2)
surface(0:1/fs:(n-1)/fs,frq,abs(cfs))
shading flat
xlim([0,0.1])
ylim([0,max(frq)])
title('Scalogram')
xlabel('Time (s)')
ylabel('Frequency (Hz)')
end


% converts signals to scalogram, and saves it as an image

function convertSignalToScalogram(ensemble,folderName)

data = read(ensemble);
fs = data.sr;
x = data.gs{:};
label = char(data.Label);
fname = char(data.FileName);
ratio = 5000/97656;
interval = ratio*fs;
N = floor(numel(x)/interval);

path = fullfile('.',folderName,label);
if ~exist(path,'dir')
  mkdir(path);
end

for idx = 1:N
  sig = envelope(x(interval*(idx-1)+1:interval*idx));
  cfs = cwt(sig,'amor', seconds(1/fs));
  cfs = abs(cfs);
  img = ind2rgb(round(rescale(flip(cfs),0,255)),jet(320));
  outfname = fullfile('.',path,[fname '-' num2str(idx) '.jpg']);
  imwrite(imresize(img,[227,227]),outfname);
end
end