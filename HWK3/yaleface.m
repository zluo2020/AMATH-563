%% Data loading
clear all; clc;
% convert '1' to '01' format
numFolder = cellstr(num2str((1:39).', '%02d'));

% container of image data
croppedData = [];
labels = [];
gender_labels = [];

% index stands for female in the database
female = [5 15 22 27 28 32 34 37];

for i = 1:39
    % directory path
    dir_to_search = strcat('H:\University of Washington\AMATH 563\HWK3\CroppedYale\yaleB',string(numFolder(i)));
    
    % dictionary of PGM file
    pgmpattern = fullfile(dir_to_search, '*.PGM');
    
    dinfo = dir(pgmpattern);
    
    for j = 1:length(dinfo)
        % file name
        pgmfile = fullfile(dir_to_search, dinfo(j).name);
        
        % use regular expression to match the file name
        label = str2num(regexp(dinfo(i).name,'\d+','match','once'));
        labels = [labels, label];

        % 1 for men, -1 for women
        if ismember(label, female)
            gender_labels = [gender_labels, 1];
        else
            gender_labels = [gender_labels, -1];
        end
        
        % read pgm file as image data
        data = imread(pgmfile);
        
        % reshape 192 * 168 image into 32256 * 1
        dataReshaped = reshape(data,[],1);
        
        % stack each reshaped image into columns
        croppedData = [croppedData, dataReshaped];
    end
        
end
%% Image Processing and SVD
% convert uint8 data into double
croppedData = double(croppedData);

% normalize data before SVD
avgface = mean(croppedData, 2);
A = croppedData-avgface*ones(1,size(croppedData,2));

% columns of u are eigenfaces, can be reshaped back into 192 * 168 images
[u,s,v] = svd(A, 'econ'); % perform SVD

% Sanity check the number should be equivalent to the number of images
numOfModes = nnz(diag(s));

%%
% Plot Energy vs. Rank
xaixs = 1:numOfModes;
figure(1)
semilogx(xaixs,100* diag(s)/sum(diag(s)), 'ko')
%plot(diag(s)/sum(diag(s)),'ko')
ylabel('Percent variability explained')
xlabel('Modes')
title('Variability explained by each column (mode)')

% plot cumulative energy vs. Rank
cum_variability = cumsum(100 * diag(s)/sum(diag(s)));
figure(2)
semilogx(xaixs,cum_variability)
ylabel('Cumulative Percent variability explained')
xlabel('Modes')
title('Cumulative variability explained by each column (mode)')
%% Construct sample data matrix (see train/test split as well)
% projection men vs women to 3D space
idx = 2;
P1 = croppedData(:,1:64);
P2 = croppedData(:,((idx - 1)*64 + 1):(idx*64));
P1 = P1 - avgface*ones(1,size(P1,2));
P2 = P2 - avgface*ones(1,size(P2,2));

PCAmodes = [4 5 6];
PCACoordsP1 = u(:,PCAmodes)'*P1;
PCACoordsP2 = u(:,PCAmodes)'*P2;

%% Projection plot
figure(3)
plot3(PCACoordsP1(1,:),PCACoordsP1(2,:),PCACoordsP1(3,:),'rx')
axis([-10000 10000 -10000 10000]), hold on, grid on
plot3(PCACoordsP2(1,:),PCACoordsP2(2,:),PCACoordsP2(3,:),'bx')
title('Project faces of two people into 3D space')
xlabel('mode 4')
ylabel('mode 5')
zlabel('mode 6')
%% Eigenface Interpretation
% Interpretation of several eigenfaces
figure(4)
subplot(2,2,1),imagesc(reshape(u(:,5),192,168))
subplot(2,2,2),imagesc(reshape(u(:,6),192,168))
subplot(2,2,3),imagesc(reshape(u(:,7),192,168))
subplot(2,2,4),imagesc(reshape(u(:,8),192,168))
%% Optimal hard Threshold
sigs = diag(s);
beta = size(A,2)/size(A,1); % aspect ratio of data matrix
thresh = optimal_SVHT_coef(beta,0) * median(sigs); % From Gavish & Donoho
thresh = round(thresh)
%% Cross-Validation and Train
accuracy_vec = [];
X = [v(:, 1:thresh)];
y = gender_labels';
num_of_trails = 10;

for jj = 1:num_of_trails;
    accuracy = train_LDA(X, y, 0.2);
    accuracy_vec = [accuracy_vec, accuracy];
end

figure(5)
bar(accuracy_vec);
hold on
%refline([0 mean(accuracy)])
hline = refline([0 mean(accuracy)]);
hline.Color = 'r';
hline.LineStyle = '--';
hline.LineWidth = 1.5;
ylim([0 100]);
xlim([0 11]);
xlabel('Individual Trial#');
ylabel('Accuracy');
title('Cross-validation using 100 number of eigenfaces')
%% Cross validation on number of feature selected
cv_plot(v, gender_labels, 1000, 10)
%% Train on Male/Female case
% repeat the process above by replace labels with gender_labels
%% Clustering problem
eva = evalclusters(X,'kmeans','CalinskiHarabasz','KList',[2:50])
%% Plot elbow 
figure(6)
plot(2:50, eva.CriterionValues)
xlabel('Number of clusters k')
ylabel('CalinskiHarabasz Index')
%% Silhouette
eva_Silhouette = evalclusters(X,'kmeans','Silhouette','KList',[2:50]);
figure(7)
plot(2:50, eva_Silhouette.CriterionValues)
xlabel('Number of clusters k')
ylabel('Silhouette Index')
%% Clustering on original data
eva_img = evalclusters(croppedData','kmeans','CalinskiHarabasz','KList',[2:50]);
figure(8)
plot(2:50, eva_img.CriterionValues)
xlabel('Number of clusters k')
ylabel('CalinskiHarabasz Index')
%% Train model
function [] = cv_plot(v, labels, thresh_range, num_of_trails)
    
    cv_score = [];
    thresh_vec = linspace(1, thresh_range, 20);
    for thresh = thresh_vec;
        X = [v(:, 1:round(thresh))];
        y = labels';
        accuracy_vec = [];

        for jj = 1:num_of_trails;
            accuracy = train_LDA(X, y, 0.2);
            accuracy_vec = [accuracy_vec, accuracy];
        end
        %accuracy = fit_multiSVM(X, y)
        cv_score = [cv_score, accuracy];
    end
    
    figure
    plot(thresh_vec, cv_score, '-','LineWidth',2)
    xlabel('Number of Modes used');
    ylabel('Accuracy');
        
end

function accuracy = train_LDA(X, y, split)
    % metadata matrix
    data = [X, y];

    % Cross validation (train: 80%, test: 20%)
    cv = cvpartition(size(data,1),'HoldOut',0.2);
    idx = cv.test;

    % Separate to training and test data
    dataTrain = data(~idx,:);
    dataTest  = data(idx,:);

    Xtrain = dataTrain(:,1:end - 1);
    ytrain = dataTrain(:,end);
    Xtest = dataTest(:,1:end - 1);
    ytest = dataTest(:,end);
    
    class = classify(Xtest,Xtrain,ytrain);
    result = ytest == class;
    accuracy = sum(result)/size(class,1) * 100;
end

function accuracy = fit_multiSVM(X, y)

    clf_svm = fitcecoc(X, y);
    
    CVclf_svm = crossval(clf_svm);
    
    genError = kfoldLoss(CVclf_svm);
    
    accuracy = (1-genError) * 100;
end