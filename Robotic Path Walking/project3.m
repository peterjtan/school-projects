%% Inpurting Data
clear all
close all
clc

input = csvread('TestApplication/bin/Debug/data.csv');

%% Moving Averaging
ws = 2;  % window size
color_avg = zeros(1, length(input) - ws-1);

for i = 1:length(input) - (ws-1)
    color_avg(i) = sum(input(i:i+ws-1,2))/ws;
end

% Top figure
subplot(2,1,1);
plot(color_avg,'r')
ylabel('Averaged Color Value')


%% Calculate Derivative with Absolute Value
color_der = abs(color_avg(2:end)-color_avg(1:end-1));

% Bottom figure
subplot(2,1,2);
plot(color_der,'b')


%% Finding Peaks
[Peaks_Y_Vals,Peaks_X_Vals] = findpeaks(color_der,'MINPEAKHEIGHT',1, 'MINPEAKDISTANCE',3);

hold on
plot(Peaks_X_Vals,Peaks_Y_Vals,'or')
xlabel('Smaple Number')
ylabel('Derivative')


%% Calculating Width and Color of Each Region
region = zeros(2,length(Peaks_X_Vals)-1);
% width
region(1,1:end) = Peaks_X_Vals(2:end) - Peaks_X_Vals(1:end-1);

% color (using midpoint light sensor value)
for i = 1:length(Peaks_X_Vals)-1
    region(2,i) = color_avg( round(sum(Peaks_X_Vals(i:i+1)/2)) );
end


%% Summarizing and Displaying Results
result = cell(2,length(region));
for i = 1:length(region)
    temp = round(region(1,i)/9);
    switch temp
        case 1
            result(1,i) = {'One'};
        case 2
            result(1,i) = {'Two'};
        case 3
            result(1,i) = {'Three'};
    end
    
    if region(2,i) > 29
        result(2,i) = {'Red'};
    elseif region(2,i) > 15
        result(2,i) = {'Yellow'};
    else
        result(2,i) = {'Green'};
    end
end

for i = 1:length(region)
    tempStr = sprintf('Region Number %d: Color is %s, Width is %s',...
              i,result{2,i},result{1,i});
    disp(tempStr);
end