clc
clear

matrix = readmatrix('officetx.csv');
num_cols = size(matrix, 2);
exponent = size(num_cols);

for i = 1:num_cols
    column = matrix(2:end, i);
    data = reshape(column.', 1, []);
    
    result = hurstf(data);
    exponent(i) = result;
end

A = mean(exponent);
min = min(exponent);
max = max(exponent);

disp('Average')
disp(A)
disp('Min')
disp(min)
disp('Max')
disp(max)

boxchart(exponent)

